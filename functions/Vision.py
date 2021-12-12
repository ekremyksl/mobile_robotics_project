import cv2 as cv
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, GeometryCollection
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from itertools import combinations
import matplotlib.pyplot  as plt
from shapely.geometry.multipoint import MultiPoint
import cv2.aruco as aruco
import yaml
import time
from functions.a_star import dijkstra

class Vision:
    GROUND_X_RANGE_MM = 1450
    GROUND_Y_RANGE_MM = 700
    BINARIZATION_THRESHOLD = 60
    THYMIO_HEIGHT_MM = 65
    THYMIO_LENGTH_MM = 130
    THYMIO_WIDTH_MM = 130
    THYMIO_MARKER_DISTANCE_BACK_MM = 40

    def __init__(self, video_source:int=0):
        """
        Opens videocamera with ID 'video_source' or no stream if it is -1 and loads/parses the camera calibration file
        """
        
        if video_source == -1:
            # Go into passive mode
            self.vid = None
        else:
            # Open the camera and set resolution
            self.vid = cv.VideoCapture(video_source,cv.CAP_DSHOW)
            self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
            self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.obstacle = MultiPolygon()
        
        # Load camera calibration and check that it has the correct format
        with open("./Vision/calibration.yaml", "r") as calib_file:
            self.camera_calibration = yaml.load(calib_file)
        assert isinstance(self.camera_calibration, dict)
        assert set(self.camera_calibration.keys()) == {"distortion", "matrix"}
        self.camera_calibration["matrix"] = np.array(self.camera_calibration["matrix"])
        self.camera_calibration["distortion"] = np.array(self.camera_calibration["distortion"])

    def obstacleSegmentation(self, img:np.ndarray) -> np.ndarray:
        """
        Binarizes 'img'-RGB-image such that only obstacle objects are black (0) and the rest is white (255). The output is a grayscale image.
        """
        # Perform set of filters in 'continuous' grayscale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv.medianBlur(img, 5)
        img = cv.bilateralFilter(img, 15, 60, 150)
        # Binarize image
        _, img = cv.threshold(img,Vision.BINARIZATION_THRESHOLD,255,cv.THRESH_BINARY)
        # Perform morphological transformations on binary image
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((25,25),np.uint8))
        return cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((51,51),np.uint8))
        
    def acquireImg(self, undistort:bool=True, demo_img:np.ndarray=None)->np.ndarray:
        """
        Captures frame and optionally applies distortion from the loaded calibration.yaml. If a demo_img is passed to test the un-distort-functionality, it will be used in any case (also, if class member vid)
        """
        if demo_img is None:
            assert self.vid is not None
            ret, img = self.vid.read()
            if not ret:
                print("Error reading source!")
                return None
        else:
            img = demo_img
        if undistort:
            # Use calibration data to correct image
            return cv.undistort(img, self.camera_calibration["matrix"], self.camera_calibration["distortion"])
        else:
            return img

    def extractWarp(self, img:np.ndarray, remove_border_aruco:bool=True)->list[bool,np.ndarray,np.ndarray]:
        """
        Detect ArUco-markers 0,1,2,3 in 'img' and create a transformation matrix
        """
        # Extract markers' positions and IDs
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # Check if border-markers are visible
        if ids is None or 0 not in ids or 1 not in ids or 2 not in ids or 3 not in ids:
            print("Could not find all 4 boundary markers")
            return False, None, None
        # Calculate markers' centroids
        corner_dict = {int(id): [sum([corner[0] for corner in cornerset[0]]) / len(cornerset[0]), sum([corner[1] for corner in cornerset[0]]) / len(cornerset[0])] for (id, cornerset) in zip(ids, corners)}
        corner_pts = []
        for i in range(4):
            corner_pts.append(corner_dict[i])
        corner_pts = np.asarray(corner_pts).astype(int)
        # Obtain 3x3 transformation matrix
        warp_transform = cv.getPerspectiveTransform(
                corner_pts.astype(np.float32),
                np.float32([[0,Vision.GROUND_Y_RANGE_MM], [Vision.GROUND_X_RANGE_MM,Vision.GROUND_Y_RANGE_MM],[Vision.GROUND_X_RANGE_MM,0], [0,0]])
            )
        if remove_border_aruco:
            # Draw white polygon over the ArUco markers
            for corner, id in zip(corners, ids):
                if id == 0 or id == 1 or id == 2 or id == 3:
                    cv.fillConvexPoly(img, corner[0].astype(int), (255, 255, 255))
        return True, img, warp_transform

    def applyWarp(self, img:np.ndarray, warp_transform:np.ndarray)->np.ndarray:
        """
        Applies given 3x3-matrix 'warp_transform' to 'img'
        """
        return cv.warpPerspective(img,warp_transform,(Vision.GROUND_X_RANGE_MM, Vision.GROUND_Y_RANGE_MM))

    

    def findThymio(self, img, marker_id:int = 4, remove_thymio:str=None)->list[bool,np.ndarray,list[float]]:
        """
        Extract Thymio's pose (x,y,yaw) in mm/degree in image 'img' and overdraw its footprint or ArUco marker optionally. Resturns success-flag, overdrawn image and 3-element pose-list.
        """
        # Extract markers' positions and IDs
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # Check if Thymio-marker is visible
        if ids is not None and len(np.where(ids==marker_id)[0]) == 1:
            aruco_corners = corners[np.where(ids==marker_id)[0][0]][0]
        else:
            print("Cannot identify Thymio in image!",ids)
            return False, None, None
        # Calculate marker's centroid
        thymio_pose = [sum([corner[0] for corner in aruco_corners]) / len(aruco_corners), Vision.GROUND_Y_RANGE_MM - sum([corner[1] for corner in aruco_corners]) / len(aruco_corners)]
        # Calculate marker's orientation in the image = marker's orientation in the world frame
        driving_direction = aruco_corners[1] - aruco_corners[0]
        thymio_pose.append(np.rad2deg(np.pi - np.arctan2(driving_direction[1], driving_direction[0])))
        
        if remove_thymio == "marker":
            # Draw white polygon over the ArUco marker
            cv.fillConvexPoly(img, np.array(aruco_corners).astype(int), (255, 255, 255))
        elif remove_thymio == "full":
            # Draw white polygon generously over the Thymio robot
            driving_dir = np.array([np.cos(np.deg2rad(thymio_pose[2])), -np.sin(np.deg2rad(thymio_pose[2]))])
            perpendicular_dir = np.array([-np.sin(np.deg2rad(thymio_pose[2])), -np.cos(np.deg2rad(thymio_pose[2]))])
            thymio_position = np.array(thymio_pose[0:2])
            thymio_position[1] = Vision.GROUND_Y_RANGE_MM - thymio_position[1]
            thymio_corners = [
                thymio_position + (Vision.THYMIO_LENGTH_MM - Vision.THYMIO_MARKER_DISTANCE_BACK_MM) * driving_dir + Vision.THYMIO_WIDTH_MM / 2 * perpendicular_dir,
                thymio_position + (Vision.THYMIO_LENGTH_MM - Vision.THYMIO_MARKER_DISTANCE_BACK_MM) * driving_dir - Vision.THYMIO_WIDTH_MM / 2 * perpendicular_dir,
                thymio_position - Vision.THYMIO_MARKER_DISTANCE_BACK_MM * driving_dir - Vision.THYMIO_WIDTH_MM / 2 * perpendicular_dir,
                thymio_position - Vision.THYMIO_MARKER_DISTANCE_BACK_MM * driving_dir + Vision.THYMIO_WIDTH_MM / 2 * perpendicular_dir,
            ]
            cv.fillConvexPoly(img, np.array(thymio_corners).astype(int), (255, 255, 255))
        return True, img, thymio_pose

    def findGoal(self, img, marker_id:int = 5, remove_border_aruco:bool=True, )->list[bool,np.ndarray,list[float]]:
        """
        Extract goal's position (x,y) in mm in image 'img' and overdraw its footprint or ArUco marker optionally. Resturns success-flag, overdrawn image and 3-element pose-list.
        """
        # Extract markers' positions and IDs
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # Check if goal-marker is visible
        if ids is not None and len(np.where(ids==marker_id)[0]) == 1:
            goal_borders = corners[np.where(ids==marker_id)[0][0]][0]
        else:
            print("Cannot identify goal in image!",ids)
            return False, None, None
        if remove_border_aruco:
            cv.fillConvexPoly(img, np.array(goal_borders).astype(int), (255, 255, 255))
        # Calculate marker's centroid
        return True, img,  [sum([corner[0] for corner in goal_borders]) / len(goal_borders), Vision.GROUND_Y_RANGE_MM - sum([corner[1] for corner in goal_borders]) / len(goal_borders)]
        

    def getContourPolygons(self,img:np.ndarray, buffer_mm:int=100) -> list[Polygon]:
        """
        Creates shapely-polygons from binarized opencv-image 'img' through Deuglas-Peucker algorithm and adds expansion of 'buffer_mm'.
        """
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        polygons = []
        # Start at second contour since the first one is the border of the image
        for contour in contours[1:]:
            # Douglas Peucker-approximation
            epsilon = 0.01 * cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            # Conversion to shapely and expansion
            polygons.append(Polygon(list(map(lambda x:(int(x[0][0]), int(Vision.GROUND_Y_RANGE_MM - x[0][1])), approx))).buffer(buffer_mm, join_style=JOIN_STYLE.mitre, cap_style=CAP_STYLE.round))
        return polygons

    def visibilityGraph(self,polygons:list[Polygon],reset_obstacle_map:bool=True,thymio_pose=None,goal_pose=None) -> list[np.ndarray, np.ndarray]:
        """
        Creates from list polygons representing obstacles collision-free visibility graph. If provided, the graph adds the 'thymio_pose' as the first node and 'goal_pose' as the last node.
        From this, it returns an adjacency matrix and a list of coordinate points, where the i.th entry corresponds to the ith row/column of the adjacency matrix.
        """
        if reset_obstacle_map: self.obstacle = MultiPolygon()
        potential_wp = []
        # Create union of all obstacle polygons and extract their corners
        for obstacle_polygon in polygons: 
            self.obstacle = self.obstacle.union(obstacle_polygon)
            new_wps = list(np.asarray(np.dstack(tuple(obstacle_polygon.exterior.xy)))[0])
            for new_wp in new_wps:
                # Onyl accept corner if it lies inside of the range
                if new_wp[0] > 0 and new_wp[0] < Vision.GROUND_X_RANGE_MM and new_wp[1] > 0  and new_wp[1] < Vision.GROUND_Y_RANGE_MM:
                    potential_wp.append(new_wp)
        
        # Add Thymio/goal positions      
        if goal_pose is not None:
            potential_wp =  potential_wp + [np.asarray(goal_pose[:2])]
        if thymio_pose is not None:
            potential_wp = [np.asarray(thymio_pose[:2])] + potential_wp
        
        adjacency_matrix = np.zeros((len(potential_wp), len(potential_wp)))
        
        # Iterate through every possible combination of graph nodes
        for indices, line_candidate in zip(combinations(range(len(potential_wp)),2), combinations(potential_wp,2)):
            if indices[0] == indices[1]:
                # No self-loops!
                continue
            # Create test-line and perform line-test
            test_line = LineString([line_candidate[0], line_candidate[1]])
            inter = test_line.intersection(self.obstacle)
            if (inter == MultiPoint([line_candidate[0], line_candidate[1]]) \
                or inter==MultiPoint([line_candidate[1], line_candidate[0]]))\
                    or ((inter == test_line or inter==LineString([line_candidate[1], line_candidate[0]])) and not test_line.within(self.obstacle))\
                        or inter==Point(line_candidate[1]) or inter==Point(line_candidate[0]) or inter==GeometryCollection() or inter==LineString():
                adjacency_matrix[indices[0], indices[1]] = 1
        # Graph is undirected, so the lower triangular component of the matrix is easily obtained
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        return adjacency_matrix, potential_wp

    def prepareForVisualization(self,img:np.ndarray, visibility_graph = None, optimal_path=None,polygons:list[Polygon]=None,thymio_pose:list=None,goal_pos:list=None)->np.ndarray:
        """
        Helper method that draws the 'visibility_graph', the 'optimal_path', obstacle-'polygons', the 'thymio_pose' and the 'goal_pos' if they are provided into image 'img'.
        """
        out = img.copy()
        # Write text detailing the goal/Thymio-pose
        if thymio_pose is not None:
            cv.putText(out, "Thymio - x: {}mm, y: {}mm, yaw: {}deg".format(*[int(p) for p in thymio_pose]), (1,45), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        if goal_pos is not None:
            cv.putText(out, "Goal - x: {}mm, y: {}mm".format(*[int(p) for p in goal_pos]), (1,Vision.GROUND_Y_RANGE_MM - 45), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        out = cv.flip(out,0)
        # Draw obstacles        
        if polygons is not None:
            for polygon in polygons:
                cv.polylines(out, [np.asarray(np.dstack(polygon.exterior.xy).astype(int))], False, (0,0,255),2)
        # Draw visibility graph
        if visibility_graph is not None:
            red_adj_mat = np.triu(visibility_graph[0])           
            (num_points,_) = np.shape(red_adj_mat)
            for start_point_index in range(num_points):
                for end_point_index in range(num_points):
                    if red_adj_mat[start_point_index][end_point_index]:
                        line = LineString([visibility_graph[1][start_point_index], visibility_graph[1][end_point_index]])
                        cv.polylines(out, [np.asarray(np.dstack(line.xy).astype(int))], False, (0,255,255),1)
        # Draw visibility graph
        if optimal_path is not None:
            cv.polylines(out, [np.asarray(np.dstack(LineString(optimal_path).xy).astype(int))], False, (0,255,0),2)
        # Draw cross on goal/Thymio-position (this done last so that as the visibility graph should not overlap this)
        if thymio_pose is not None:
            out = cv.drawMarker(out, np.array(thymio_pose[:2]).astype(int), (255,0,0), cv.MARKER_CROSS, 20, 2)
        if goal_pos is not None:
            out = cv.drawMarker(out, np.array(goal_pos[:2]).astype(int), (255,0,255), cv.MARKER_CROSS, 20, 2)
        return cv.flip(out,0)


    def getTrajectory(self, buffer_mm:int=150)->np.ndarray:
        """
        Performs complete vision pipeline and yields list of waypoints of optimal path (in mm).
        """
        while True:
            # Empty pipeline and ignore bad images
            for _ in range(5):
                img_orig = self.acquireImg(True)
            img_orig = cv.resize(img_orig, (1920, 1080))
            ret, img, warp = self.extractWarp(img_orig)
            if not ret:
                continue
            img = self.applyWarp(img, warp)
            ret, img, thymio_pose = self.findThymio(img, 4, remove_thymio="marker")
            if not ret:
                continue
            ret, img, goal_position = self.findGoal(img)
            if not ret:
                continue
            img_seg = self.obstacleSegmentation(img)
            polygons = self.getContourPolygons(img_seg, buffer_mm = buffer_mm)
            adj_matrix,polypoints = self.visibilityGraph(polygons, thymio_pose=thymio_pose, goal_pose=goal_position)
            try:
                adj_matrix1 = np.copy(adj_matrix)
                polypoints1 = np.copy(polypoints)
                points = dijkstra(adj_matrix1, polypoints1)
            except Exception as e:
                print("Could not find optimal path!")
                continue
            return points
    
    def getThymioPose(self)->list[float]:
        """
        Performs complete pipeline to extract Thymio's pose and gives it in cm/degree (x,y,yaw)
        """
        while True:
            for _ in range(5):
                img_orig = self.acquireImg(True)
            img_orig = cv.resize(img_orig, (1920, 1080))
            ret, img, warp = self.extractWarp(img_orig)
            if not ret:
                continue
            img = self.applyWarp(img, warp)
            ret, _, pos = self.findThymio(img, 4, remove_thymio="marker")
            if ret:
                return [pos[0] / 10, pos[1] / 10, pos[2]], img


if __name__ == "__main__":
    # Used for calibration purposes: Run with test-obstacle and adjust class-constant "BINARIZATION_THRESHOLD" until result is sufficient
    v = Vision(1)
    while True:
        img_orig = v.acquireImg(True)
        img_orig = cv.resize(img_orig, (1920, 1080))
        ret, img, warp = v.extractWarp(img_orig)
        if not ret:
            continue
        ret, img, pos = v.findThymio(img, 4, remove_thymio="marker")
        if not ret:
            continue
        ret, img, pos_g = v.findGoal(img)
        if not ret:
            continue
        img_2 = v.obstacleSegmentation(img)
        polygons = v.getContourPolygons(img_2, buffer_mm = 100)
        adj_matrix,polypoints = v.visibilityGraph(polygons, thymio_pose=pos, goal_pose=pos_g)        
        try:
            adj_matrix1 = np.copy(adj_matrix)
            polypoints1 = np.copy(polypoints)
            points = dijkstra(adj_matrix1, polypoints1)
        except Exception as e:
            print("Could not find optimal path!")
            points = None
        view_1 = v.prepareForVisualization(cv.cvtColor(img_2, cv.COLOR_GRAY2BGR), thymio_pose=pos, goal_pos=pos_g,polygons=polygons,visibility_graph=[adj_matrix,polypoints],optimal_path=points)
        view_2 = v.prepareForVisualization(img, thymio_pose=pos, goal_pos=pos_g,polygons=polygons,visibility_graph=[adj_matrix,polypoints],optimal_path=points)
        cv.imshow("Threshold calibration - Press 'q' to quit and 's' to store current result", np.hstack([view_1, view_2]))
        ret = cv.waitKey(10)
        if ret== ord("q"):
            break
        elif ret == ord("s"):
            time_now = time.time()
            cv.imwrite("img_orig_{}.png".format(time_now), img_orig)
            cv.imwrite("img_warped_{}.png".format(time_now), img)
            cv.imwrite("img_processed_{}.png".format(time_now), img_2)
            cv.imwrite("img_processed_{}.png".format(time_now), img_2)
            cv.imwrite("img_drawn_{}.png".format(time_now), view_2)