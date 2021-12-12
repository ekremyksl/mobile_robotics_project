import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.shape_base import vstack
from numpy.testing import verbose
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, linestring, GeometryCollection
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from itertools import combinations
import matplotlib.pyplot  as plt
from shapely.geometry.multipoint import MultiPoint
import cv2.aruco as aruco
import yaml
import os
import time
from a_star import dijkstra
class Vision:
    GROUND_X_RANGE_MM = 1600
    GROUND_Y_RANGE_MM = 1200
    BINARIZATION_THRESHOLD = 50
    THYMIO_HEIGHT_MM = 65
    THYMIO_LENGTH_MM = 130
    THYMIO_WIDTH_MM = 130
    THYMIO_MARKER_DISTANCE_BACK_MM = 40

    AUTOTUNE_THRESHOLD_AREA = 10000
    AUTOTUNE_THRESHOLD_STEP_SIZE = 10

    def __init__(self, video_source:int=0):
        #self.vid = cv.VideoCapture(0)
        self.vid = cv.VideoCapture(video_source,cv.CAP_DSHOW)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


        self.threshold = Vision.BINARIZATION_THRESHOLD
        self.obstacle = MultiPolygon()
        self.warp_transform = None
        with open("D:\EPFL\Basics of Mobile Robotics\mobile_robotics_project\motion_control\calibration.yaml", "r") as calib_file:
            self.camera_calibration = yaml.load(calib_file)
        assert isinstance(self.camera_calibration, dict)
        assert set(self.camera_calibration.keys()) == {"distortion", "matrix"}
        self.camera_calibration["matrix"] = np.array(self.camera_calibration["matrix"])
        self.camera_calibration["distortion"] = np.array(self.camera_calibration["distortion"])

    def applyPreprocessing(self, src:np.ndarray) -> np.ndarray:
        img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv.medianBlur(img, 5)
        img = cv.bilateralFilter(img, 15, 60, 150)
        ret, img = cv.threshold(img,self.threshold,255,cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((25,25),np.uint8))
        return cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((51,51),np.uint8))
        
    def acquireImg(self, undistort:bool=True)->np.ndarray:
        ret, img = self.vid.read()
        if not ret:
            print("Error reading source!")

            return None
        else:
            if undistort:
                return cv.undistort(img, self.camera_calibration["matrix"], self.camera_calibration["distortion"])
            else:
                return img

    def extractWarp(self, img:np.ndarray, remove_border_aruco:bool=True)->list[bool,np.ndarray,np.ndarray]:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None or 0 not in ids or 1 not in ids or 2 not in ids or 3 not in ids:
            print("Could not find all 4 boundary markers")
            return False, None, None
        corner_dict = {int(id): [sum([corner[0] for corner in cornerset[0]]) / len(cornerset[0]), sum([corner[1] for corner in cornerset[0]]) / len(cornerset[0])] for (id, cornerset) in zip(ids, corners)}
        corner_pts = []
        for i in range(4):
            corner_pts.append(corner_dict[i])
        corner_pts = np.asarray(corner_pts).astype(int)
        w = int(max(corner_dict.values(), key=lambda x: x[0])[0] - min(corner_dict.values(), key=lambda x: x[0])[0])
        h = int(Vision.GROUND_Y_RANGE_MM * w / Vision.GROUND_X_RANGE_MM)
        warp_transform = cv.getPerspectiveTransform(
                corner_pts.astype(np.float32),
                np.float32([[0,Vision.GROUND_Y_RANGE_MM], [Vision.GROUND_X_RANGE_MM,Vision.GROUND_Y_RANGE_MM],[Vision.GROUND_X_RANGE_MM,0], [0,0]])
            )
        if remove_border_aruco:
            for corner, id in zip(corners, ids):
                if id == 0 or id == 1 or id == 2 or id == 3:
                    cv.fillConvexPoly(img, corner[0].astype(int), (255, 255, 255))
        return True, img, warp_transform

    def applyWarp(self, img:np.ndarray, warp_transform:np.ndarray)->np.ndarray:
        return cv.warpPerspective(img,warp_transform,(Vision.GROUND_X_RANGE_MM, Vision.GROUND_Y_RANGE_MM))

    

    def findThymio(self, img, marker_id = 4, remove_thymio=None):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and len(np.where(ids==marker_id)[0]) == 1:
            aruco_corners = corners[np.where(ids==marker_id)[0][0]][0]
        else:
            print("Cannot identify Thymio in image!",ids)
            return False, None, None
        

        thymio_pose = [sum([corner[0] for corner in aruco_corners]) / len(aruco_corners), Vision.GROUND_Y_RANGE_MM - sum([corner[1] for corner in aruco_corners]) / len(aruco_corners)]
        #thymio_pose[1] = Vision.GROUND_Y_RANGE_MM - thymio_pose[1]
        driving_direction = aruco_corners[1] - aruco_corners[0]
        thymio_pose.append(np.rad2deg(np.pi - np.arctan2(driving_direction[1], driving_direction[0])))
        if remove_thymio == "marker":
            cv.fillConvexPoly(img, np.array(aruco_corners).astype(int), (255, 255, 255))
            print(aruco_corners)
        elif remove_thymio == "full":
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

    def findGoal(self, img, marker_id = 5, remove_border_aruco=True):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and len(np.where(ids==marker_id)[0]) == 1:
            goal_borders = corners[np.where(ids==marker_id)[0][0]][0]
        else:
            print("Cannot identify goal in image!",ids)
            return False, None, None
        if remove_border_aruco:
            cv.fillConvexPoly(img, np.array(goal_borders).astype(int), (255, 255, 255))
        
        return True, img,  [sum([corner[0] for corner in goal_borders]) / len(goal_borders), Vision.GROUND_Y_RANGE_MM - sum([corner[1] for corner in goal_borders]) / len(goal_borders)]
        

    def getContourPolygons(self,src:np.ndarray, buffer_mm:int=100) -> list[Polygon]:
        contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        polygons = []
        for contour in contours[1:]:
            epsilon = 0.01 * cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            polygons.append(Polygon(list(map(lambda x:(int(x[0][0]), int(Vision.GROUND_Y_RANGE_MM - x[0][1])), approx))).buffer(buffer_mm, join_style=JOIN_STYLE.mitre, cap_style=CAP_STYLE.round))
        return polygons

    def visibilityGraph(self,polygons:list[Polygon],reset_obstacle_map:bool=True,thymio_pose=None,goal_pose=None) -> list[LineString]:
        if reset_obstacle_map: self.obstacle = MultiPolygon()
        potential_wp = []
        potential_segments = []
        for obstacle_polygon in polygons: 
            self.obstacle = self.obstacle.union(obstacle_polygon)
            new_wps = list(np.asarray(np.dstack(tuple(obstacle_polygon.exterior.xy)))[0])
            for new_wp in new_wps:
                if new_wp[0] > 0 and new_wp[0] < Vision.GROUND_X_RANGE_MM and new_wp[1] > 0  and new_wp[1] < Vision.GROUND_Y_RANGE_MM:
                    potential_wp.append(new_wp)
        if goal_pose is not None:
            potential_wp =  potential_wp + [np.asarray(goal_pose[:2])]
        if thymio_pose is not None:
            potential_wp = [np.asarray(thymio_pose[:2])] + potential_wp
        
        adjacency_matrix = np.zeros((len(potential_wp), len(potential_wp)))
        for indices, line_candidate in zip(combinations(range(len(potential_wp)),2), combinations(potential_wp,2)):
            if indices[0] == indices[1]:
                # No self-loops!
                continue
            test_line = LineString([line_candidate[0], line_candidate[1]])
            inter = test_line.intersection(self.obstacle)
            
            
            self.obstacle
            if (inter == MultiPoint([line_candidate[0], line_candidate[1]]) \
                or inter==MultiPoint([line_candidate[1], line_candidate[0]]))\
                    or ((inter == test_line or inter==LineString([line_candidate[1], line_candidate[0]])) and not test_line.within(self.obstacle))\
                        or inter==Point(line_candidate[1]) or inter==Point(line_candidate[0]) or inter==GeometryCollection() or inter==LineString():
                potential_segments.append(test_line)
                adjacency_matrix[indices[0], indices[1]] = 1
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        
        return potential_segments, adjacency_matrix, potential_wp

    def prepareForVisualization(self,img:np.ndarray, visibility_graph = None, optimal_path=None,polygons:list[Polygon]=None,thymio_pose:list=None,goal_pos:list=None)->np.ndarray:
        out = img.copy()
        if thymio_pose is not None:
            cv.putText(out, "Thymio - x: {}mm, y: {}mm, yaw: {}deg".format(*[int(p) for p in thymio_pose]), (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if goal_pos is not None:
            cv.putText(out, "Goal - x: {}mm, y: {}mm".format(*[int(p) for p in thymio_pose]), (1,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        out = cv.flip(out,0)
                    
        if polygons is not None:
            for polygon in polygons:
                cv.polylines(out, [np.asarray(np.dstack(polygon.exterior.xy).astype(int))], False, (0,0,255),2)
        if visibility_graph is not None:
            red_adj_mat = np.triu(visibility_graph[0])
            
            (num_points,_) = np.shape(red_adj_mat)

            for start_point_index in range(num_points):
                for end_point_index in range(num_points):
                    if red_adj_mat[start_point_index][end_point_index]:
                        line = LineString([visibility_graph[1][start_point_index], visibility_graph[1][end_point_index]])
                        cv.polylines(out, [np.asarray(np.dstack(line.xy).astype(int))], False, (0,255,255),1)
        if optimal_path is not None:
            cv.polylines(out, [np.asarray(np.dstack(LineString(optimal_path).xy).astype(int))], False, (0,255,0),2)
        if thymio_pose is not None:
            out = cv.drawMarker(out, np.array(thymio_pose[:2]).astype(int), (255,0,0), cv.MARKER_CROSS, 20, 2)
        if goal_pos is not None:
            out = cv.drawMarker(out, np.array(goal_pos[:2]).astype(int), (255,0,255), cv.MARKER_CROSS, 20, 2)
        return cv.flip(out,0)


    def autoTuneThreshold(self, verbose:bool=False)-> bool:
        img = self.acquireImg()
        print(type(img))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        roi = cv.selectROI("Select calibration shape (a bit generously)", img)
        target_polygon = Polygon([(roi[0],roi[1]),(roi[0], roi[1]+roi[3]),(roi[0]+roi[2], roi[1]+roi[3]),(roi[0]+roi[2], roi[1])])
        
        
        lower_threshold = 0
        self.threshold = 0
        while self.threshold < 255:
            correct_counter = 0
            for i in range(20):
                img_raw = self.acquireImg()
                img = self.applyPreprocessing(img_raw)
                polygons = self.getContourPolygons(img, False)
                if len(polygons) == 1 and polygons[0].within(target_polygon) and target_polygon.difference(polygons[0]).area < Vision.AUTOTUNE_THRESHOLD_AREA:
                    correct_counter += 1
            if verbose:
                print("counter {}/20 at threshold {}".format(correct_counter, self.threshold))
            if correct_counter >= 19 and lower_threshold == 0:
                lower_threshold = self.threshold
            elif correct_counter < 19 and lower_threshold != 0:
                self.threshold = (lower_threshold + self.threshold) / 2
                return True
            self.threshold = self.threshold + Vision.AUTOTUNE_THRESHOLD_STEP_SIZE
            if verbose:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                img = cv.putText(np.vstack([img_raw, img]), "Lower threshold: {}".format(lower_threshold), (10,50), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,0))
                cv.imshow("out",cv.resize(img,(480,320)))
                cv.waitKey(10)
        cv.destroyAllWindows()
        return True

    def getTrajectory(self):
        while True:
            # Empty pipeline and ignore bad images
            for _ in range(5):
                img_orig = self.acquireImg(True)
            img_orig = cv.resize(img_orig, (1920, 1080))
            ret, img, warp = self.extractWarp(img_orig)
            if not ret:
                continue
            img = self.applyWarp(img, warp)
            ret, img, pos = self.findThymio(img, 4, remove_thymio="marker")
            if not ret:
                print("No thymio")
                continue
            ret, img, pos_g = self.findGoal(img)
            if not ret:
                print("No goal")
                continue
            img_2 = self.applyPreprocessing(img)
            polygons = self.getContourPolygons(img_2, buffer_mm = 150)
            _,adj_matrix,polypoints = self.visibilityGraph(polygons, thymio_pose=pos, goal_pose=pos_g)
            try:
                adj_matrix1 = np.copy(adj_matrix)
                polypoints1 = np.copy(polypoints)
                points = dijkstra(adj_matrix1, polypoints1)
            except Exception as e:
                print("Could not find optimal path!")
                continue
            return points
    
    def getThymioPose(self):
        for _ in range(5):
            for _ in range(5):
                img_orig = self.acquireImg(True)
            img_orig = cv.resize(img_orig, (1920, 1080))
            ret, img, warp = self.extractWarp(img_orig)
            if not ret:
                print("Could not find all 4 markers")
                continue
            img = self.applyWarp(img, warp)
            ret, _, pos = self.findThymio(img, 4, remove_thymio="marker")
            if ret:
                return [pos[0] / 10, pos[1] / 10, pos[2]], img
        return False, False

if __name__ == "__main__":
    v = Vision(1)
    for i in range(10):
        v.acquireImg()
        time.sleep(0.3)
    # v.autoTuneThreshold(verbose=True)
    
    print("AT")
    while True:
        img_orig = v.acquireImg(True)
        img_orig = cv.resize(img_orig, (1920, 1080))
        
        ret, img, warp = v.extractWarp(img_orig)
        if not ret:
            continue

        img = v.applyWarp(img, warp)
        
  
        ret, img, pos = v.findThymio(img, 4, remove_thymio="marker")
        if not ret:
            continue
        ret, img, pos_g = v.findGoal(img)
        if not ret:
            continue
        
        

        
        #print(pos)
        
        img_2 = v.applyPreprocessing(img)
        polygons = v.getContourPolygons(img_2, buffer_mm = 120)
        potential_segments,adj_matrix,polypoints = v.visibilityGraph(polygons, thymio_pose=pos, goal_pose=pos_g)
        
        try:
            adj_matrix1 = np.copy(adj_matrix)
            polypoints1 = np.copy(polypoints)
            points = dijkstra(adj_matrix1, polypoints1)
        except Exception as e:
            print("Could not find optimal path!")
            points = None

        view_1 = v.prepareForVisualization(cv.cvtColor(img_2, cv.COLOR_GRAY2BGR), thymio_pose=pos, goal_pos=pos_g,polygons=polygons,visibility_graph=[adj_matrix,polypoints],optimal_path=points)
        view_2 = v.prepareForVisualization(img, thymio_pose=pos, goal_pos=pos_g,polygons=polygons,visibility_graph=[adj_matrix,polypoints],optimal_path=points)
        cv.imshow("TEST", np.hstack([view_2]))
        ret = cv.waitKey(10)
        input(img.shape)
        if ret== ord("q"):
            break

        elif ret == ord("s"):
            time_now = time.time()
            cv.imwrite("img_orig_{}.png".format(time_now), img_orig)
            cv.imwrite("img_warped_{}.png".format(time_now), img)
            cv.imwrite("img_processed_{}.png".format(time_now), img_2)
            cv.imwrite("img_processed_{}.png".format(time_now), img_2)
            cv.imwrite("img_drawn_{}.png".format(time_now), view_2)
        continue
        img = cv.resize(img, (Vision.GROUND_X_RANGE_MM, Vision.GROUND_Y_RANGE_MM))
        img = cv.flip(img,0)
        plt.imshow(img)
        for polygon in polygons:
            plt.plot(*polygon.exterior.xy,'r',2)
        for potential_segment in potential_segments:
            plt.plot(*potential_segment.xy,'b',1)
        plt.plot(*LineString(points).xy,'r',linewidth=10)
        plt.axis("equal")
        cv.putText(img, "Thymio - x: {}mm, y: {}mm, yaw: {}deg".format(*[int(p) for p in pos]), (5,50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.imshow("IMAGE", img)
        
        plt.show()
        


