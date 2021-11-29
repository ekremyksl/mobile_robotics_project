import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.shape_base import vstack
from numpy.testing import verbose
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from itertools import combinations
import matplotlib.pyplot  as plt
from shapely.geometry.multipoint import MultiPoint
import cv2.aruco as aruco
import yaml
import os
import time
from scipy import sparse

class Vision:
    AUTOTUNE_THRESHOLD_AREA = 10000
    AUTOTUNE_THRESHOLD_STEP_SIZE = 10
    GROUND_X_RANGE_MM = 500
    GROUND_Y_RANGE_MM = 700

    def __init__(self, video_source=0):
        #self.vid = cv.VideoCapture(0)
        self.vid = cv.VideoCapture(0,cv.CAP_DSHOW)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        self.threshold = 30
        self.obstacle = MultiPolygon()
        self.warp_transform = None
        with open("calibration.yaml", "r") as calib_file:
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
        return cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((25,25),np.uint8))

    def acquireImg(self, undistort=True):
        ret, img = self.vid.read()
        if not ret:
            print("Error reading source!")

            return None
        else:
            if undistort:
                return cv.undistort(img, self.camera_calibration["matrix"], self.camera_calibration["distortion"])
            else:
                return img

    def extractWarp(self, img, markersize=4, remove_border_aruco=True):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None or 0 not in ids or 1 not in ids or 2 not in ids or 3 not in ids:
            print("Could not find all 4 boundary markers")
            return False, None
        corner_dict = {int(id): [sum([corner[0] for corner in cornerset[0]]) / len(cornerset[0]), sum([corner[1] for corner in cornerset[0]]) / len(cornerset[0])] for (id, cornerset) in zip(ids, corners)}
        corner_pts = []
        for i in range(4):
            corner_pts.append(corner_dict[i])
        corner_pts = np.asarray(corner_pts).astype(int)
        w = int(max(corner_dict.values(), key=lambda x: x[0])[0] - min(corner_dict.values(), key=lambda x: x[0])[0])
        h = int(Vision.GROUND_Y_RANGE_MM * w / Vision.GROUND_X_RANGE_MM)
        self.warp_transform = {"tf": cv.getPerspectiveTransform(
                corner_pts.astype(np.float32),
                np.float32([[0,h], [w,h],[w,0], [0,0]])
            ), "wh":(w,h)}
        if remove_border_aruco:
            for corner, id in zip(corners, ids):
                if id == 0 or id == 1 or id == 2 or id == 3:
                    cv.fillConvexPoly(img, corner[0].astype(int), (255, 255, 255))
        return True, img

    def applyWarp(self, img):
        assert self.warp_transform is not None
        return cv.warpPerspective(img,self.warp_transform["tf"],self.warp_transform["wh"])

    def transformPixelXY(self, coords, pixel_to_xy):
        if pixel_to_xy:
            return [
                coords[0] * Vision.GROUND_X_RANGE_MM / self.warp_transform["wh"][0],
                Vision.GROUND_Y_RANGE_MM - coords[1] * Vision.GROUND_Y_RANGE_MM / self.warp_transform["wh"][1]
                ]
        else:
            return [
                coords[0] * self.warp_transform["wh"][0] / Vision.GROUND_X_RANGE_MM,
                self.warp_transform["wh"][1] - coords[1] * self.warp_transform["wh"][1] / Vision.GROUND_Y_RANGE_MM
                ]

    def findThymio(self, img, marker_id = 4, remove_border_aruco=True):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.equalizeHist(gray)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and ids[0] == 4:
            thymio_borders_px = corners[0][0]
        else:
            print("Cannot identify Thymio in image!",ids)
            return False, None, None
        if remove_border_aruco:
            cv.fillConvexPoly(img, np.array(thymio_borders_px).astype(int), (255, 255, 255))
        thymio_borders_m = []
        for thymio_border_px in thymio_borders_px:
            thymio_borders_m.append(self.transformPixelXY(thymio_border_px, True))
        thymio_pose = [sum([corner[0] for corner in thymio_borders_m]) / len(thymio_borders_m), sum([corner[1] for corner in thymio_borders_m]) / len(thymio_borders_m)]
        #thymio_pose[1] = Vision.GROUND_Y_RANGE_MM - thymio_pose[1]
        driving_direction = thymio_borders_px[1] - thymio_borders_px[0]
        thymio_pose.append(np.rad2deg(np.pi - np.arctan2(driving_direction[1], driving_direction[0])))
        return True, img, thymio_pose

    def getContourPolygons(self,src:np.ndarray, in_m = True) -> list[Polygon]:
        contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        img_local = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        img_local = cv.drawContours(img_local, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        
        if in_m:
            new_contours = list()
            for i, contour in enumerate(contours[1:]):
                new_entry = []
                for j, point in enumerate(contour):
                    new_entry.append([self.transformPixelXY(point[0], True)])
                    
                new_contours.append(np.array(new_entry).astype(np.int32))
            
        else:
            new_contours = contours[1:]

        polygons = []
        print(contours[:2])
        print(new_contours[:2])
        
        for contour in new_contours:
            epsilon = 0.03 * cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            polygons.append(Polygon(list(map(lambda x:(round(x[0][0],3), round(x[0][1],3)), approx))).buffer(20, join_style=JOIN_STYLE.mitre, cap_style=CAP_STYLE.round))
        return polygons

    def visibilityGraph(self,polygons:list[Polygon],reset_obstacle_map:bool=True,thymio_pose=None,goal_pose=None) -> list[LineString]:
        if reset_obstacle_map: self.obstacle = MultiPolygon()
        potential_wp = []
        potential_segments = []
        for obstacle_polygon in polygons: 
            self.obstacle = self.obstacle.union(obstacle_polygon)
            potential_wp = potential_wp + list(np.asarray(np.dstack(tuple(obstacle_polygon.exterior.xy)))[0])
        if goal_pose is not None:
            potential_wp = [np.asarray(thymio_pose[:2])] + potential_wp
        if thymio_pose is not None:
            potential_wp = [np.asarray(thymio_pose[:2])] + potential_wp
        
        adjacency_matrix = np.zeros((len(potential_wp), len(potential_wp)))
        for indices, line_candidate in zip(combinations(range(len(potential_wp)),2), combinations(potential_wp,2)): 
            test_line = LineString([line_candidate[0], line_candidate[1]])
            inter = test_line.intersection(self.obstacle)
            if (inter == MultiPoint([line_candidate[0], line_candidate[1]]) \
                or inter==MultiPoint([line_candidate[1], line_candidate[0]]))\
                    or ((inter == test_line or inter==LineString([line_candidate[1], line_candidate[0]])) and not test_line.within(self.obstacle))\
                        or inter==Point(line_candidate[1]) or inter==Point(line_candidate[0]):
                potential_segments.append(test_line)
                adjacency_matrix[indices[0], indices[1]] = 1
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        
        return potential_segments, adjacency_matrix, potential_wp

    

    def shapelyXYtoCVLine(xy_tuple):
        to_draw = np.dstack(xy_tuple)
        to_draw = np.asarray(to_draw.astype(int))
        to_draw = np.asarray(to_draw)

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
                polygons = self.getContourPolygons(img)
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
                cv.imshow("out",cv.putText(np.vstack([img_raw, img]), "Lower threshold: {}".format(lower_threshold), (10,50), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,0)))
                cv.waitKey(10)
        return True
if __name__ == "__main__":
    v = Vision(0)
    for i in range(10):
        v.acquireImg()
        time.sleep(0.3)
    #v.autoTuneThreshold(verbose=True)
    print(v.threshold)
    print("AT")
    while True:
        img_orig = v.acquireImg()
        ret, img = v.extractWarp(img_orig)
        if not ret:
            continue
        img = v.applyWarp(img)
        
        ret, img, pos = v.findThymio(img)
        if not ret:
            pos=[0,0,0]
        
        #cv.imshow("img", img)
        if cv.waitKey(10) == ord("q"):
            break
        #print(pos)
        
        img_2 = v.applyPreprocessing(img)
        polygons = v.getContourPolygons(img_2)
        potential_segments,_,_ = v.visibilityGraph(polygons, thymio_pose=pos)
        for polygon in polygons:
            plt.plot(*polygon.exterior.xy,'r',2)
        for potential_segment in potential_segments:
            plt.plot(*potential_segment.xy,'b',1)
        cv.putText(img, "Thymio - x: {}mm, y: {}mm, yaw: {}deg".format(*[int(p) for p in pos]), (5,50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.imshow("IMAGE", img)
        
        plt.show()
        if cv.waitKey(10) == ord("q"):
            break

        continue
        img = v.applyPreprocessing(img_orig)
        polygons = v.getContourPolygons(img)
        potential_segments = v.visibilityGraph(polygons)
        img = vstack([255*np.ones([*img.shape]), img,cv.cvtColor(img_orig, cv.COLOR_RGB2GRAY)])
        plt.imshow(img,cmap="gray")
        for polygon in polygons:
            plt.plot(*polygon.exterior.xy,'r',2)
        for potential_segment in potential_segments:
            plt.plot(*potential_segment.xy,'b',1)
        
        plt.show()
        cv.waitKey(50)



