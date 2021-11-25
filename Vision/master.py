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
import cv2.aruco

class Vision:
    AUTOTUNE_THRESHOLD_AREA = 10000
    AUTOTUNE_THRESHOLD_STEP_SIZE = 40

    def __init__(self, video_source=0):
        self.vid = cv.VideoCapture(0)
        self.threshold = 50
        self.obstacle = MultiPolygon


    def applyPreprocessing(self, src:np.ndarray) -> np.ndarray:
        img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv.medianBlur(img, 5)
        img = cv.bilateralFilter(img, 15, 60, 150)
        ret, img = cv.threshold(img,self.threshold,255,cv.THRESH_BINARY)
        return cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((15,15),np.uint8))

    def acquireImg(self):
        ret, img = self.vid.read()
        if not ret:
            print("Error reading source!")

            return None
        else:
            return img

    def getContourPolygons(self,src:np.ndarray) -> list[Polygon]:
        contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        polygons = []
        for contour in contours[1:]:
            epsilon = 0.03 * cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            polygons.append(Polygon(list(map(lambda x:(round(x[0][0],3), round(x[0][1],3)), approx))).buffer(20, join_style=JOIN_STYLE.mitre, cap_style=CAP_STYLE.round))
        return polygons

    def visibilityGraph(self,polygons:list[Polygon],reset_obstacle_map:bool=True) -> list[LineString]:
        if reset_obstacle_map: self.obstacle = MultiPolygon()
        potential_wp = []
        potential_segments = []
        for obstacle_polygon in polygons: 
            self.obstacle = self.obstacle.union(obstacle_polygon)
            potential_wp = potential_wp + list(np.asarray(np.dstack(tuple(obstacle_polygon.exterior.xy)))[0])
        for line_candidate in combinations(potential_wp,2):
            print(line_candidate)
            test_line = LineString([line_candidate[0], line_candidate[1]])
            inter = test_line.intersection(self.obstacle)
            if (inter == MultiPoint([line_candidate[0], line_candidate[1]]) \
                or inter==MultiPoint([line_candidate[1], line_candidate[0]]))\
                    or ((inter == test_line or inter==LineString([line_candidate[1], line_candidate[0]])) and not test_line.within(self.obstacle)):
                potential_segments.append(test_line)
        return potential_segments

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
    #v.autoTuneThreshold(verbose=True)
    print(v.threshold)
    print("AT")
    while True:
        img_orig = v.acquireImg()
        
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



