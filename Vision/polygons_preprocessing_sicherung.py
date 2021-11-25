import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from itertools import combinations
import matplotlib.pyplot  as plt
from shapely.geometry.multipoint import MultiPoint
#img = cv.imread("test_polygons.jpg", cv.IMREAD_GRAYSCALE)
#img = cv.imread("polygon_test.png", cv.IMREAD_GRAYSCALE)
img = cv.imread("sand_polygon_test.png", cv.IMREAD_GRAYSCALE)
img_orig = cv.imread("sand_polygon_test.png", cv.IMREAD_COLOR)
h,w= img.shape
img = cv.resize(img, dsize=[int(w/4),int(h/4)])
img_orig = cv.resize(img_orig, dsize=[int(w/4),int(h/4)])

img = cv.equalizeHist(img)
img = cv.bilateralFilter(img, 15, 60, 150)
#img = cv.GaussianBlur(img, (5,5), 0)
ret, img = cv.threshold(img,25,255,cv.THRESH_BINARY)
img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((9,9),np.uint8))
#cv.imshow('img',img)
#cv.waitKey(0)
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
img_col = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
obstacles_list = []
obstacle = MultiPolygon()
points = []
plt.imshow(img_orig)
for contour in contours[1:]:
    epsilon = 0.05*cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour,epsilon,True)
    poly = Polygon(list(map(lambda x:(round(x[0][0],3), round(x[0][1],3)), approx))).buffer(20, join_style=JOIN_STYLE.mitre, cap_style=CAP_STYLE.round)
    obstacle = obstacle.union(poly)
    to_draw = np.dstack(tuple(poly.exterior.xy))
    plt.plot(*poly.exterior.xy,'r',linewidth=3.0)
    #to_draw = np.asarray(to_draw.astype(int))
    to_draw = np.asarray(to_draw)
   # to_draw = np.clip(to_draw, 0,100000)
    points = points + list(to_draw[0])
    #img_col = cv.polylines(img_col,[approx], True,(255,0,255), 3)
#obstacle = MultiPolygon(obstacles_list)
valid_connections = []
for line_candidate in combinations(points,2):
    l = LineString([line_candidate[0], line_candidate[1]])
    inter = l.intersection(obstacle)
    if inter == MultiPoint([line_candidate[0], line_candidate[1]]) \
        or inter==MultiPoint([line_candidate[1], line_candidate[0]]):
        valid_connections.append(l)
        plt.plot(*l.xy, 'g')
        continue
    if (inter == l or inter==LineString([line_candidate[1], line_candidate[0]])) and not l.within(obstacle):
        valid_connections.append(l)
        plt.plot(*l.xy, 'g')
plt.show()        


