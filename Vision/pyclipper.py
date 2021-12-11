import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pyclipper
img = cv.imread("test_polygons.jpg", cv.IMREAD_GRAYSCALE)
#img = cv.imread("polygon_test.png", cv.IMREAD_GRAYSCALE)
h,w= img.shape
img = cv.resize(img, dsize=[int(w/2),int(h/4)])

img = cv.equalizeHist(img)
img = cv.bilateralFilter(img, 15, 60, 150)
#img = cv.GaussianBlur(img, (5,5), 0)
ret, img = cv.threshold(img,20,255,cv.THRESH_BINARY)
img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((11,11),np.uint8))
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
img_col = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
for contour in contours:
    epsilon = 0.03*cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour,epsilon,True)
    approx = np.append(approx, [approx[0]], axis=0)
    # https://github.com/fonttools/pyclipper/tree/main/src
    subj = tuple(map(lambda x:tuple(x[0]), approx))
    pco = pyclipper.PyclipperOffset(10, 0)
    pco.AddPath(subj, 2, pyclipper.ET_CLOSEDPOLYGON)
    result=pco.Execute(25.0)
    res = []
    for i in range(len(result[0])):
        res.append([[int(result[0][i][0]), int(result[0][i][1])]])
    res = np.asarray(res)
    img_col = cv.polylines(img_col,[res], False,(255,0,255), 5)
    
cv.imshow("IMAGE", img_col)
cv.waitKey(0)

