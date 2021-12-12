from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from Vision import Vision

import math
import numpy as np
import time
from tdmclient import ClientAsync, aw
from kalman_filter import KalmanFilter
from repeated_timer import RepeatedTimer
from plot_map import PlotMap
import cv2 as cv
from AstolfiController import Astolfi
from SimpleController import SimpleController
from Controller import Controller

def measure(self):
        #await node.wait_for_variables() # wait for Thymio variables values
        self._data["time"].append(time.time()) # save timestamp
        self._data["ground_prox"].append(list(self._node["prox.ground.reflected"]))

def motors(left, right):
    left = int((500/16)*left)
    right = int((500/16)*right)
    return { "motor.left.target": [left], "motor.right.target": [right] }

def thymioPosePipeline(v:Vision):
    while True:
        for i in range(10):
            img_orig = v.acquireImg(True)
        img_orig = cv.resize(img_orig, (1920, 1080))
        ret, img, warp = v.extractWarp(img_orig)
        if not ret:
            print("Could not find all 4 markers")
            continue
        img = v.applyWarp(img, warp)
        ret, _, pos = v.findThymio(img, 4, remove_thymio="marker")
        if ret:
            return [pos[0] / 10, pos[1] / 10, pos[2]], img

if __name__ == '__main__':
    # set up connection to thymio, if node=0 no thymio is connected
    client = ClientAsync()
    node = aw(client.wait_for_node())
    aw(node.lock())
    v = Vision(1)
    pose, image = thymioPosePipeline(v)

    # initialization of controller
    cont = Controller()
    #preparing global path
    flag = False
    while flag==False:
        temp = v.prepareWaypoints()
        if temp is not None:
            angle = temp[0]
            trajectory = temp[1]
            flag = True

    # trajectory = [[300,200],[800,200],[300,200],[200,200]]
    # setting global path

    cont.set_global(trajectory,cont.normalize_ang(np.deg2rad(angle)),verbose=True)
    # set up kalman filter
    angle = np.deg2rad(pose[2])
    vl = 2
    vr = 2
    vx = math.cos(angle)*(vl+vr)/2
    vy = math.sin(angle)*(vl+vr)/2
    va = (vr-vl)/9.4
    state_vector = np.array([[pose[0],pose[1],angle,vx,vy,va]], dtype=float)
    uncertainty_matrix = np.array([[[0.1,0,0,0,0,0],
                                    [0,0.1,0,0,0,0],
                                    [0,0,0.1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]]], dtype=float)

    period = 0.1
    filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0], print_variables=True)
    
    t1 = RepeatedTimer(period, filter.update_filter)

    t1.start()
    aw(node.set_variables(motors(vl,vr)))
    i=0
    state = 'TURN'
    aw(node.set_variables(motors(0,0)))
    pos=[]
    while cont.on_goal==False:
        time.sleep(0.2) 
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

        # calc. second std: probability is more than 95% that robot is inside second std
        eigenvalues, _ = np.linalg.eig(uncertainty_matrix[i+1])
        stds2 = 2*np.sqrt(np.absolute(eigenvalues))
        #taking current state of the robot
        curr = filter.get_state_vector().copy()
        curr[2]=cont.normalize_ang(curr[2])
        cont.set_curr(curr[0:3].copy())
        temp.append([curr])
        # #to test with dummy trajectory
        # if i==0: 
        #     temp=[[curr[0],curr[1]]]
        #     j=0
        #     while len(temp)<len(trajectory)+1:
        #         temp.append(trajectory[j])
        #         j+=1
        #     cont.set_global(temp,np.deg2rad(angle),verbose=True)

       
        print('***********',cont.next, cont.curr)
        cont.motion_control(node, fkine=False ,astolfi=True, verbose=True)

    
        
        if np.amax(stds2) > 3:
            aw(node.set_variables(motors(0,0)))
            t1.stop()
            print("taking picture at {}".format(i))
            pose, _ = thymioPosePipeline(v)


            print("second stds: {}".format(stds2))
            #x_pos = float(input("enter x position: "))
            #y_pos = float(input("enter y_position: "))
            #angle = np.deg2rad(float(input("enter angle [in °]: ")))
            pose[2] = np.deg2rad(pose[2])
            print(pose)
            filter.set_position_measurement(pose)
            t1.start()
            aw(node.set_variables(motors(vl,vr)))

        pos.append(cont.curr)
        i+=1
        
    temp=np.array(temp)
    aw(node.set_variables(motors(0,0)))
    print('****GOAL IS REACHED****')
    t1.stop()
    path=cont.path
    pos=np.array(pos)
    plt.subplots()
    plt.plot(path[:,0], path[:,1],'r*')
    plt.plot(path[:,0], path[:,1],'r--')
    plt.plot(pos[:,0],pos[:,1],'b')
    
    PlotMap(period, state_vector, uncertainty_matrix, image, [int(Vision.GROUND_X_RANGE_MM / 10), int(Vision.GROUND_Y_RANGE_MM / 10)])
    plt.show()
    