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

    # initialization of
    # astolfi = Astolfi()
    # simp = SimpleController()
    cont = Controller()
    #preparing global path
    flag = False
    while flag==False:
        temp = v.prepareWaypoints()
        if temp is not None:
            angle = temp[0]
            trajectory =temp[1]
            flag = True

    
    # setting global path
    # simp.set_global(trajectory,np.deg2rad(angle))
    # astolfi.set_path(trajectory,np.deg2rad(angle))
    cont.set_global(trajectory,np.deg2rad(angle),verbose=True)
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
    while cont.on_goal==False:
        time.sleep(0.2) 
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

        # calc. second std: probability is more than 95% that robot is inside second std
        eigenvalues, _ = np.linalg.eig(uncertainty_matrix[i+1])
        stds2 = 2*np.sqrt(np.absolute(eigenvalues))
        curr = filter.get_state_vector().copy()
        curr[2]=curr[2]%(2*math.pi)
        #setting current position
        # simp.set_curr(curr[0:3].copy())
        # astolfi.set_curr(curr[0:3].copy())
        cont.set_curr(curr[0:3].copy())
        #correcting heading
        # error = ((simp.next[2]-simp.curr[2] + np.pi) % (2 * np.pi) - np.pi)

        cont.motion_control(node, astolfi=True, verbose=True)

        
        # dist = astolfi.norm(simp.curr-simp.next)
        # if abs(error) > simp.heading_th:
        #     vl,vr = simp.correct_heading(error)
        #     aw(node.set_variables(motors(vl,vr)))
        # else:
        #     rho,alpha,beta = astolfi.polar_rep()
        #     print(dist)
        #     vl,vr = simp.follow_line(dist,error)
        #     aw(node.set_variables(motors(vl,vr)))
        #     simp.check_node()
        #     # vl,vr = simp.correct_heading(error)
        #     aw(node.set_variables(motors(vl,vr)))
        #     astolfi.set_goal(simp.get_goal())
        #     astolfi.check_nodes(verbose=True)
        # #print(rho)
        # if astolfi.rho < 10000000000:
            
        # elif astolfi.rho >= 2000:      
        #     vl,vr = astolfi.compute_phi_dot()
        #     aw(node.set_variables(motors(vl,vr)))
        #     simp.set_goal(astolfi.get_goal())


        
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

        i+=1
        
        
    aw(node.set_variables(motors(0,0)))
    print('****GOAL IS REACHED****')
    t1.stop()


    PlotMap(period, state_vector, uncertainty_matrix, image, [int(Vision.GROUND_X_RANGE_MM / 10), int(Vision.GROUND_Y_RANGE_MM / 10)])

    