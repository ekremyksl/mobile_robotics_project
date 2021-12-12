from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from functions.Vision import Vision
import math
import numpy as np
import time
from tdmclient import ClientAsync, aw
from functions.kalman_filter import KalmanFilter
from functions.repeated_timer import RepeatedTimer
from functions.plot_map import PlotMap
import cv2 as cv
from functions.Controller import Controller

def RoboMove():
        # set up connection to thymio, if node=0 no thymio is connected
    client = ClientAsync()
    node = aw(client.wait_for_node())
    aw(node.lock())
    v = Vision(1)
    pose = False
    while pose is False:
        pose, image = v.getThymioPose()

    # initialization of controller
    cont = Controller(node)
    #preparing global path
    flag = False
    while flag==False:
        temp = v.getTrajectory()
        if temp is not None:
            angle = pose[2]
            trajectory = temp
            flag = True


    # setting global path
    cont.set_global(trajectory, cont.normalize_ang(np.deg2rad(angle)),conversion='mm2cm',verbose=True)
    # set up kalman filter
    vl = 0
    vr = 0
    vx = 0
    vy = 0
    va = 0
    state_vector = np.array([[pose[0],pose[1],angle,vx,vy,va]], dtype=float)
    uncertainty_matrix = np.array([[[0.1,0,0,0,0,0],
                                    [0,0.1,0,0,0,0],
                                    [0,0,0.1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]]], dtype=float)

    period = 0.1
    filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0], print_variables=False)
    
    t1 = RepeatedTimer(period, filter.update_filter)

    t1.start()

    while cont.on_goal==False:
        time.sleep(period*2) 
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

        # calc. second std: probability is more than 95% that robot is inside second std
        eigenvalues, _ = np.linalg.eig(uncertainty_matrix[len(uncertainty_matrix)-1])
        stds2 = 2*np.sqrt(np.absolute(eigenvalues))
        #taking current state of the robot
        curr = filter.get_state_vector().copy()
        curr[2]=cont.normalize_ang(curr[2])
        cont.set_curr(curr[0:3].copy())

        cont.motion_control(fkine=False ,astolfi=True, verbose=False)    
        
        #correct the estimation of Kalman Filter if uncertainty gets bigger than certain threshold
        #in that case, it takes another picture to obtain the pose of Thymio 
        if np.amax(stds2) > 3:
            print("taking picture at {}".format(len(uncertainty_matrix)-1))
            pose, _ = v.getThymioPose()
            if pose is not False:
                pose[2] = np.deg2rad(pose[2])                
                filter.set_position_measurement(pose)

        
    temp=np.array(temp)
    aw(node.set_variables(cont.motors(0,0)))
    print('****GOAL IS REACHED****')
    t1.stop()

    PlotMap(period, state_vector, uncertainty_matrix, image, cont.path)
    plt.show()

# if __name__ == '__main__':
#     # set up connection to thymio, if node=0 no thymio is connected
#     client = ClientAsync()
#     node = aw(client.wait_for_node())
#     aw(node.lock())
#     v = Vision(1)
#     pose = False
#     while pose is False:
#         pose, image = v.getThymioPose()

#     # initialization of controller
#     cont = Controller(node)
#     #preparing global path
#     flag = False
#     while flag==False:
#         temp = v.getTrajectory()
#         if temp is not None:
#             angle = pose[2]
#             trajectory = temp
#             flag = True


#     # setting global path
#     cont.set_global(trajectory, cont.normalize_ang(np.deg2rad(angle)),conversion='mm2cm',verbose=True)
#     # set up kalman filter
#     vl = 0
#     vr = 0
#     vx = 0
#     vy = 0
#     va = 0
#     state_vector = np.array([[pose[0],pose[1],angle,vx,vy,va]], dtype=float)
#     uncertainty_matrix = np.array([[[0.1,0,0,0,0,0],
#                                     [0,0.1,0,0,0,0],
#                                     [0,0,0.1,0,0,0],
#                                     [0,0,0,0,0,0],
#                                     [0,0,0,0,0,0],
#                                     [0,0,0,0,0,0]]], dtype=float)

#     period = 0.1
#     filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0], print_variables=False)
    
#     t1 = RepeatedTimer(period, filter.update_filter)

#     t1.start()

#     while cont.on_goal==False:
#         time.sleep(period*2) 
#         state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
#         uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

#         # calc. second std: probability is more than 95% that robot is inside second std
#         eigenvalues, _ = np.linalg.eig(uncertainty_matrix[len(uncertainty_matrix)-1])
#         stds2 = 2*np.sqrt(np.absolute(eigenvalues))
#         #taking current state of the robot
#         curr = filter.get_state_vector().copy()
#         curr[2]=cont.normalize_ang(curr[2])
#         cont.set_curr(curr[0:3].copy())

#         cont.motion_control(fkine=False ,astolfi=True, verbose=False)    
        
#         #correct the estimation of Kalman Filter if uncertainty gets bigger than certain threshold
#         #in that case, it takes another picture to obtain the pose of Thymio 
#         if np.amax(stds2) > 3:
#             print("taking picture at {}".format(len(uncertainty_matrix)-1))
#             pose, _ = v.getThymioPose()
#             if pose is not False:
#                 pose[2] = np.deg2rad(pose[2])                
#                 filter.set_position_measurement(pose)

        
#     temp=np.array(temp)
#     aw(node.set_variables(cont.motors(0,0)))
#     print('****GOAL IS REACHED****')
#     t1.stop()

#     PlotMap(period, state_vector, uncertainty_matrix, image, cont.path)
#     plt.show()
    