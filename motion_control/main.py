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
from Controller import Controller

def measure(self):
        #await node.wait_for_variables() # wait for Thymio variables values
        self._data["time"].append(time.time()) # save timestamp
        self._data["ground_prox"].append(list(self._node["prox.ground.reflected"]))


if __name__ == '__main__':
    # set up connection to thymio, if node=0 no thymio is connected
    client = ClientAsync()
    node = aw(client.wait_for_node())
    aw(node.lock())
    v = Vision(1)
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
    aw(node.set_variables(cont.motors(vl,vr)))
    i=0
    state = 'TURN'
    aw(node.set_variables(cont.motors(0,0)))
    pos=[]
    phi_dot=[]
    pba=[]
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

        cont.motion_control(fkine=False ,astolfi=True, verbose=True)
        phi_dot.append([cont.phiL,cont.phiR])
        pba.append([cont.rho,cont.alpha,cont.beta])
    
        
        if np.amax(stds2) > 3:
            aw(node.set_variables(cont.motors(0.5,0.5)))
            t1.stop()
            print("taking picture at {}".format(i))
            pose, _ = v.getThymioPose()


            print("second stds: {}".format(stds2))
            #x_pos = float(input("enter x position: "))
            #y_pos = float(input("enter y_position: "))
            #angle = np.deg2rad(float(input("enter angle [in °]: ")))
            pose[2] = np.deg2rad(pose[2])
            print(pose)
            filter.set_position_measurement(pose)
            t1.start()
            aw(node.set_variables(cont.motors(vl,vr)))

        pos.append(cont.curr)
        i+=1
        
    temp=np.array(temp)
    aw(node.set_variables(cont.motors(0,0)))
    print('****GOAL IS REACHED****')
    t1.stop()
    path=cont.path
    pos=np.array(pos)
    pba=np.array(pba)
    phi_dot=np.array(phi_dot)
    plt.subplots()
    plt.plot(path[:,0], path[:,1],'r*')
    plt.plot(path[:,0], path[:,1],'r--')
    plt.plot(pos[:,0],pos[:,1],'b')
    plt.ylabel('Y')
    plt.xlabel('X')    
    plt.subplot()
    plt.plot(phi_dot[:,0],'b')
    plt.plot(phi_dot[:,1],'r')
    plt.xlabel('time')
    plt.ylabel('$\dot\phi$')
    plt.legend()
    plt.plot(pba[:,0],'b')
    plt.plot(pba[:,1],'g')
    plt.plot(pba[:,2],'r')
    # PlotMap(period, state_vector, uncertainty_matrix, image)
    plt.show()
    