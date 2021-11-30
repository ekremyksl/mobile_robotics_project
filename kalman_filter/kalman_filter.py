#from kalman_calc import KalmanCalc

import math
import time
import numpy as np
from kalman_calc import KalmanCalc
from repeated_timer import RepeatedTimer
from plot_map import PlotMap

class KalmanFilter():
    def __init__(self, node, period, state_vector, position_uncertainty, angle_uncertainty):
        # node for connection with thymio
        self.node = node

        self.period = period # fiter updating period
        self.thymio_width = 10 # distance from wheel to wheel of thymio

        # state estimation vector: [position x, position y, velocity x, velocity y, angle, angular velocity]
        self.x = np.array(state_vector)

        # estimation uncertainty matrix (covariance matrix) of position and angle
        self.Pxy = np.array(position_uncertainty)
        self.Pangle = np.array(angle_uncertainty)

        # memorize past state vectors and covariance matrices
        self.x_list = []
        self.Pxy_list = []

        # positional measurement flag (if set then update kalman filter including positional measurement)
        self.m_pos_flag = False

        # positional measurement (measured by picture): [position x, position y, angel]
        self.m_pos = np.array([0.,0.,0.])

        self.kalman_pose = KalmanCalc()
        self.kalman_position = KalmanCalc()

        # self.speed_list = [[5,5],[5,5],[5,5],[5,5],[5,5],[5,5]]
        # self.speed_list = [[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #               [18,22],[18,22],[18,22],[18,22],[18,22],[20,20],[20,20],[20,20],[20,20]]
        # self.speed_list = [[20,20],[20,20],[20,20],[10,30],[10,30],[10,30],[10,30],[10,30],[10,30],[20,20],[20,20],[20,20]]
        self.speed_list = [[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
                            [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
                            [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
                            [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
                            [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
                            [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20]]
        
    def get_state_vector(self):
        return self.x

    def get_state_vector_list(self):
        return self.x_list

    def get_covariance_list(self):
        return self.Pxy_list       

    def update_filter(self):        
        speed = self._measure_speed()

        self._estimate_pose(speed)
        self._estimate_position(speed)
        
        self.x_list.append(list(self.x))
        self.Pxy_list.append(self.Pxy)

        # print("{} a:\t{}".format(i,np.round(self.pose, 2)))
        print("x:\t{}".format(np.round(self.x, 2)))
        # print("{} s:\t{}".format(i,speed))
        # print("{} z:\t{}".format(i,z))


    def _measure_speed(self):
        if self.node != 0:
            return [self.node["motor.left.speed"], self.node["motor.right.speed"]]
        else:
            return self.speed_list.pop(0)

    def _estimate_pose(self, speed):
        m_vel_angle = (speed[1]-speed[0])/self.thymio_width
        if self.m_pos_flag:
            z = np.array([self.m_pos[2], m_vel_angle])
        else:
            z = np.array([m_vel_angle])

        F, Q, H, R = self._pose_matrices()

        self.x[4:6], self.Pangle = self.kalman_pose.update(self.x[4:6], z, F, Q, H, R, self.Pangle)


    def _estimate_position(self, speed):
        m_vel_x = math.cos(self.x[4])*(speed[0]+speed[1])/2
        m_vel_y = math.sin(self.x[4])*(speed[0]+speed[1])/2 
        if self.m_pos_flag:
            z = np.array([self.m_pos[0], self.m_pos[1], m_vel_x, m_vel_y])
        else:
            z = np.array([m_vel_x, m_vel_y])       

        F, Q, H, R = self._position_matrices()

        self.x[0:4], self.Pxy = self.kalman_position.update(self.x[0:4], z, F, Q, H, R, self.Pxy)

    def _position_matrices(self):
        # state transition matrix
        F = np.array([[1,0,self.period,0], 
                      [0,1,0,self.period], 
                      [0,0,1,0], 
                      [0,0,0,1]])

        # process noise matrix
        # Q = np.array([[0.1,0,0,0,], 
        #               [0,0.1,0,0,], 
        #               [0,0,0.2,0,],
        #               [0,0,0,0.2]])
        r_speed = 40
        std_x = math.sqrt(self.period*abs(math.cos(self.x[4]))*r_speed)
        std_y = math.sqrt(self.period*abs(math.sin(self.x[4]))*r_speed)
        std_vx = math.sqrt(abs(math.cos(self.x[4]))*r_speed)
        std_vy = math.sqrt(abs(math.sin(self.x[4]))*r_speed)
        # Q = np.array([[std_x*std_x,std_x*std_y,0,0], 
        #               [std_x*std_y,std_y*std_y,0,0], 
        #               [0,0,std_vx*std_vx,std_vx*std_vy,],
        #               [0,0,std_vx*std_vy,std_vy*std_vy]])
        Q = np.array([[std_x*std_x,0,0,0], 
                      [0,std_y*std_y,0,0], 
                      [0,0,std_vx*std_vx,0],
                      [0,0,0,std_vy*std_vy]])

        # observation matrix
        H_pos = np.array([[1,0,0,0,],
                          [0,1,0,0,]])

        H_speed = np.array([[0,0,1,0],
                            [0,0,0,1]])

        # measurement noise matrix
        measure_noise = 10 # variance of speed measurement
        std_mx = math.sqrt(abs(math.cos(self.x[4]))*measure_noise)
        std_my = math.sqrt(abs(math.sin(self.x[4]))*measure_noise)
        R_speed = np.array([[std_mx*std_mx,std_mx*std_my],
                            [std_mx*std_my,std_my*std_my]])
        # R_speed = np.array([[0.05,0.05],
        #                     [0.05,0.05]])
        R_pos = np.array([[measure_noise,0],
                          [0,measure_noise]])

        H, R = self._concernate_matrices(H_pos, H_speed, R_pos, R_speed)
        
        return F, Q, H, R

    def _pose_matrices(self):
        # state transition matrix
        F = np.array([[1,self.period], 
                      [0,1]])

        # process noise matrix
        Q = np.array([[0.04,0], 
                      [0,0.04]])

        # observation matrix
        H_pos = np.array([[1,0]])
        H_speed = np.array([[0,1]])

        # measurement noise matrix
        R_speed = np.array([[0.01]])
        R_pos = np.array([[0.01]])

        H, R = self._concernate_matrices(H_pos, H_speed, R_pos, R_speed)
        return F, Q, H, R

    def _concernate_matrices(self, H_pos, H_speed, R_pos, R_speed):
        if self.m_pos_flag: # merge position and speed matrices
            dim = np.size(R_pos, axis=0)
            Z = np.zeros((dim,dim))
            H = np.concatenate((H_pos, H_speed), axis=0)
            R = np.concatenate((np.concatenate((R_pos, Z), axis=0), \
                                np.concatenate((Z, R_speed), axis=0)), axis=1)
            return H, R
        else:
            return H_speed, R_speed


if __name__ == '__main__':
    angle = angle = np.deg2rad(0)
    v = 5
    vx = math.cos(angle)*v
    vy = math.sin(angle)*v

    state_vector = [0,0,vx,vy, angle,0]
    position_uncertainty = [[0.1,0,0,0],
                            [0,0.1,0,0],
                            [0,0,0,0],
                            [0,0,0,0]]
    angle_uncertainty = [[0,0],
                         [0,0]]

    # no thymio is used
    node = 0
    period = 0.1


    filter = KalmanFilter(node, period, state_vector, position_uncertainty, angle_uncertainty)
    t1 = RepeatedTimer(period, filter.update_filter)

    t1.start()
    time.sleep(10) 
    t1.stop()

    print(filter.get_covariance_list())

    PlotMap(period, filter.get_state_vector_list(), filter.get_covariance_list())