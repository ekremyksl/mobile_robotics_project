#from kalman_calc import KalmanCalc

import math
import time
import numpy as np
from tdmclient import ClientAsync, aw
from kalman_calc import KalmanCalc
from repeated_timer import RepeatedTimer
from plot_map import PlotMap

class KalmanFilter():
    def __init__(self, node, period, state_vector, position_uncertainty):
        # node for connection with thymio
        self.node = node

        self.period = period # fiter updating period
        self.thymio_width = 9.4 # distance from wheel to wheel of thymio

        # state estimation vector: [position x, position y, angle, velocity x, velocity y, angular velocity]
        self.x = np.array(state_vector, dtype=float)

        # estimation uncertainty matrix (covariance matrix) of position and angle
        self.P = np.array(position_uncertainty, dtype=float)

        # positional measurement flag (if set then update kalman filter including positional measurement)
        self.m_pos_flag = False

        # positional measurement (measured by picture): [position x, position y, angel]
        self.m_pos = np.zeros((3), dtype=float)

        self.speed_correction = 1.23

        self.m_speed_noise = 2 # variance of speed measurement
        self.p_speed_noise = 0.4 # variance of speed during processing

        self.kalman_pose = KalmanCalc()
        self.kalman_position = KalmanCalc()

        self.speed_list = [[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],
                            [5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5]]
        # self.speed_list = [[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #               [18,22],[18,22],[18,22],[18,22],[18,22],[20,20],[20,20],[20,20],[20,20]]
        # self.speed_list = [[20,20],[20,20],[20,20],[10,30],[10,30],[10,30],[10,30],[10,30],[10,30],[20,20],[20,20],[20,20]]
        # self.speed_list = [[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
        #                     [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
        #                     [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
        #                     [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],\
        #                     [7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[7.,3.],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],\
        #                     [20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20],[20,20]]

    def set_position_measurement(self, position):
        self.m_pos = np.array(position, dtype=float)
        self.m_pos_flag = True

    def get_state_vector(self):
        return self.x

    def get_covariance_matrix(self):
        return self.P      

    def update_filter(self):        
        speed = self._measure_speed()
        z = self._calc_velocity(speed)

        F, Q, H, R = self._calc_matrices()
        self.x, self.P = self.kalman_position.update(self.x, z, F, Q, H, R, self.P)

        if self.m_pos_flag:
            self.m_pos_flag = False

        # print("{} a:\t{}".format(i,np.round(self.pose, 2)))
        print("x:\t{} {} {} {}".format(np.round(self.x[0:2], 2), np.round(np.rad2deg(self.x[2]),2), \
                                        np.round(self.x[4], 2), np.round(np.rad2deg(self.x[5]),2)))
        print("s:\t{}".format(np.round(speed, 1)))
        # print("{} z:\t{}".format(i,z))


    def _measure_speed(self):
        if self.node != 0:
            aw(self.node.wait_for_variables())
            left = (16/500)*self.node["motor.left.speed"]*self.speed_correction
            right = (16/500)*self.node["motor.right.speed"]*self.speed_correction
            return [left, right]
        else:
            return self.speed_list.pop(0)

    def _calc_velocity(self, speed):
        m_vel = np.zeros((3), dtype=float)
        m_vel[0] = math.cos(self.x[2])*(speed[0]+speed[1])/2
        m_vel[1] = math.sin(self.x[2])*(speed[0]+speed[1])/2
        m_vel[2] = (speed[1]-speed[0])/self.thymio_width

        if self.m_pos_flag:
            return np.concatenate((self.m_pos, m_vel), axis=0)
        else:
            return m_vel

    def _calc_matrices(self):
        # state transition matrix
        F = np.array([[1,0,0,self.period,0,0], 
                      [0,1,0,0,self.period,0], 
                      [0,0,1,0,0,self.period], 
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]], dtype=float)

        # process uncertainty matrix
        p_std = np.zeros((6), dtype=float)
        p_std[0] = math.sqrt(self.period*abs(math.cos(self.x[2]))*self.p_speed_noise)
        p_std[1] = math.sqrt(self.period*abs(math.sin(self.x[2]))*self.p_speed_noise)
        p_std[2] = math.sqrt(self.period*2*self.p_speed_noise/self.thymio_width)
        p_std[3] = math.sqrt(abs(math.cos(self.x[2]))*self.p_speed_noise)
        p_std[4] = math.sqrt(abs(math.sin(self.x[2]))*self.p_speed_noise)
        p_std[5] = math.sqrt(2*self.p_speed_noise/self.thymio_width)
        Q = np.array([[p_std[0]*p_std[0],0,0,0,0,0], 
                      [0,p_std[1]*p_std[1],0,0,0,0], 
                      [0,0,p_std[2]*p_std[2],0,0,0],
                      [0,0,0,p_std[3]*p_std[3],0,0],
                      [0,0,0,0,p_std[4]*p_std[4],0],
                      [0,0,0,0,0,p_std[5]*p_std[5]]], dtype=float)
        # Q = np.array([[p_std[0]*p_std[0],p_std[1]*p_std[0],p_std[2]*p_std[0],p_std[3]*p_std[0],p_std[4]*p_std[0],p_std[5]*p_std[0]], 
        #               [p_std[0]*p_std[1],p_std[1]*p_std[1],p_std[2]*p_std[1],p_std[3]*p_std[1],p_std[4]*p_std[1],p_std[5]*p_std[1]], 
        #               [p_std[0]*p_std[2],p_std[1]*p_std[2],p_std[2]*p_std[2],p_std[3]*p_std[2],p_std[4]*p_std[2],p_std[5]*p_std[2]],
        #               [p_std[0]*p_std[3],p_std[1]*p_std[3],p_std[2]*p_std[3],p_std[3]*p_std[3],p_std[4]*p_std[3],p_std[5]*p_std[3]],
        #               [p_std[0]*p_std[4],p_std[1]*p_std[4],p_std[2]*p_std[4],p_std[3]*p_std[4],p_std[4]*p_std[4],p_std[5]*p_std[4]],
        #               [p_std[0]*p_std[5],p_std[1]*p_std[5],p_std[2]*p_std[5],p_std[3]*p_std[5],p_std[4]*p_std[5],p_std[5]*p_std[5]]], dtype=float)

        # observation matrix
        H_pos = np.array([[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0]], dtype=float)

        H_speed = np.array([[0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]], dtype=float)

        # measurement uncertainty matrix
        m_std = np.zeros((3), dtype=float)
        m_std[0] = math.sqrt(abs(math.cos(self.x[2]))*self.m_speed_noise)
        m_std[1] = math.sqrt(abs(math.sin(self.x[2]))*self.m_speed_noise)
        m_std[2] = math.sqrt(2*self.m_speed_noise/self.thymio_width)
        R_speed = np.array([[m_std[0]*m_std[0],m_std[0]*m_std[1],m_std[0]*m_std[2]],
                            [m_std[0]*m_std[1],m_std[1]*m_std[1],m_std[1]*m_std[2]],
                            [m_std[0]*m_std[2],m_std[1]*m_std[2],m_std[2]*m_std[2]]], dtype=float)
        R_pos = np.array([[0.01,0,0],
                          [0,0.01,0],
                          [0,0,0.001]], dtype=float)

        # merge position and speed matrices if position measurement was made
        if self.m_pos_flag: 
            dim = np.size(R_pos, axis=0)
            Z = np.zeros((dim,dim))
            H = np.concatenate((H_pos, H_speed), axis=0)
            R = np.concatenate((np.concatenate((R_pos, Z), axis=0), \
                                np.concatenate((Z, R_speed), axis=0)), axis=1)
            return F, Q, H, R
        else:
            return F, Q, H_speed, R_speed

if __name__ == '__main__':
    angle = angle = np.deg2rad(90)
    v = 5
    vx = math.cos(angle)*v
    vy = math.sin(angle)*v
    state_vector = np.array([[0,0,angle,vx,vy,0]], dtype=float)
    uncertainty_matrix = np.array([[[0.1,0,0,0,0,0],
                                    [0,0.1,0,0,0,0],
                                    [0,0,0.1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]]], dtype=float)

    # no thymio is used
    node = 0
    

    period = 0.1
    filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0])
    t1 = RepeatedTimer(period, filter.update_filter)


    t1.start()
    for i in range(8):
        time.sleep(1) 
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        position_uncertainty = np.append(position_uncertainty, [filter.get_covariance_matrix()], axis=0)

        # t1.stop()
        # print(state_vector[i+1])
        # x_pos = float(input("enter x position: "))
        # y_pos = float(input("enter y_position: "))
        # angle = float(input("enter angle: "))
        # filter.set_position_measurement([x_pos, y_pos, angle])
        # t1.start()

    t1.stop()


    PlotMap(period, state_vector, position_uncertainty)