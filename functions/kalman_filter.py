import math
import time
import numpy as np
from tdmclient import ClientAsync, aw
from repeated_timer import RepeatedTimer
from plot_map import PlotMap

class KalmanFilter():
    def __init__(self, node, period, state_vector, position_uncertainty, print_variables=False):
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

        self.m_pos_noise = 0.01 # variance of positional measurement in x and y direction (estimated variance of camera)
        self.m_angular_noise = 0.003 # variance of angular pose measurement (estimated variance of camera)
        self.m_speed_noise = 2 # variance of speed measurement
        self.p_speed_noise = 0.4 # variance of speed during processing

        self.print_variables = print_variables # if true the state vector and the speed measurement are printed

        # if node=0 this list contains the "speed measurements" with which the kalman filter is tested
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
        # measure speed and calc. velocity       
        speed = self._measure_speed()
        z = self._calc_velocity(speed)

        # calc. matrices used for kalman filter
        F, Q, H, R = self._calc_matrices()

        # make prediction step
        self.x, self.P = self._prediction(self.x, F, Q, self.P)

        # make correction step 
        self.x, self.P = self._correction(self.x, z, H, R, self.P)

        # reset flag because position measurement was used in last correction step
        if self.m_pos_flag:
            self.m_pos_flag = False

        # print variables for debugging if print_variable is set true
        if self.print_variables:
            print("x:\t{} {} {} {}".format(np.round(self.x[0:2], 2), np.round(np.rad2deg(self.x[2]),2), \
                                            np.round(self.x[4], 2), np.round(np.rad2deg(self.x[5]),2)))
            print("s:\t{}".format(np.round(speed, 2)))


    def _measure_speed(self):
        # if node is defined, the speed from thymio is measured
        if self.node != 0:
            # If the speed is measured directly like in the lines afterwards (wait for variables -> measure speed) the python code gets just the last
            # value memorized in its cach. Nevertheless, thymio does not automatically update its cash. Therefore, first a LED is set (at a really
            # low value) and as consequence, the cash is updated. This is not a good solution! However, it is working and because of time constraints 
            # is was not improved.
            aw(self.node.wait_for_variables())
            aw(self.node.set_variables({"leds.top":[0,0,10]}))
            left = self._convert_speed_to_cm(self.node["motor.left.speed"])
            right = self._convert_speed_to_cm(self.node["motor.right.speed"])          
            return [left, right]
        else:
            # if the kalman filter is tested without using the thymio the speed is taken from the speed_list (and not really measured)
            return self.speed_list.pop(0)

    def _convert_speed_to_cm(self, speed):
        # convert measured speed from thymio units back to cm/s
        return (16.0/500.0) * speed

    def _calc_velocity(self, speed):
        # calc. with speed (thymio reference frame) the velocity in x and y direction (absolute reference frame)
        m_vel = np.zeros((3), dtype=float)
        m_vel[0] = math.cos(self.x[2])*(speed[0]+speed[1])/2
        m_vel[1] = math.sin(self.x[2])*(speed[0]+speed[1])/2
        m_vel[2] = (speed[1]-speed[0])/self.thymio_width

        # return measurement variable z, if position was measured by the main (m_pos_flag was set) then the velocity and the positional measurements are merged
        if self.m_pos_flag:
            return np.concatenate((self.m_pos, m_vel), axis=0)
        else:
            return m_vel
        
    def _prediction(self, x, F, Q, P):   
        # calc. state estimation and estimation uncertainty after pretiction step
        x = F@x
        P = F@(P@np.transpose(F)) + Q       
        return x, P
    
    def _correction(self, x, z, H, R, P):
        # measurement prediction covariance
        S = H@(P@np.transpose(H)) + R

        # in case the inverse of S is singular add a small value to its diagonal
        # this step is done that the calculations are stable and has a neglictable influenc on the the result
        if np.linalg.det(S) == 0: 
            S = S + 0.01*np.identity(np.size(R, axis=0))

        # calc. kalman gain
        K = P@(np.transpose(H)@np.linalg.inv(S))              

        # calc inovation
        inovation = z - H@x

        # calc. state estimation and estimation uncertainty after correction step
        x = x + K@inovation

        # calc. estimation uncertainty after correction step, same as P = (I -K@H)@P but less unstable (see “Bucy, R. S. and Joseph, P. D. (1968). 
        # Filtering for Stochastic Processes with Applications to Guidance. Interscience, New York”, Chapter 16, “Roundoff errors” section.)
        I = np.identity(np.size(x))
        P = (I - K@H)@(P@np.transpose(I - K@H)) + K@(R@np.transpose(K))
        return x, P

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
        R_pos = np.array([[self.m_pos_noise,0,0],
                          [0,self.m_pos_noise,0],
                          [0,0,self.m_angular_noise]], dtype=float)

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
    # define initiale speed and orientation
    angle = np.deg2rad(90)
    speed = 5

    # calc. initiale velocity (speed in x and y direction)
    vx = math.cos(angle)*speed
    vy = math.sin(angle)*speed

    # define initiale state vector: [position x, position y, angle, velocity x, velocity y, angular velocity]
    state_vector = np.array([[0,0,angle,vx,vy,0]], dtype=float)

    # define initiale covariance matrix, for the estimated variance of the camera is taken
    uncertainty_matrix = np.array([[[0.01,0,0,0,0,0],
                                    [0,0.01,0,0,0,0],
                                    [0,0,0.03,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]]], dtype=float)

    # if no thymio is used set node equal to zero
    # if node=0 speed is not measured but read from speed_list inside the kalman filter
    node = 0
    
    # define period at which the kalman filter is updated
    period = 0.1

    # create instance of the kalman filter
    filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0])

    # create instance of a repeated timer that calls the fct. update_filter at defined period
    # the kalman filter runs in a separate thread prallel to the main
    t1 = RepeatedTimer(period, filter.update_filter)

    # start the kalman filter
    t1.start()

    # main loop
    for i in range(4):
        # wait for a certain time until main requests next update from the kalman filter
        time.sleep(0.5)

        # get estimated position (state_vector) and uncertainty (covariance matrix) from kalman filter
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

        # calc. second std: probability is more than 95% that robot is inside second std
        eigenvalues, _ = np.linalg.eig(uncertainty_matrix[i+1])
        stds2 = 2*np.sqrt(np.absolute(eigenvalues))

        # if uncertainty (second std) becomes larger than certain threshold the user is ask to enter the true position
        # in the end this step is automated by the camera, but for testing it is done by eye
        if np.amax(stds2) > 3:
            # stop kalman filter
            t1.stop()

            # ask user for true position/pose
            print("second stds: {}".format(stds2))
            x_pos = float(input("enter x position: "))
            y_pos = float(input("enter y_position: "))
            angle = np.deg2rad(float(input("enter angle [in °]: ")))

            # send entered values to kalman filter which will include them in the next correction step
            filter.set_position_measurement([x_pos, y_pos, angle])

            #start again kalman filter
            t1.start()

    # stopp kalman filter
    t1.stop()

    # plot results
    PlotMap(period=0.5, position_list=state_vector, cov_list=uncertainty_matrix)