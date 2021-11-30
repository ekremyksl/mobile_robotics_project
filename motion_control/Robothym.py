import os
import sys
import math
from statistics import NormalDist, mean
import pandas as pd
import numpy as np
#import tdmclient.notebook


class Robothym:

    def _init_(self, lspeed=0, rspeed=0, astolfi_param=(1,15,-1.5), curr_pos=(0,0,0)):
        '''
        Constructer of the Thymio Robot which contains following parameters
        lspeed : left wheel motor speed, as default set to 100
        rspeed : right wheel motor speed, as default set to 100
        curr_pos : current position, as initial it taken as (0,0,0) indicating (x,y,q)
        '''
        #thymio default parameters and thresholds
        self.wheel_radius = 0.32
        self.wheel_length = 11 #distance of wheels from each other [cm] 
        self.max_speed = 500   #maximum thymio wheel speed
        self.pos_th = 0.001    #
        #setting wheel speeds
        self.phiL = lspeed
        self.phiR = rspeed
        #position and trajectory parameters
        self.curr_pos = curr_pos
        self.node_index = 0
        #self.wheel_radius =

        #Astolfi parameters
        self.rho   = None
        self.alpha = None
        self.beta  = None

        #obstacle avoidance threshold
        self.obs_th = 500 
        #astolfi controller parameters
        self.kr = astolfi_param[0]
        self.ka = astolfi_param[1]
        self.kb = astolfi_param[2]

    def set_speed(self, lspeed, rspeed):
        #setting motor speeds to given parameters
        self.phiL = lspeed
        self.phiR = rspeed
    
    #setting the sampling time
    def set_steptime(self, Ts):
        self.Ts = Ts

    #givin global path, the trajectory to thymio
    def set_globalpath(self, globpath):
        self.traj = globpath

    def setCurrPos(self, curr):
        #setting current position to given value
        self.curr_pos=curr

    def getCurrPos(self):
        #returning current position
        return self.curr_pos

    def turn(self, angle, coords):
            R = np.array(((np.cos(angle), -np.sin(angle)),
                  (np.sin(angle),  np.cos(angle))))

    def setAstolfiA(self):
        self.A_ast = np.array((-self.kr,    0,          0),
                                (0, -self.ka+self.kr, -self.kb),
                                 0, -self.kr,            0)

    #finding speeds in polar coordinates
    def get_rho_alpha_beta_dot(self):
        return np.array(-self.kr*self.rho*math.cos(self.alpha),                             #rho_dot
                        -self.kr*math.sin(self.alpha)-self.ka*self.alpha-self.kb*self.beta, #alpha_dot
                        -self.kr*math.sin(self.alpha) )                                     #beta_dot

    #saturating inputs incase wheel speeds exceeds maximum speed
    #checking the limit of inputs before giving to Thymio
    def check_limits(self):
        #left wheel
        self.phiL =  self.max_speed if self.phiL > self.max_speed else self.phiL #larger then 500
        self.phiL = -self.max_speed if self.phiL <-self.max_speed else self.phiL #smaller than -500
        #right wheel  
        self.phiR =  self.max_speed if self.phiR > self.max_speed else self.phiR #larger than -500
        self.phiR = -self.max_speed if self.phiR <-self.max_speed else self.phiR #smaller than -500 


    def ikine(self):
        #finding polar coordinates for Astolfi controller
        #dx, dy, theta
        delta=self.compute_dist(self.curr_pos,self.next)
        theta=self.curr_pos[2]
        #polar coordinates alpha, beta, rho
        self.rho = math.sqrt(delta[0]**2 + delta[1]**2)
        self.alpha = -theta + math.atan2(delta[1],delta[0])
        self.beta = -theta - self.alpha
        #velocities v and w
        vel = self.kr*self.rho
        omega = self.ka*self.alpha + self.kb*self.beta
        #conversion to cartesian speeds
        x_dot = vel*math.cos(theta)
        y_dot = vel*math.sin(theta)

        #finding corresponding wheel speeds
        phi_L = (x_dot*math.cos(theta) + y_dot*math.sin(theta) + self.wheel_length*omega)/self.wheel_radius
        phi_R = (x_dot*math.cos(theta) + y_dot*math.sin(theta) - self.wheel_length*omega)/self.wheel_radius

    #checking if we reached the next goal
    def check_pos(self):
        delta = self.compute_dist(self.curr, self.next)
        if  delta < self.pose_th:
            if self.compute_dist(self.next,self.set_globalpath[-1]) < 0.001:
                self.node_index = -1
                self.next = self.set_globalpath[-1]
                return 0
            else: 
                self.node_index=+1  
                self.next = self.set_globalpath[self.node_index]
                delta = self.compute_dist(self.curr, self.next) 
        return delta 
            
    #computing delta
    def compute_dist(self, curr, next):
        return next[0]-self.curr[0]
        