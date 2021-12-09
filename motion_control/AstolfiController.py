import numpy as np
import math as m
import matplotlib.pyplot as plt
import utilities as ut
from Thymio import Thymio
import time
import serial

class Astolfi:

    def __init__(self,curr=np.array((0,0,0)), on_goal=False):
        '''
        Constructer of the Thymio Robot which contains following parameters
        lspeed : left wheel motor speed, as default set to 100]
        rspeed : right wheel motor speed, as default set to 100
        curr_pos : current position, as initial it taken as (0,0,0) indicating (x,y,q)
        '''   
        self.curr = curr
        self.on_goal = on_goal
        self.next_index = 0

        self.rho = 0
        self.alpha = 0
        self.beta = 0

        #entering initial parameters
        self.wheel_radius = ut.THYMIO_PARAMS['WHEEL_RAD']          #wheel radius of thymio [cm]
        self.wheel_length = ut.THYMIO_PARAMS['WHEEL_LENGTH']       #distance of wheels from each other [cm] 
        self.max_speed    = ut.THYMIO_PARAMS['MAX_SPEED']          #maximum thymio wheel speed
        self.max_speed_cms= ut.THYMIO_PARAMS['MAX_SPEED_CM']       #maximum thymio wheel speed [cm/s]
        self.cm2thym      = ut.THYMIO_PARAMS['SPEED_CM2THYM']      #maximum thymio wheel speed
        self.Ts           = ut.THYMIO_PARAMS['SAMPLING_TIME']

        #defining thresholds for controller
        self.on_node = ut.THRESHOLDS['ON_NODE']        

        #setting astolfi gains
        self.Kr = ut.ASTOLFI_PARAM['K_RHO']
        self.Ka = ut.ASTOLFI_PARAM['K_ALPHA']
        self.Kb = ut.ASTOLFI_PARAM['K_BETA']

    def set_curr(self,curr):
        self.curr = curr

    def get_curr(self):
        return self.curr

    def set_goal(self,next):
        self.next = next
    
    def get_goal(self):
        return self.next

    def get_global(self):
        return self.path

    def check_nodes(self,verbose=False):

        if self.rho < self.on_node and self.next_index < len(self.path)-1:
            self.next_index+=1
            self.set_goal(self.path[self.next_index])
            if verbose: 
                print('proceeding to next node')
                print('current goal is', self.next)
        elif self.next_index >= len(self.path)-1 and self.rho < self.on_node:
            self.on_goal = True
        

    def set_path(self,traj,verbose=False):
        j=1
        path = []
        path.append([traj[0][0], traj[0][1], 0])

        while j < len(traj):
            pos1=traj[j]
            pos2=traj[j-1]
            temp=[pos1[0]-pos2[0],pos1[1]-pos2[1]]
            ang=m.atan2(temp[1],temp[0])
            path.append([pos1[0], pos1[1], ang])
            j+=1    
        self.path=np.array(path) 
        self.set_goal(self.path[self.next_index])
        if verbose: print(self.path)      

    def delt(self, pos1, pos2):
        return np.array([pos2[0]-pos1[0],pos2[1]-pos1[1]])

    def norm(self,vec):
        return m.sqrt(vec[0]**2+vec[1]**2)

    def normalize_ang(self, angle):
        return m.atan2(m.sin(angle), m.cos(angle))

    def polar_rep(self):

        delta = self.delt(self.curr,self.next)        
        self.rho = self.norm(delta)
        self.alpha = self.normalize_ang(-self.curr[2] + m.atan2(delta[1],delta[0]))
        self.beta = self.normalize_ang(-self.alpha-self.curr[2]-self.next[2])

        return self.rho, self.alpha, self.beta

    def scale(self,left,right):
        left = left/100*self.max_speed_cms
        right = right/100*self.max_speed_cms

        return left,right

    def compute_phi_dot(self, fkine=False):

        vel = self.Kr*self.rho
        # vel = self.saturate(vel,self.max_speed_cms)

        omega = self.Ka*self.alpha + self.Kb*self.beta

        self.phiL = (vel + omega*self.wheel_length)/self.wheel_radius
        self.phiR = (vel - omega*self.wheel_length)/self.wheel_radius

        # self.phiL, self.phiR = self.scale(self.phiL, self.phiR )

        self.phiL = self.saturate(self.phiL,self.max_speed_cms)
        self.phiR = self.saturate(self.phiR,self.max_speed_cms)

        if fkine:
            q_dot = omega
            v=(self.phiL+self.phiR)*self.wheel_radius/2
            self.curr[2] += q_dot*self.Ts/2
            x_dot = v*m.cos(self.curr[2])
            y_dot = v*m.sin(self.curr[2])

            self.curr[0]+=x_dot*self.Ts/2
            self.curr[1]+=y_dot*self.Ts/2

        return self.phiL, self.phiR


    def saturate(self,speed,limit):

        speed = min(speed,  limit)
        speed = max(speed, -limit)

        return speed

    def speed4thymio(self,left,right):
        left = left if left > 0  else 2**16-1+left
        right = right if right > 0 else 2**16-1+right

        return left,right

    def run_on_thymio(self,th):
        
        left  = int( self.saturate(self.phiL*self.cm2thym, self.max_speed))
        right = int( self.saturate(self.phiR*self.cm2thym, self.max_speed))

        left, right = self.speed4thymio(left,right)

        th.set_var("motor.left.target", left)
        th.set_var("motor.right.target", right)


