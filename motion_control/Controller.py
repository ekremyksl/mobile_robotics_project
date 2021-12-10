from cv2 import norm
import numpy as np
import math as m
import matplotlib.pyplot as plt
import utilities as ut
from Thymio import Thymio
import time
import serial
from tdmclient import ClientAsync, aw

class Controller:

    def __init__(self):
        '''
        Constructer of the Thymio Robot which contains following parameters
        lspeed : left wheel motor speed, as default set to 100]
        rspeed : right wheel motor speed, as default set to 100
        curr_pos : current position, as initial it taken as (0,0,0) indicating (x,y,q)
        '''   
        self.on_goal = False
        self.node_index = 0

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
        self.heading_th = ut.SIMPLE_CONT['HEADING_THRESHOLD']  #sensory threshold for obstacle avoidance
        self.dist_th    = ut.SIMPLE_CONT['DISTANCE_THRESHOLD']             #threshold to determine if we are on the node
 

        #setting astolfi gains
        self.Kr = ut.ASTOLFI_PARAM['K_RHO']
        self.Ka = ut.ASTOLFI_PARAM['K_ALPHA']
        self.Kb = ut.ASTOLFI_PARAM['K_BETA']

    #setting the current node with estimate coming from Kalman
    def set_curr(self,curr):
        self.curr=curr

    #setting the next goal for robot
    def set_goal(self,next):
        self.next=next

    def delt(self, pos1, pos2):
        return np.array([pos2[0]-pos1[0],pos2[1]-pos1[1]])

    def norm(self,vec):
        return m.sqrt(vec[0]**2+vec[1]**2)

    #normalizing angle between -pi and pi
    def normalize_ang(self, angle):
        return m.atan2(m.sin(angle), m.cos(angle))

    def set_speed(self, lspeed, rspeed):
        self.phiL = lspeed
        self.phiR = rspeed

    #saturating the speeds whether
    #to keep it general, limit is also given to function
    def saturate(self,speed,limit):

        speed = min(speed,  limit)
        speed = max(speed, -limit)

        return speed

    #sends the speed command to thymio
    def motors(self, left, right):
        left = int((500/16)*left)
        right = int((500/16)*right)
        return { "motor.left.target": [left], "motor.right.target": [right] }


    def run_on_thymio(self,th):
        ''' give values to the motors of the thymio'''
        # left  = int(self.phiL * self.cm2thym)
        # right = int(self.phiR * self.cm2thym)

        # left,right = self.check_limits(left,right)
        # left,right = self.spd4thym(self.phiL,self.phiR)

        # th.set_var("motor.left.target", left)
        # th.set_var("motor.right.target", right)
        # aw(th.set_variables(self.motors(left,right)))
       

    #checks if next goal or even global goal is reached
    def check_node(self, verbose=False):
        #to initialize the next node
        if self.node_index == 0 and self.norm(self.path[1]-self.path[0]) > self.dist_th:
            self.node_index = 1
            self.next = self.path[self.node_index]
        elif self.node_index < len(self.path)-1 and self.norm(self.next-self.curr) < self.dist_th:
            self.node_index+=1
            self.next = self.path[self.node_index]
            if verbose: print('Controller is proceeding to next node:', self.next)
        elif self.node_index >= len(self.path)-1 and self.norm(self.next-self.curr) <= self.on_node:
            self.on_goal = True
            if verbose: print('Global Goal Is Reached')

    #process the optimal trajectory coming from global navigation
    #list of nodes [(x,y),..] is given and functions converts it to
    #np.array with the third element, angle of the line between nodes np.array((x,y,q))
    def set_global(self,traj,angle,verbose=False):

        path = []
        path.append([traj[0][0], traj[0][1], angle])
        
        j=1
        while j < len(traj):
            delta = self.delt(traj[j-1], traj[j])   #distance vector between two consecutive nodes
            ang = m.atan2(delta[1], delta[0])
            path.append([traj[j][0], traj[j][1], ang])
            j+=1
        #adding the last node as dummy node
        path.append([traj[-1][0], traj[-1][1], ang])
        self.path = np.array(path) / 10 #converting from mm to cm
        if verbose: print('Global Trajectory:', self.path)
        self.check_node()

    
    def correct_heading(self,error,fkine=False,verbose=False):

        if verbose: print('correcting heading')
        if abs(error) > np.pi / 6:
            spd = 3
        else:
            spd = 1.5
        if error < self.heading_th:
                self.set_speed(spd,-spd)
        else:
                self.set_speed(-spd,spd)    
        if fkine:
                dq=self.wheel_radius/(2*self.wheel_length)*(self.phiL-self.phiR)/self.cm2thym
                self.curr[2]+=dq*self.Ts/2
                self.curr[2]=self.normalize_ang(self.curr[2])

        if verbose: print('heading corrected')
        
        return self.phiL,self.phiR

    def follow_line(self,dist,error,fkine=False,verbose=False):

        if verbose: print('following line')
        corr = error / self.heading_th * 1
        print(corr)
        if dist > self.on_node:
            spd = 3
            self.set_speed(spd-corr,spd+corr)
        else:
            self.set_speed(0,0)
        if verbose: print('following line completed')

        return self.phiL, self.phiR

    #polar coordinates for astolfi controller
    def polar_rep(self):

        delta = self.delt(self.curr,self.next)        
        self.rho = self.norm(delta)
        self.alpha = self.normalize_ang(-self.curr[2] + m.atan2(delta[1],delta[0]))
        self.beta = self.normalize_ang(-self.alpha-self.curr[2]-self.next[2])

        return self.rho, self.alpha, self.beta

    def compute_phi_dot(self, fkine=False, verbose=False):

        self.polar_rep()
        if verbose: print('Astolfi Controller')
        vel = self.Kr*self.rho
        omega = self.Ka*self.alpha + self.Kb*self.beta

        #wheel speeds in [cm/s]
        self.phiL = ((vel + omega*self.wheel_length)/self.wheel_radius)
        self.phiR = ((vel - omega*self.wheel_length)/self.wheel_radius)

        # self.phiL, self.phiR = self.scale(self.phiL, self.phiR )

        self.phiL = self.saturate(self.phiL,self.max_speed_cms)
        self.phiR = self.saturate(self.phiR,self.max_speed_cms)

        #in case no estimator is used, this part enables controller to be used by itself
        if fkine:
            q_dot = omega
            v=(self.phiL+self.phiR)*self.wheel_radius/2
            self.curr[2] += q_dot*self.Ts/2
            x_dot = v*m.cos(self.curr[2])
            y_dot = v*m.sin(self.curr[2])

            self.curr[0]+=x_dot*self.Ts/2
            self.curr[1]+=y_dot*self.Ts/2

        if verbose: print('wheel speeds are:',self.phiL,self.phiR)
        return self.phiL, self.phiR

    def motion_control(self, node, astolfi=False, verbose=False):

        orientation_error = self.normalize_ang(self.next[2]-self.curr[2])
        dist = self.norm(self.curr-self.next)
        # state = 'PASS'
        if verbose:
            print('Orientation error is:', orientation_error)
            print('Distance from the next node is:', dist)
        #deciding the state of the robot
        #should it correct heading
        if abs(orientation_error) > self.heading_th:
            state = 'HEADING' 
        #should it follow line if it is close to node
        elif abs(orientation_error) <= self.heading_th and dist < self.dist_th and dist > self.on_node:
            state = 'FOLLOW LINE'
        #or should it follow the trajectory with Astolfi
        else:
            state = 'ASTOLFI' if astolfi is not False else 'PASS'
        
            
        print(state)

        if state == 'HEADING':
            vl,vr = self.correct_heading(orientation_error,verbose)
        elif state == 'FOLLOW LINE':
            vl,vr = self.follow_line(dist,orientation_error,verbose)
        elif state == 'ASTOLFI':

            vl,vr = self.compute_phi_dot(verbose=True)
        else:
            vl=0
            vr=0

        aw(node.set_variables(self.motors(vl,vr)))
        self.check_node()

 

        




            
