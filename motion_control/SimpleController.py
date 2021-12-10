from cv2 import sqrt
import numpy as np
import math
import time
import utilities as ut
import serial
import Thymio
from tdmclient import ClientAsync, aw

class SimpleController:

    def __init__(self, 
                 curr=np.array((0.0,0.0,0.0)),
                 global_path=None):
        self.curr = curr
        self.global_path = global_path
        self.node_index = 0
        self.on_goal = False
        self.phiL = 0
        self.phiR = 0
        self.pos = []
        #setting hyperparameters
        self.set_physParams()
        self.set_thresholds()

    def set_thymio(self,th):
        self.th = th

    def set_physParams(self):
        self.wheel_radius = ut.THYMIO_PARAMS['WHEEL_RAD']          #wheel radius of thymio [cm]
        self.wheel_length = ut.THYMIO_PARAMS['WHEEL_LENGTH']       #distance of wheels from each other [cm] 
        self.max_speed    = ut.THYMIO_PARAMS['MAX_SPEED']          #maximum thymio wheel speed
        self.max_speed_cms= ut.THYMIO_PARAMS['MAX_SPEED_CM']       #maximum thymio wheel speed [cm/s]
        self.cm2thym      = ut.THYMIO_PARAMS['SPEED_CM2THYM']      #maximum thymio wheel speed
        self.Ts           = ut.THYMIO_PARAMS['SAMPLING_TIME']

    def set_thresholds(self): 
        self.heading_th = ut.SIMPLE_CONT['HEADING_THRESHOLD']  #sensory threshold for obstacle avoidance
        self.dist_th    = ut.SIMPLE_CONT['DISTANCE_THRESHOLD']             #threshold to determine if we are on the node

    def set_curr(self,curr):
        self.curr=curr

    def get_curr(self):
        return self.curr

    def set_global(self,traj,angle):
        global_path = []
        global_path.append([traj[0][0], traj[0][1], angle])
        j=1
        while j<len(traj):
           pos1=traj[j]
           pos2=traj[j-1]
           temp=[pos1[0]-pos2[0],pos1[1]-pos2[1]]
           ang=math.atan2(temp[1],temp[0])
           global_path.append([pos1[0], pos1[1], ang])
           j+=1
        global_path.append([pos1[0], pos1[1], ang])
        self.global_path=np.array(global_path)/10
        self.global_path=self.global_path
        self.check_node()

    def check_node(self):
        
        if self.node_index==0: 
            self.node_index+=1
            self.set_goal(self.global_path[self.node_index])
        delt = self.global_path[self.node_index]-self.curr
        dist = math.sqrt(delt[0]**2+delt[1]**2)        
        if self.node_index<len(self.global_path)-1 and dist < 4:            
            self.node_index+=1
            self.set_goal(self.global_path[self.node_index])
            print('simp is proceeding to next node',self.next)
        elif self.node_index < len(self.path)-1:
            print('goal reached by simp')
            self.on_goal=True

    def goalReached(self):
        return self.on_goal

    def set_goal(self,next):
        self.next=next

    def get_goal(self):
        return self.next

    def set_speed(self, lspeed, rspeed):
        self.phiL = lspeed
        self.phiR = rspeed

    def spd4thym(self,left,right):
        left = left % 2**16 #abs(left) if left > 0 else 2**16-abs(left)
        right = right % 2**16 #abs(right) if right < 0 else 2**16-abs(right)
        return left,right    

    def normalize_ang(self,angle):   #ensuring theta is between pi and -pi
        return math.atan2(math.sin(angle),math.cos(angle))

    def correct_heading(self,error,fkine=False,verbose=False):

        if verbose: print('correcting heading')
        if abs(error) > np.pi / 6:
            spd = 3
        else:
            spd = 1
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
                
        
        # dt = (self.next[2]-self.curr[2])*self.wheel_length*5/16.7
        # if dt is not 0:
        #    th.set_var("motor.left.target", 100)
        #    th.set_var("motor.right.target", 2**16-100)
        
        # time.sleep(abs(dt))
        while abs(self.next[2]-self.curr[2]) > self.heading_th:
            if self.next[2]-self.curr[2] > self.heading_th:
                self.set_speed(150,-150)
                self.run_on_thymio(self.th)
                #finding angle each iteration
                if fkine:
                    dq=self.wheel_radius/(2*self.wheel_length)*(self.phiL-self.phiR)/self.cm2thym
                    self.curr[2]+=dq*self.Ts/2
                    self.curr[2]=self.normalize_ang(self.curr[2])
            else:
                self.set_speed(-150,150)
                self.run_on_thymio(self.th)
                #finding angle each iteration
                if fkine:
                    dq=self.wheel_radius/(2*self.wheel_length)*(self.phiL-self.phiR)/self.cm2thym
                    self.curr[2]+=dq*self.Ts/2
                    self.curr[2]=self.normalize_ang(self.curr[2])
            # self.update_pos()
            # print(self.curr)    
            time.sleep(self.Ts)
        print('heading corrected')
        self.set_speed(0,0)
        self.run_on_thymio(self.th)
   

    def run_on_thymio(self,th):
        ''' give values to the motors of the thymio'''
        # left  = int(self.phiL * self.cm2thym)
        # right = int(self.phiR * self.cm2thym)

        #left,right = self.check_limits(left,right)
        left,right = self.spd4thym(self.phiL,self.phiR)

        # th.set_var("motor.left.target", left)
        # th.set_var("motor.right.target", right)
        aw(th.set_variables(self.motors(left,right)))

    def motors(left, right):
        return { "motor.left.target": [left], "motor.right.target": [right] }

    def compute_dist(self,next,curr):
        delta = [next[0]-curr[0],next[1]-curr[1]]
        dist = math.sqrt(delta[0]**2+delta[1]**2)
        return dist

    def update_pos(self):
        self.pos.append([self.curr[0],self.curr[1], self.curr[2]])

    def follow_line(self,dist,error,fkine=False,verbose=False):

        if verbose: print('following line')
        corr = error / self.heading_th * 1
        print(corr)
        if dist > self.dist_th:
            spd = 3
            self.set_speed(spd-corr,spd+corr)
        else:
            self.set_speed(0,0)
        if verbose: print('following line completed')

        return self.phiL, self.phiR
        
        dist = self.compute_dist(self.next,self.curr)
        while dist > self.dist_th and dist <= 4:
            self.set_speed(250,250)
            # self.run_on_thymio(self.th)
            if fkine:
                #finding positions each iteration
                vel=self.wheel_radius*(self.phiL+self.phiR)/(2*self.cm2thym)
                dx=vel*math.cos(self.curr[2])
                dy=vel*math.sin(self.curr[2])
                self.curr[0]+=dx*self.Ts/2
                self.curr[1]+=dy*self.Ts/2
                self.update_pos()
            # print(self.curr)
            #finding distance for next iteration
            dist = self.compute_dist(self.next,self.curr)
            time.sleep(self.Ts)
        print('following line completed')
        self.set_speed(0,0)
        self.run_on_thymio(self.th)
        
     


