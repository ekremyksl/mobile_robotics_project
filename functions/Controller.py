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

    def __init__(self,node=None):
        '''
        Constructer of the Thymio Robot which contains following parameters
        lspeed : left wheel motor speed, as default set to 100]
        rspeed : right wheel motor speed, as default set to 100
        curr_pos : current position, as initial it taken as (0,0,0) indicating (x,y,q)
        '''   
        self.on_goal = False
        self.node_index = 0     #flag to keep track of the index of next goal position

        self.rho = 0
        self.alpha = 0
        self.beta = 0

        self.state = 'HEADING'     #initial state of Thymio
        self.node = node

        #entering initial parameters
        self.wheel_radius = ut.THYMIO_PARAMS['WHEEL_RAD']          #wheel radius of thymio [cm]
        self.wheel_length = ut.THYMIO_PARAMS['WHEEL_LENGTH']       #distance of wheels from each other [cm] 
        self.max_speed    = ut.THYMIO_PARAMS['MAX_SPEED']          #maximum thymio wheel speed
        self.max_speed_cms= ut.THYMIO_PARAMS['MAX_SPEED_CM']       #maximum thymio wheel speed [cm/s]
        self.cm2thym      = ut.THYMIO_PARAMS['SPEED_CM2THYM']      #maximum thymio wheel speed
        self.Ts           = ut.THYMIO_PARAMS['SAMPLING_TIME']

        #defining thresholds for controller
        self.on_node = ut.THRESHOLDS['ON_NODE']
        self.heading_th = ut.THRESHOLDS['HEADING_THRESHOLD']  #sensory threshold for obstacle avoidance
        self.obst_th_low = ut.THRESHOLDS['OBSTACLE_TH_LOW']     #obstacle lower threshold 
        self.obst_th_high = ut.THRESHOLDS['OBSTACLE_TH_HIGH']   #obstacle high threshold

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

    #differenc vector (P2-P1)
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

    #checks if next goal or even global goal is reached
    def check_node(self, verbose=False):
        #to initialize the next node
        if self.node_index == 0:
            self.curr = self.path[0]
            self.node_index = 1
            self.next = self.path[1]
            if self.norm(self.next-self.curr) < self.on_node:
                self.on_goal = True
        elif self.node_index < len(self.path)-1 and self.norm(self.path[self.node_index]-self.curr) < self.on_node:
            self.node_index+=1
            self.next = self.path[self.node_index]
            if verbose: print('Controller is proceeding to next node:', self.next)
        elif self.node_index >= len(self.path)-1 and self.norm(self.next-self.curr) <= self.on_node:
            self.on_goal = True
            if verbose: print('Global Goal Is Reached')

    #process the optimal trajectory coming from global navigation
    #list of nodes [(x,y),..] is given and functions converts it to
    #np.array with the third element, angle of the line between nodes np.array((x,y,q))
    def set_global(self,traj,angle,conversion='mm2cm',verbose=False):
        #depending on unit of pixel to cm, mm, or else
        if conversion == 'mm2cm' or conversion == 'cm2m':
            conversion_coeff = 0.1
        elif conversion == 'm2cm':
            conversion_coeff = 100
        elif conversion == 'cm2m':
            conversion_coeff = 0.01
        else:  #else it is assumed to be cm2cm
            conversion_coeff = 1

        path = []
        path.append([traj[0][0], traj[0][1], angle])    #adding the first node to path (initial position of Thymio )
        
        j=1
        while j < len(traj):
            delta = self.delt(traj[j-1], traj[j])   #distance vector between two consecutive nodes
            ang = m.atan2(delta[1], delta[0])
            path.append([traj[j][0], traj[j][1], ang])
            j+=1
        #adding the last node as dummy node
        path.append([traj[-1][0], traj[-1][1], ang])
        self.path = np.array(path) * conversion_coeff 
        if verbose: print('Global Trajectory:', self.path)
        #adding next goal
        self.next = self.path[1]
        self.node_index = 1

    #if orientation error, alpha is not within [-pi/2, pi/2] , then we first rotate robot to avoid
    # undesired motion of Astolfi controller   
    def correct_heading(self,fkine=False,verbose=False):

        if verbose: print('correcting heading')
        #depending on orientation error (alpha), deciding the rotation direction
        if self.alpha > 0:
            omega = 1
        else: 
            omega = -1
        #linear velocity is zero since we only want it to turn around robot frame
        vel = 0
        #wheel speeds in [cm/s]
        # assigning wheel speeds        
        self.phiL = ((vel - omega*self.wheel_length)/self.wheel_radius)
        self.phiR = ((vel + omega*self.wheel_length)/self.wheel_radius)

        return self.phiL,self.phiR        

    #polar coordinates for astolfi controller
    def polar_rep(self):

        delta = self.delt(self.curr,self.next)        
        self.rho = self.norm(delta)                                                     #rho = sqrt(dx^2+dy^2)
        self.alpha = self.normalize_ang(-self.curr[2] + m.atan2(delta[1],delta[0]))     #alpha = atan2(dy,dx)-theta
        self.beta = self.normalize_ang(-self.alpha-self.curr[2]-self.next[2])           #beta = -theta-alpha-beta_ref  where beta_ref is the angle of approach to the next node [as next[2]]

        return self.rho, self.alpha, self.beta

    def compute_phi_dot(self, fkine=False, verbose=False):
        #computing rho, alpha, beta
        self.polar_rep()
        if verbose: print('Astolfi Controller')
        vel = self.Kr                                   #for constant linear speed
        omega = self.Ka*self.alpha + self.Kb*self.beta
        omega /=(self.rho)                              #normalizing omega with rho for constant linear speed

        #wheel speeds in [cm/s]
        #using inverse kinematics  to obtain wheel speeds using the relation
        #vel = (left+right)*(r/2)
        #omega = (-left+right)*(r/2l)

        self.phiL = ((vel - omega*self.wheel_length)/self.wheel_radius)     
        self.phiR = ((vel + omega*self.wheel_length)/self.wheel_radius)

        #saturating the input variables to ensure safety of motors
        self.phiL = self.saturate(self.phiL,self.max_speed_cms)
        self.phiR = self.saturate(self.phiR,self.max_speed_cms)

        #in case no estimator (Kalman Filter) is used, this part enables controller to be used by itself
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

    def get_sensoryprox(self):
        aw(self.node.wait_for_variables({"prox.horizontal"}))
        obst = [self.node["prox.horizontal"][0], self.node["prox.horizontal"][4]]

        return obst

    def obstacle_aviodance(self,obst):
        speed0 = 50
        obstSpeedGain = 5

        leds_top = [30,30,30]
        # obstacle avoidance: accelerate wheel near obstacle
        vl = (speed0 + obstSpeedGain * (obst[0] // 100)) / 16
        vr = (speed0 + obstSpeedGain * (obst[1] // 100)) / 16

        return vl,vr

    def motion_control(self, astolfi=False, verbose=False, fkine=False):

        self.polar_rep()
        obst = self.get_sensoryprox()

        
        if verbose:
            print('Orientation error is:', self.alpha)
            print('Distance from the next node is:', self.rho)
        #deciding the state of the robot

        
        #should it correct heading
        if abs(self.alpha) > (m.pi/4): #Astolfi is able to control it if alpha is in [-pi/2,pi/2]
            self.state = 'HEADING'               #but as a safety margin, we take 0.95(pi/2) as threshold for heading correction
        #or should it follow the trajectory with Astolfi
        elif abs(self.alpha) and astolfi is not False:
            self.state = 'ASTOLFI' 
        #or do nothing and stop the robot at the moment for safety 
        else:
            self.state = 'PASS'

        if self.state != 'AVOIDANCE': 
            # switch from goal tracking to obst avoidance if obstacle detected
            if (obst[0] > self.obst_th_high) or (obst[1] > self.obst_th_high) :
                self.state = 'AVOIDANCE'
        elif self.state == 'AVOIDANCE':
            if obst[0] < self.obst_th_low and obst[1] < self.obst_th_low:
                # switch from obst avoidance to goal tracking if obstacle got unseen
                self.state = 'ASTOLFI' 

        if verbose: print('Current State is', self.state)
        #depending on the state, determine the action of robot
        if self.state == 'AVOIDANCE':
            vl,vr = self.obstacle_aviodance(obst)
        elif self.state == 'HEADING':
            vl,vr = self.correct_heading(fkine=fkine,verbose=verbose)
        elif self.state == 'ASTOLFI':
            vl,vr = self.compute_phi_dot(fkine=fkine,verbose=verbose)
        else:
            vl=0.0
            vr=0.0


        #send inputs to Thymio
        aw(self.node.set_variables(self.motors(vl,vr)))
        if self.state == 'AVOIDANCE': time.sleep(0.5)
        #check if next goal or even global goal is reached
        self.check_node()

#debuggin or running by itself
if __name__ == '__main__':
        
        trajectory = [(0.0,0.0), (50.0,40.0), (70.0,80.0),(0.0, 80.0), (0.0, 0.0)]        

        # set up connection to thymio, if node=0 no thymio is connected
        # client = ClientAsync()
        # node = aw(client.wait_for_node())
        # aw(node.lock())

        cont = Controller()

        cont.set_global(trajectory,np.deg2rad(0.0),verbose=True)
        i=0
        curr = np.array((0.0,0.0,0.0))
        curr[2]=cont.normalize_ang(curr[2])
        cont.set_curr(curr[0:3])
        pos=[]
        pba=[]
        phi_dot=[]
        while cont.on_goal==False:
            time.sleep(0.2)            

            cont.motion_control(astolfi=False,fkine=True, verbose=True)
            i+=1
            pos.append([cont.curr])
            phi_dot.append([cont.phiL,cont.phiR])
            pba.append([cont.rho,cont.alpha,cont.beta])
               

        
        print('****GOAL IS REACHED****')

        pos=np.array(pos)
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
        


            
