import math
import numpy as np
import sys
sys.path.append('mobile_robotics_project/utilities.py')
import utilities as ut
import scipy.linalg as linal


#[TODO]: need to implement sender function to thymio and test it on the real system
#[TODO]: add more comments to code

class Robothym:

    def __init__(self, lspeed=0, rspeed=0, curr_pos=np.array([0,0,0])):
        '''
        Constructer of the Thymio Robot which contains following parameters
        lspeed : left wheel motor speed, as default set to 100]
        rspeed : right wheel motor speed, as default set to 100
        curr_pos : current position, as initial it taken as (0,0,0) indicating (x,y,q)
        '''
        # #setting wheel speeds
        # self.thymio = th
        self.phiL = lspeed
        self.phiR = rspeed
        #position and trajectory parameters
        self.curr_pos = curr_pos
        #Astolfi parameters
        self.rho   = None
        self.alpha = None
        self.beta  = None
        #astolfi controller parameters
        self.node_index = 0
        self.set_astolfiGains()
        #thymio default parameters and thresholds
        self.set_physParams()
        self.set_thresholds()
        # self.set_spatial_gains()
        self.turn=False
        

    #defining physical parameters of Thymio
    def set_physParams(self):
        self.wheel_radius = ut.THYMIO_PARAMS['WHEEL_RAD']          #wheel radius of thymio [cm]
        self.wheel_length = ut.THYMIO_PARAMS['WHEEL_LENGTH']       #distance of wheels from each other [cm] 
        self.max_speed    = ut.THYMIO_PARAMS['MAX_SPEED']          #maximum thymio wheel speed
        self.max_speed_cms= ut.THYMIO_PARAMS['MAX_SPEED_CM']       #maximum thymio wheel speed [cm/s]
        self.cm2thym      = ut.THYMIO_PARAMS['SPEED_CM2THYM']      #maximum thymio wheel speed
        self.Ts           = ut.THYMIO_PARAMS['SAMPLING_TIME']


    #setting Astolfi Controller Gains
    def set_astolfiGains(self):
        self.kr = ut.ASTOLFI_PARAM['K_RHO']
        self.ka = ut.ASTOLFI_PARAM['K_ALPHA']
        self.kb = ut.ASTOLFI_PARAM['K_BETA']

    def set_spatial_gains(self):
        self.K_theta = ut.ASTOLFI_PARAM['HEADING']
        self.K_parallel = ut.ASTOLFI_PARAM['PARALLEL']
        self.K_normal = ut.ASTOLFI_PARAM['NORMAL']

    #setting thresholds for obstacle avoidance and node arrived
    def set_thresholds(self): 
        self.avoidance_th = ut.THRESHOLDS['OBJ_AVOIDANCE']  #sensory threshold for obstacle avoidance
        self.on_node = ut.THRESHOLDS['ON_NODE']             #threshold to determine if we are on the node
        
    #setting motor speeds to given parameters
    def set_speed(self, lspeed, rspeed):
        self.phiL = lspeed
        self.phiR = rspeed
    
    #setting the sampling time
    def set_steptime(self, Ts):
        self.Ts = Ts

    #givin global path, the trajectory to thymio
    def set_globalpath(self, globpath):
        self.path = globpath
    
    #setting a goal
    def set_goal(self,goal):
        self.next=goal

    #setting current position to given value    
    def setCurrPos(self, curr):
        self.curr_pos=curr

    def set_next(self,next):
        self.next=next
        
    #returning current position
    def getCurrPos(self):
        return np.array([self.curr_pos[0],self.curr_pos[1],self.curr_pos[2]])

    def turn(self, angle, coords):
            R = np.array(((np.cos(angle), -np.sin(angle)),
                         (np.sin(angle),  np.cos(angle))))

    #finding speeds in polar coordinates
    def get_rho_alpha_beta_dot(self):
        return np.array((-self.kr*self.rho*math.cos(self.alpha),                             #rho_dot
                         -self.kr*math.sin(self.alpha)-self.ka*self.alpha-self.kb*self.beta, #alpha_dot
                         -self.kr*math.sin(self.alpha)))                                     #beta_dot

    #saturating inputs incase wheel speeds exceeds maximum speed
    #checking the limit of inputs before giving to Thymio
    def check_limits(self, left, right):
        #left wheel
        left =  min(left, self.max_speed)    #larger then 500
        left =  max(left,-self.max_speed)    #smaller than -500
        left = int(left)
        #right wheel  
        right = min(right, self.max_speed)   #larger than -500
        right = max(right,-self.max_speed)  #smaller than -500 
        right = int(right)

        return left, right

    def check_max_cms(self, left, right):
        #left wheel
        left =  min(left, self.max_speed_cms)    #larger then 20
        left =  max(left,-self.max_speed_cms)    #smaller than -20
        #right wheel  
        right = min(right, self.max_speed_cms)   #larger than 20
        right = max(right,-self.max_speed_cms)   #smaller than -20

        return left, right

    def spd4thym(self,left,right):
        left = left % 2**16 #abs(left) if left > 0 else 2**16-abs(left)
        right = right % 2**16 #abs(right) if right < 0 else 2**16-abs(right)

        return left,right

    def ikine(self):
        #finding polar coordinates for Astolfi controller
        #dx, dy, theta
        delta = self.check_pos()
        theta = self.curr_pos[2]
        ref_angle = math.atan2(self.next[1],self.next[0])
        #polar coordinates alpha, beta, rho
        self.rho = math.sqrt(delta[0]**2 + delta[1]**2)
        self.alpha = math.atan2(delta[1],delta[0])-theta 
        #self.alpha = self.normalize_ang(self.alpha)
        self.beta  = - theta - self.alpha#+ ref_angle
        #self.beta = self.normalize_ang(self.beta)
        #velocities v and w
        vel = self.kr*self.rho
        omega = (self.ka*self.alpha + self.kb*self.beta)

        # #finding corresponding wheel speeds
        phi_L = vel/self.wheel_radius + omega*self.wheel_length/self.wheel_radius
        phi_R = (vel/self.wheel_radius - omega*self.wheel_length/self.wheel_radius)  

        phi_L, phi_R = self.check_max_cms(phi_L, phi_R)
        # self.check_limits(phi_L,phi_R)
        self.set_speed(phi_L, phi_R)

    #checking if we reached the next goal
    def check_pos(self):
        delta = self.compute_dist(self.curr_pos,self.next)          
        return delta #if any(delta)>0.1 else np.array([0,0])
            
    #computing delta
    def compute_dist(self,curr,next):
        return np.array([next[0]-curr[0],next[1]-curr[1]])
    
    def normalize_ang(self, angle):
        return ((angle+math.pi)%2*math.pi)-math.pi

    # def astolfi(self,)
    
    #forward kinematics for debugging
    def fkine(self):
        theta=self.curr_pos[2]
        R = np.array(((math.cos(theta),  math.sin(theta), 0),
                      (-math.sin(theta),  math.cos(theta),  0),
                      (      0,                  0,        1)))

        
        
        wheel = np.array(((self.phiL+self.phiR)*self.wheel_radius/2
                            ,0
                        ,(self.phiL-self.phiR)*self.wheel_radius/2/self.wheel_length))
        #temp_forward.transpose()

        forward = np.dot(R,wheel)
        
        temp = self.getCurrPos()
        vel=self.wheel_radius/2*(self.phiL+self.phiR)
        w=self.wheel_radius/2*(self.phiL-self.phiR)/self.wheel_length
        x_dot=vel*np.cos(theta)
        y_dot=-vel*np.sin(theta)
        theta_dot=w

        x = temp[0] + x_dot*self.Ts
        y = temp[1] + y_dot*self.Ts
        q = temp[2] + theta_dot*self.Ts
        self.setCurrPos((x,y,q))

    def run_on_thymio(self,th):
        ''' give values to the motors of the thymio'''
        left  = int(self.phiL * self.cm2thym)
        right = int(self.phiR * self.cm2thym)

        left,right = self.check_limits(left,right)
        left,right = self.spd4thym(left,right)

        th.set_var("motor.left.target", left)
        th.set_var("motor.right.target", right)

    def ref_angle(self):
        theta=[]
        i=0
        path=self.set_globalpath()
        while i<len(path)-2:
            theta.append(math.atan2(path[i+1,1]-path[i,1],
                                    path[i+1,0]-path[i,0]))
        return theta

    def track_nodes(self):
        if self.rho < 3 and self.node_index<=(len(self.path)-2):
            self.node_index+=1

    def set_ref_angle(self):        
        if self.node_index > 0:
            delta = self.compute_dist(self.traj[self.node_index-1],self.traj[self.node_index])
            self.ref_angle=math.atan2(delta[1],delta[0])

    def polar_pos(self):
        delta = self.compute_dist(self.curr_pos[0:1],self.traj[self.node_index])
        theta=self.curr_pos[2]
        #polar coordinates alpha, beta, rho
        self.rho = math.sqrt(delta[0]**2 + delta[1]**2)
        self.alpha = math.atan2(delta[1],delta[0])-theta 
        self.alpha = self.normalize_ang(self.alpha)
        self.beta  = math.pi/2 - theta - self.ref_angle
        self.beta = self.normalize_ang(self.beta)

    def polar_vel(self):
        rho_dot = self.kr
        
        return self.rho
    def simpler_controller(self):
        return

    def go_straight(self):
        self.set_speed(100,100)

    def turn_cw(self):
        pos=self.curr_pos
        theta=pos[2]
        node1=self.path[self.node_index+1]
        node2=self.path[self.node_index]
        delta=[node2[0]-node1[0],node2[1]-node1[1]]
        theta_desired=math.atan2(delta[1],delta[0])
        if abs(theta-theta_desired) < 0.5:
            self.set_speed(0,0)
            if self.node_index<len(self.path)-2:
                self.node_index+=1
        else:
            self.set_speed(50,2**16-50)



         
        