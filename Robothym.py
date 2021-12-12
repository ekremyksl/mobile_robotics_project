import math
import numpy as np
import utilities as ut
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
        self.set_astolfiGains()
        #thymio default parameters and thresholds
        self.set_physParams()
        self.set_thresholds()

    #defining physical parameters of Thymio
    def set_physParams(self):
        self.wheel_radius = ut.THYMIO_PARAMS['WHEEL_RAD']          #wheel radius of thymio [cm]
        self.wheel_length = ut.THYMIO_PARAMS['WHEEL_LENGTH']       #distance of wheels from each other [cm] 
        self.max_speed    = ut.THYMIO_PARAMS['MAX_SPEED']          #maximum thymio wheel speed
        self.max_speed_cms= ut.THYMIO_PARAMS['MAX_SPEED_CM']       #maximum thymio wheel speed [cm/s]
        self.cm2thym      = ut.THYMIO_PARAMS['SPEED_CM2THYM']      #maximum thymio wheel speed


    #setting Astolfi Controller Gains
    def set_astolfiGains(self):
        self.kr = ut.ASTOLFI_PARAM['K_RHO']
        self.ka = ut.ASTOLFI_PARAM['K_ALPHA']
        self.kb = ut.ASTOLFI_PARAM['K_BETA']

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
        self.Ts = ut.THYMIO_PARAMS['SAMPLING_TIME']

    #givin global path, the trajectory to thymio
    def set_globalpath(self, globpath):
        self.traj = globpath
    
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
        left =  self.max_speed if left > self.max_speed else left #larger then 500
        left = -self.max_speed if left <-self.max_speed else left #smaller than -500
        #right wheel  
        right =  self.max_speed if right > self.max_speed else right #larger than -500
        right = -self.max_speed if right <-self.max_speed else right #smaller than -500 

    def check_max_cms(self,vel):
        speed = vel if vel <=  self.max_speed_cms else  self.max_speed_cms
        speed = vel if vel >= -self.max_speed_cms else -self.max_speed_cms

    def ikine(self):
        #finding polar coordinates for Astolfi controller
        #dx, dy, theta
        delta = self.check_pos()
        theta = self.curr_pos[2]%(2*math.pi)
        #polar coordinates alpha, beta, rho
        self.rho = math.sqrt(delta[0]**2 + delta[1]**2)
        self.alpha = (math.pi-theta + math.atan2(delta[1],delta[0]))%(2*math.pi)-math.pi
        self.beta = -theta - self.alpha
        #velocities v and w
        vel = self.kr*self.rho
        omega = (self.ka*self.alpha + self.kb*self.beta)
        vel = self.check_max_cms(vel)
        #conversion to cartesian speeds
        x_dot = vel*math.cos(theta)
        y_dot = vel*math.sin(theta)

        #finding corresponding wheel speeds
        phi_L = (x_dot*math.cos(theta) + y_dot*math.sin(theta) + self.wheel_length*omega)/self.wheel_radius
        phi_R = (x_dot*math.cos(theta) + y_dot*math.sin(theta) - self.wheel_length*omega)/self.wheel_radius

        self.check_limits()
        self.set_speed(phi_L, phi_R)

    #checking if we reached the next goal
    def check_pos(self):
        delta = self.compute_dist()          
        return delta if any(delta)>0.001 else np.array([0,0])
            
    #computing delta
    def compute_dist(self):
        return np.array([self.next[0]-self.curr_pos[0],self.next[1]-self.curr_pos[1]])
    
    #forward kinematics for debugging
    def fkine(self):
        theta=self.curr_pos[2]%(2*math.pi)

        R = np.array(((math.cos(theta), -math.sin(theta), 0),
                      (math.sin(theta),  math.cos(theta), 0),
                      (      0,                  0,       1)))

        coef=self.wheel_radius/2
        
        temp_forward = np.array(((self.phiL+self.phiR)*coef,0,(self.phiL-self.phiR)*coef/self.wheel_length))
        #temp_forward.transpose()

        forward = R.dot(temp_forward.transpose())
        
        temp = self.getCurrPos()

        x = temp[0]+forward[0]*self.Ts
        y = temp[1]+forward[1]*self.Ts
        q = temp[2]+forward[2]*self.Ts

        self.setCurrPos((x,y,q))

    def runOnThymio(self,th):
        ''' give values to the motors of the thymio'''
        left  = self.phiL * self.cm2thym
        right = self.phiR * self.cm2thym
        th.set_var("motor.left.target", left)
        th.set_var("motor.right.target", right)