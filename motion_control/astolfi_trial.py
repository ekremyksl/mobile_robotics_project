import numpy as np
import math as m
import matplotlib.pyplot as plt
import numpy as np
import math as m
import matplotlib.pyplot as plt
import utilities as ut
from Thymio import Thymio
import os
import time
import serial
from AstolfiController import Astolfi
from SimpleController import SimpleController


trajectory = [(0.0,0.0),(3.0,0.0),(5.0,4.0),(6.0,7.0)]
# trajectory = [(0.0,0.0),(5.0,0.0),(10.0,5.0),(10.0,0.0),(0.0,0.0)]
# trajectory = [(0.0,0.0),(3.0,0.0),(3.0,3.0),(0.0,3.0),(0.0,0.0)]
# trajectory = [(0.0,0.0), (1.0,-1.0), (2.0,-1.5), (2.5, 1.5), (1.0,3.0),(0.0,0.0)]

curr=np.array((0.0,0.0,0.0))
 
use_class=True   
if use_class==False:
    r=4.3/2
    L=11.3/2

    j=1
    global_path = []
    global_path.append([trajectory[0][0], trajectory[0][1], 0])
    print(global_path)
    while j<len(trajectory):
        pos1=trajectory[j]
        pos2=trajectory[j-1]
        temp=[pos1[0]-pos2[0],pos1[1]-pos2[1]]
        ang=m.atan2(temp[1],temp[0])
        # print([pos2[0], pos2[1], ang])
        global_path.append([pos1[0], pos1[1], ang])
        j+=1

    global_path=np.array(global_path)
    print(global_path)
    ref=global_path
    next_index=1
    on_goal=False
    next=global_path[1]
    Ts=0.01
    print(next)
    theta=[curr[2]]
    pos=[]
    pba=[]
    pos.append(curr)
    i=0
    j=1
    Kr=1
    Ka=20
    Kb=-1.5
    phi_dot=[]
    while on_goal==False:

        delta=next-curr
        norm=m.sqrt(delta[0]**2+delta[1]**2)
        if norm < 0.3 and next_index<len(global_path)-1:
            print('proceeding next node',next_index,i)
            next_index+=1
            next=global_path[next_index]
            delta=next-curr
            print('new node is:', next)        
        elif next_index >= len(global_path)-1 and norm < 0.3:
            on_goal=True

        # print('current node is:',curr)

        rho = m.sqrt(delta[0]**2+delta[1]**2)
        alpha = -curr[2] + m.atan2(delta[1],delta[0])
        alpha = m.atan2(m.sin(alpha),m.cos(alpha))
        beta = -alpha - curr[2] - next[2]
        beta = m.atan2(m.sin(beta),m.cos(beta))

        pba.append([rho, alpha, beta])

        vel = Kr*rho
        omega = Ka*alpha+Kb*beta

        phiL=(vel + omega*L)/r
        phiR=(vel - omega*L)/r

        phi_dot.append([phiL, phiR])

        q_dot=omega
        curr[2]+=q_dot*Ts
        x_dot = vel*m.cos(curr[2])
        y_dot = vel*m.sin(curr[2])

        curr[0]+=x_dot*Ts
        curr[1]+=y_dot*Ts

        pos.append([curr[0], curr[1], curr[2]])

        i+=1
    plt.subplots()
    plt.plot(pos[:,0],pos[:,1],'b')
    plt.plot(global_path[:,0],global_path[:,1],'r*')
    plt.plot(global_path[:,0],global_path[:,1],'r--')

    plt.subplots()
    plt.plot(pba[:,0],'g',label='rho')
    plt.subplots()
    plt.plot(pba[:,1],'b',label='alpha')
    plt.subplots()
    plt.plot(pba[:,2],'r',label='beta')
    plt.subplots()

    plt.subplots()
    plt.plot(phi_dot[:,0],'g')
    plt.plot(phi_dot[:,1],'b')
    plt.show()

else:

    # trajectory = [(0.0,0.0),(3.0,0.0),(5.0,4.0),(6.0,7.0)]
    # trajectory = [(0.0,0.0),(5.0,0.0),(10.0,5.0),(10.0,0.0),(0.0,0.0)]
    trajectory = [(0.0,0.0),(5.0,0.0),(5.0,5.0),(0.0,5.0),(0.0,0.0)]
    # trajectory = [(0.0,0.0), (1.0,-1.0), (2.0,-1.5), (2.5, 1.5), (1.0,3.0),(0.0,0.0)]
    # trajectory = [(0,0), (5,1), (10,1) ]

    curr=np.array((0.0,0.0,0.0))

    astol = Astolfi()
    astol.set_curr(curr)
    astol.set_path(trajectory)
    simp = SimpleController()

    simp.set_curr(curr)
    simp.set_global(trajectory)
    pba=[]
    pos=[]
    phi_dot=[]
    i=0
    # simp.set_thymio(th)
    simp.correct_heading()

    while astol.on_goal==False:  

        astol.check_nodes(verbose=True)
        rho,alpha,beta = astol.polar_rep()
        if astol.rho < 1:
            pos.insert(i,[simp.curr[0],simp.curr[1],simp.curr[2]])
            simp.follow_line()
            simp.check_node()
            simp.correct_heading()
            astol.set_curr(simp.get_curr())
            astol.set_goal(simp.get_goal())
        
        elif astol.rho >= 1:
            pos.insert(i,[astol.curr[0],astol.curr[1],astol.curr[2]])
            
            
            phiL,phiR = astol.compute_phi_dot(fkine=True)
            phi_dot.append([phiL, -phiR])
            # astol.run_on_thymio()
            simp.set_curr(astol.get_curr())  
            simp.set_goal(astol.get_goal())
            pba.append([rho,alpha,beta])      

        
        print(astol.curr)    
        time.sleep(astol.Ts)
        i+=1
    # th.set_var("motor.left.target", 0)
    # th.set_var("motor.right.target", 0)
    pos.insert(i,[astol.curr[0],astol.curr[1],astol.curr[2]])
    temp=pos
    pos=np.array(pos)
    global_path = np.array(astol.path)
    pba = np.array(pba)
    phi_dot=np.array(phi_dot)

    plt.subplots()
    plt.plot(pos[:,0],pos[:,1],'b')
    plt.plot(global_path[:,0],global_path[:,1],'r*')
    plt.plot(global_path[:,0],global_path[:,1],'r--')
    plt.subplots()
    plt.plot(pba[:,0],'g',label='rho')
    plt.subplots()
    plt.plot(pba[:,1],'b',label='alpha')
    plt.subplots()
    plt.plot(pba[:,2],'r',label='beta')

    plt.subplots()
    plt.plot(phi_dot[:,0],'g')
    plt.plot(phi_dot[:,1],'b')
    plt.show()