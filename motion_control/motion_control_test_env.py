import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import Robothym 


import serial
import time

thym=Robothym.Robothym()
thym.setCurrPos((0.0, 0.0,0))
#glob=[(2,2),(4,4),(5,5),(6,6)]
glob=[(5,0),(5,5),(0,5),(0,0)]
thym.set_globalpath(glob)
x=[]
x.append(thym.getCurrPos())
i=0
node_index=0
# Ts=0.01
# thym.set_steptime(Ts)
next=glob[0]
thym.set_goal(next)
on_goal=False
speedy=[]
pba=[]
speeds=[]
print(next)
while on_goal==False:
    if thym.rho:
        if node_index < len(glob) and thym.rho < thym.on_node:   
            node_index+=1
            print(i)          
            if node_index < len(glob):
                next=glob[node_index] 
            thym.set_goal(next)            
            print(next)
        elif node_index >= len(glob) and thym.rho < thym.on_node:
            print(next)
            on_goal=True
            
    thym.ikine()
   # print(thym.phiL,thym.phiR)
    left  = int(thym.phiL * thym.cm2thym)
    right = int(thym.phiR * thym.cm2thym)        
    left,right = thym.check_limits(left,right)    
    left,right = thym.spd4thym(left,right)

    pba.append([thym.rho, thym.alpha, thym.beta])
    speeds.append([left,right])
    # thym.run_on_thymio(th)
    
    thym.fkine()
    temp=thym.getCurrPos()
    x.append(temp)
    speedy.append([thym.phiL,thym.phiR])
    i+=1
    
print('all done, plotting...')
speedy=np.array(speedy)
x=np.array(x)
path=np.array(glob)
speeds=np.array(speeds)
pba=np.array(pba)
j=0
figg3, (ax3) = plt.subplots()
plt.plot(-speedy[:,1],'b',label='right')
plt.plot(speedy[:,0],'r',label='left')
ax3.set_xlabel('sampling time')
ax3.set_ylabel('wheel speed')
figg3.legend(bbox_to_anchor=(0, 0.8))
figg3, (ax3) = plt.subplots()
plt.plot(speeds[:,1],'b',label='right')
plt.plot(speeds[:,0],'r',label='left')
ax3.set_xlabel('sampling time')
ax3.set_ylabel('wheel speed')
# # while j<len(zibido):
# #     if zibido[j,1]>500:
# #         zibido[j,1]-=2**16
# #     if zibido[j,0]>500:
# #         zibido[j,0]-=2**16
# #     j+=1    
figg2, (ax2) = plt.subplots()
plt.plot(pba[:,0],'g',label='left')
plt.subplots()
plt.plot(pba[:,1],'b',label='right')

plt.subplots()
plt.plot(pba[:,2],'r',label='left')
# ax2.set_xlabel('sampling time')
# ax2.set_ylabel('wheel speed')
# figg2.legend(bbox_to_anchor=(0, 0.8))
# figg4, (ax4) = plt.subplots()
# plt.plot(path[:,1],'r',label="nodes")
# figg5, (ax5) = plt.subplots()
# plt.plot(path[:,0],path[:,1],'r--',label="nodes")

figg1, (ax1) = plt.subplots()
ax1.plot(path[:,0],path[:,1],'r*',label="nodes")
ax1.plot(path[:,0],path[:,1],'r--',label="nodes")
ax1.plot(x[:,0],x[:,1],'b',label="traj")
ax1.set_xlabel('sampling time')
ax1.set_ylabel('trajectory')
figg1.legend(bbox_to_anchor=(0, 0.8))
plt.show()