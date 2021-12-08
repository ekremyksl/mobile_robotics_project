import numpy as np
import math
import time
import matplotlib.pyplot as plt
import serial
from SimpleController import SimpleController




r=4.3/2
L=11.3/2

def normalize_ang(angle):
    return math.atan2(math.sin(angle),math.cos(angle))

trajectory = [(0.0,0.0),(5.0,0.0),(5.0,5.0),(0.0,5.0),(0.0,0.0)]
trajectory = [(0.0,0.0),(5.0,4.0),(12.0,12.5),(13.0,13.0),(26.0,20.0)]
curr=np.array((0.0,0.0,0.0))
j=1
global_path = []
global_path.append([trajectory[0][0], trajectory[0][1], 0])
print(global_path)
while j<len(trajectory):
    pos1=trajectory[j]
    pos2=trajectory[j-1]
    temp=[pos1[0]-pos2[0],pos1[1]-pos2[1]]
    ang=math.atan2(temp[1],temp[0])
    # print([pos2[0], pos2[1], ang])
    global_path.append([pos1[0], pos1[1], ang])
    j+=1


global_path=np.array(global_path)
print(global_path)
print('************')
print(np.rad2deg(0.1))

print(np.rad2deg(0.1))
next_index=1
on_goal=False
next=global_path[1]
Ts=0.001
print(next)
theta=[curr[2]]
pos=[]
pos.append(curr)
i=0
while on_goal == False:
    print('correcting heading')

    while abs(curr[2]-next[2])>0.03:
        
        phiL=100
        phiR=-phiL
        dq=r/(2*L)*(phiL-phiR)
        curr[2]+=dq*Ts 
        curr[2]=normalize_ang(curr[2])
        pos.append([curr[0],curr[1], curr[2]])
        i+=1
    
    delta=[next[0]-curr[0],next[1]-curr[1]]
    dist=math.sqrt(delta[0]**2+delta[1]**2)
    print('moving straight')
    while dist > 0.3:
        
        delta=[next[0]-curr[0],next[1]-curr[1]]
        dist=math.sqrt(delta[0]**2+delta[1]**2)
        phiL=100
        phiR=phiL
        vel=r*(phiL+phiR)/2
        dx=vel*math.cos(curr[2])
        dy=vel*math.sin(curr[2])
        curr[0]+=dx*Ts
        curr[1]+=dy*Ts
        pos.append([curr[0],curr[1], curr[2]])
        i+=1
    next_index+=1
    delta=[next[0]-curr[0],next[1]-curr[1]]
    dist=math.sqrt(delta[0]**2+delta[1]**2)
    if next_index<len(global_path):
        next=global_path[next_index]
    else:
        on_goal=True
    i+=1

print('it has finished')
pos=np.array(pos)
print(pos)
print(i)

figg1, (ax1) = plt.subplots()
plt.plot(pos[0,:],pos[1,:],'b')
plt.plot(global_path[0,:],global_path[1,:],'r*')
plt.plot(global_path[0,:],global_path[1,:],'r--')
    

# # node_index=0
# while node_index < len(trajectory)-1:
#     ang_ref = 

