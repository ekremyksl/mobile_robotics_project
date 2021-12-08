import sys

import math
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import SimpleController

import os
import time 
import serial



simp = SimpleController.SimpleController()
print(simp.goalReached())

trajectory = [(0.0,0.0),(5.0,0.0),(5.0,5.0),(0.0,5.0),(0.1,0.1)]
curr=np.array((0.0,0.0,0.0))

simp.set_global(trajectory)
i=0
while not simp.goalReached():
    simp.check_node()
    simp.correct_heading()
    # simp.run_on_thymio(simp.th)  
    time.sleep(simp.Ts)
    # simp.run_on_thymio(simp.th)   
    simp.follow_line()
    time.sleep(simp.Ts)
    i+=1

pos=np.array(simp.pos)
global_path = np.array(simp.global_path)