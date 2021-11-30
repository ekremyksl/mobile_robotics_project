import math
import numpy as np
import Robothym 
import matplotlib
import os
import sys
import math
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

thym=Robothym(0, 0, (1,15,-1.5), (0,0,0))
print(thym.getCurrPos())

print(0)