import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import math
import cv2 as cv

class PlotMap():
    def __init__(self, period, position_list, cov_list, img, path):
        self.period = period # period of data of position_list
        self.position_list = position_list*10 # position at each time step: list of [x position, y position]
        self.cov_list =cov_list*100 # list of covariance matrices between x and y position
        self.img = cv.flip(img, 0)
        self.path=path.copy()*10

        # create plot
        self._plot()

    def _plot(self):
        fig, ax = plt.subplots(nrows=2, figsize=(16,12))
        path=self.path
        
        ax[0].imshow(self.img)
        ax[0].axis("equal")
        uncertainty = [] # list that contains the largest uncertainty of each time step
        for i, position in enumerate(self.position_list):
            if i%10 != 0 and i!= len(self.position_list)-1:
                continue
            # calc. eigenvalues and eigenvectors to determine the absolute value and the direction of the variance of the uncertainty
            eigenvalues, eigenvectors = np.linalg.eig(self.cov_list[i,0:2,0:2])
            stds2 = 2*np.sqrt(np.absolute(eigenvalues))
            uncertainty.append(np.amax(stds2))

            # plot ellipse representing the second standard deviation
            ell = Ellipse(xy=(position[0], position[1]),width=stds2[0], height=stds2[1], \
                                angle=np.rad2deg(np.arctan2(eigenvectors[1, 0],eigenvectors[0,0])), edgecolor="r", label='uncertainty')
            ell.set_facecolor('none')
            ax[0].add_patch(ell)

            # plot center point of Thymio
            ax[0].scatter(position[0], position[1], color="g", label='estimated positions') 

            ax[0].text(position[0], position[1], str(i), fontsize=12)     
        ax[0].set_title("Position of Thymio")
        ax[0].set_xlabel("x axis [mm]")
        ax[0].set_ylabel("y axis [mm]")
        
        ax[0].plot(path[:,0], path[:,1],'r*-',label='optimal trajectory')
        ax[0].plot(path[0,0], path[0,1],'rv',label='start')
        ax[0].plot(path[-1,0], path[-1,1],'r^',label='goal')
        ax[0].plot(path[:,0], path[:,1],'r--')
        # ax[0].legend()
  

        # plot uncertainty in fct. of time
        ax[1].plot(uncertainty)
        ax[1].axis("equal")
        ax[1].set_title("Positional uncertainty (largest second standard deviation)")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("uncertainty [mm]")

        plt.show()