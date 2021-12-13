import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import math

class PlotMap():
    def __init__(self, period, position_list, cov_list, path):
        self.period = period # period of data of position_list
        self.position_list = position_list # position at each time step: list of [x position, y position]
        self.cov_list =cov_list # list of covariance matrices between x and y position
        self.path = path

        # create plot
        self._plot()

    def _plot(self):
        fig, ax = plt.subplots(ncols=2, figsize=(10,6))
        

        uncertainty = [] # list that contains the largest uncertainty of each time step
        for i, position in enumerate(self.position_list):
            # calc. eigenvalues and eigenvectors to determine the absolute value and the direction of the variance of the uncertainty
            eigenvalues, eigenvectors = np.linalg.eig(self.cov_list[i,0:2,0:2])
            stds2 = 2*np.sqrt(np.absolute(eigenvalues))
            uncertainty.append(np.amax(stds2))

            # plot ellipse representing the second standard deviation
            ell = Ellipse(xy=(position[0], position[1]),width=stds2[0], height=stds2[1], \
                                angle=np.rad2deg(np.arctan2(eigenvectors[1, 0],eigenvectors[0,0])), edgecolor="r")
            ell.set_facecolor('none')
            ax[0].add_patch(ell)

            # plot center point of Thymio
            ax[0].scatter(position[0], position[1], color="b")

            # plot time step beside the points
            ax[0].text(position[0], position[1], str(i), fontsize=12)     
        ax[0].set_title("Position of Thymio")
        ax[0].set_xlabel("x axis [cm]")
        ax[0].set_ylabel("y axis [cm]")
        ax[0].set_xlim([-2, 20])
        ax[0].set_ylim([-2, 20])
        ax[0].plot(self.path[:,0], self.path[:,1],'r*-',label='optimal trajectory')
        ax[0].plot(self.path[0,0], self.path[0,1],'rv',label='start')
        ax[0].plot(self.path[-1,0], self.path[-1,1],'r^',label='goal')
        ax[0].plot(self.path[:,0], self.path[:,1],'r--')

        # create x axis for uncertainty plot
        time = [i for i in range(len(uncertainty))]
        time = np.array(time, dtype=float)*self.period

        # plot uncertainty in fct. of time
        ax[1].plot(time, uncertainty)
        ax[1].axis("equal")
        ax[1].set_title("Positional uncertainty")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("max. second std [cm]")
        ax[1].set_ylim([-1, 4])
        ax[1].set_xlim([0, len(time)*self.period])

        plt.show()