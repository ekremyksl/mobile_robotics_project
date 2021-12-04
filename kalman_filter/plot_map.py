import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import math


# fig, ax = plt.subplots()
# sin_l, = ax.plot(np.sin(0))
# cos_l, = ax.plot(np.cos(0))
# ax.set_ylim(-1, 1)
# ax.set_xlim(0, 5)
# dx = 0.1

# def update(i):
#     # i is a counter for each frame.
#     # We'll increment x by dx each frame.
#     x = np.arange(0, i) * dx
#     sin_l.set_data(x, np.sin(x))
#     cos_l.set_data(x, np.cos(x))
#     return sin_l, cos_l

# ani = animation.FuncAnimation(fig, update, frames=51, interval=50)
# plt.show()


class PlotMap():
    def __init__(self, period, position_list, cov_list):
        self.period = period
        self.thymio_rel_coord = np.array([[-5.5,-5.5],[5.5,-5.5],[5.5,3],[4.7,3.8],[2.5,4.9],
                                          [0,5.5],[-2.4,5],[-4.4,3.9],[-5.5,3],[-5.5,-5.5]])
        self._plot(position_list, cov_list)

    def _plot(self, position_list, cov_list):

        fig, ax = plt.subplots(nrows=2)

        uncertainty = []
        for i, position in enumerate(position_list):
            eigenvalues, eigenvectors = np.linalg.eig(cov_list[i])
            stds = np.sqrt(np.absolute(eigenvalues))
            uncertainty.append(stds[0])

            ell = Ellipse(xy=(position[0], position[1]),width=stds[0]*2, height=stds[1]*2, \
                                angle=np.rad2deg(np.arctan2(eigenvectors[1, 0],eigenvectors[0,0])), edgecolor="r")
            ell.set_facecolor('none')
            ax[0].add_patch(ell)

            # # plot entire Thymio
            # thymio_coord = self._rotate(position[4],self.thymio_rel_coord)
            # ax[0].plot(position[0]+thymio_coord[:,0], position[1]+thymio_coord[:,1], color="b")

            # plot only center point of Thymio
            ax[0].scatter(position[0], position[1], color="b")
        
        ax[1].plot(uncertainty)
        ax[1].axis("equal")
        plt.show()

    def _rotate(self, angle, coords):
        """
        Rotates the coordinates of a matrix by the desired angle
        :param angle: angle in radians by which we want to rotate
        :return: numpy.array() that contains rotated coordinates
        """
        angle = angle - math.pi/2 # quater rotation in negative direction because thymio_rel_coord
                                  # assume that angle is at y-axis equal to zero
        R = np.array(((np.cos(angle), -np.sin(angle)),
                    (np.sin(angle),  np.cos(angle))))
        
        return R.dot(coords.transpose()).transpose()



if __name__ == '__main__':
    pos = [[0,0,0],
           [0,20,0],
           [0,40,math.pi]]
    PlotMap(pos)