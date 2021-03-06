import math
import numpy as np
import time
from tdmclient import ClientAsync, aw
from kalman_filter import KalmanFilter
from repeated_timer import RepeatedTimer
from plot_map import PlotMap

def measure(self):
        #await node.wait_for_variables() # wait for Thymio variables values
        self._data["time"].append(time.time()) # save timestamp
        self._data["ground_prox"].append(list(self._node["prox.ground.reflected"]))

def motors(left, right):
    left = int((500/16)*left)
    right = int((500/16)*right)
    return { "motor.left.target": [left], "motor.right.target": [right] }

if __name__ == '__main__':
    # set up connection to thymio, if node=0 no thymio is connected
    client = ClientAsync()
    node = aw(client.wait_for_node())
    aw(node.lock())

    # set up kalman filter
    angle = angle = np.deg2rad(0)
    vl = 2
    vr = 2
    vx = math.cos(angle)*(vl+vr)/2
    vy = math.sin(angle)*(vl+vr)/2
    va = (vr-vl)/9.4
    state_vector = np.array([[0,0,angle,vx,vy,va]], dtype=float)
    uncertainty_matrix = np.array([[[0.1,0,0,0,0,0],
                                    [0,0.1,0,0,0,0],
                                    [0,0,0.1,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]]], dtype=float)

    period = 0.1
    filter = KalmanFilter(node, period, state_vector[0], uncertainty_matrix[0])
    t1 = RepeatedTimer(period, filter.update_filter)

    t1.start()
    aw(node.set_variables(motors(vl,vr)))
    for i in range(16):
        time.sleep(1) 
        state_vector = np.append(state_vector, [filter.get_state_vector()], axis=0)
        uncertainty_matrix = np.append(uncertainty_matrix, [filter.get_covariance_matrix()], axis=0)

        # calc. second std: probability is more than 95% that robot is inside second std
        eigenvalues, _ = np.linalg.eig(uncertainty_matrix[i+1])
        stds2 = 2*np.sqrt(np.absolute(eigenvalues))

        if np.amax(stds2) > 3:
            aw(node.set_variables(motors(0,0)))
            t1.stop()
            print("second stds: {}".format(stds2))
            x_pos = float(input("enter x position: "))
            y_pos = float(input("enter y_position: "))
            angle = np.deg2rad(float(input("enter angle [in ??]: ")))
            filter.set_position_measurement([x_pos, y_pos, angle])
            t1.start()
            aw(node.set_variables(motors(vl,vr)))

        if i == 4:
            vl = 1
            vr = 3
            aw(node.set_variables(motors(vl,vr)))

        if i == 6:
            vl = 2
            vr = 2
            aw(node.set_variables(motors(vl,vr)))

        if i == 10:
            vl = 3
            vr= 1
            aw(node.set_variables(motors(vl,vr)))

        if i == 12:
            vl = 2
            vr = 2
            aw(node.set_variables(motors(vl,vr)))
        
    aw(node.set_variables(motors(0,0)))
    t1.stop()


    PlotMap(period, state_vector, uncertainty_matrix)

    