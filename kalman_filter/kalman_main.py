import math
import numpy as np
import time
from tdmclient import ClientAsync
from kalman_filter import KalmanFilter
from repeated_timer import RepeatedTimer
from plot_map import PlotMap

def measure(self):
        #await node.wait_for_variables() # wait for Thymio variables values
        self._data["time"].append(time.time()) # save timestamp
        self._data["ground_prox"].append(list(self._node["prox.ground.reflected"]))


def motors(left, right):
    return { "motor.left.target": [left], "motor.right.target": [right], }

if __name__ == '__main__':
    # set up connection to thymio, if node=0 no thymio is connected
    node = 0
    # client = ClientAsync()
    # node = await client.wait_for_node()
    # await node.lock()



    # set up kalman filter
    angle = angle = np.deg2rad(0)
    v = 5
    vx = math.cos(angle)*v
    vy = math.sin(angle)*v
    position = [0,0,vx,vy]
    position_cov = [[0.1,0,0,0],
                    [0,0.1,0,0],
                    [0,0,0,0],
                    [0,0,0,0]]
    pose = [angle,0]
    pose_cov = [[0,0],
                [0,0]]
    period = 0.1
    filter = KalmanFilter(node, period, position, position_cov, pose, pose_cov)

    # set up timer
    t1 = RepeatedTimer(period, filter.update_filter)

    t1.start()
    if node == 0:
        time.sleep(10)
    else:
        node.send_set_variables(motors(50, 50))
        # await client.sleep(12)
        node.send_set_variables(motors(0, 0))    
    t1.stop()

    PlotMap(period, filter.get_position_list(), filter.get_covariance_list(), filter.get_pose_list())
    