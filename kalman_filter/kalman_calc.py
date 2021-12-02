import math
import numpy as np

class KalmanCalc():
    def update(self, x, z, F, Q, H, R, P):
        #print("update: z: {}".format(z))

        # make prediction step
        x, P = self._prediction(x, F, Q, P)
        #print("prediction:\t{}". format(x))

        # make correction step 
        x, P = self._correction(x, z, H, R, P)
        #print("correction:\t{}". format(x))
        return x, P
        
    def _prediction(self, x, F, Q, P):   
        # calc. state estimation and estimation uncertainty after pretiction step
        x = F@x
        P = F@(P@np.transpose(F)) + Q
        
        return x, P
    
    def _correction(self, x, z, H, R, P):
        I = np.identity(np.size(x))

        # measurement prediction covariance
        S = H@(P@np.transpose(H)) + R
        if np.linalg.det(S) == 0: # preventing singular matrix when calc. inverse
            S = S + 0.01*np.identity(np.size(R, axis=0))

        # calc. kalman gain
        K = P@(np.transpose(H)@np.linalg.inv(S))
               

        # calc inovation
        inovation = z - H@x

        # calc. state estimation and estimation uncertainty after correction step
        x = x + K@inovation
        #P = (I -K@H)@P
        P = (I - K@H)@(P@np.transpose(I - K@H)) + K@(R@np.transpose(K))

        # print("P: {}".format(P))
        # print("S: {}".format(S))
        # print("K: {}".format(K))
        # print("inovation: {}".format(inovation))
        # print("K@inov: {}".format(K@inovation))

        return x, P

