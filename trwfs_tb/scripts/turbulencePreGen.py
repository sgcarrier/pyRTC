
import numpy as np

class DummyAtm():

    def __init__(self, prerecorded_atm) -> None:

        self.atm = prerecorded_atm
        self.maxPos = self.atm.shape[0]
        self.mask = self.genMask(11)
        self.currentPos = None
        self.speed = 1


    def setSpeed(self, newSpeed):
        self.speed = newSpeed
    
    def getNextTurbAsModes(self):
        if self.currentPos is None:
            self.currentPos = 0
        elif self.currentPos >= self.maxPos:
            self.currentPos = 0
        else:
            self.currentPos += int(1 * self.speed)
        
        return self.atm[self.currentPos,:]
    

    # def rebin(self, arr, new_shape):
    #     shape = (new_shape[0], arr.shape[0] // new_shape[0],
    #              new_shape[1], arr.shape[1] // new_shape[1])        
    #     out = (arr.reshape(shape).mean(-1).mean(1)) * (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])        
    #     return out

    def genMask(self, res):

        xx,yy = np.meshgrid(np.arange(res),np.arange(res))
        zz = np.sqrt((xx-(res//2))**2 + (yy-(res//2))**2)
        actMap = np.zeros((res,res)).astype('bool')
        actMap[zz<=(res/2)] = True
        return actMap
    
    # def setC2MFromM2C(self, M2C):
    #     self.C2M = np.linalg.pinv(M2C)