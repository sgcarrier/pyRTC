from OOPAO.Atmosphere import Atmosphere
#from OOPAO.DeformableMirror import DeformableMirror
#from OOPAO.MisRegistration import MisRegistration
#from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
#from OOPAO.calibration.ao_calibration import ao_calibration
#from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
#from OOPAO.tools.displayTools import displayMap

from res.parameterFile_atm import initializeParameterFile
from OOPAO.tools.displayTools           import displayMap
import numpy as np

class OOPAO_atm():

    def __init__(self) -> None:

        self.param = initializeParameterFile()


        #Create our Telescope Simulatation
        self.tel = Telescope(   resolution          =  352,
                                diameter            =  self.param['diameter'],
                                samplingTime        =  self.param['samplingTime'],
                                centralObstruction  =  self.param['centralObstruction'])

        #Create a guide star
        self.ngs = Source(optBand =  self.param['opticalBand'], 
                          magnitude =  self.param['magnitude'])
        self.ngs* self.tel

        self.atm = Atmosphere(  telescope     = self.tel,
                                r0            =  self.param['r0'],
                                L0            =  self.param['L0'],
                                windSpeed     =  self.param['windSpeed'],
                                fractionalR0  =  self.param['fractionnalR0'],
                                windDirection =  self.param['windDirection'],
                                altitude      =  self.param['altitude'])

        
        self.atm.initializeAtmosphere(telescope=self.tel)

        self.tel+self.atm
        self.M2C=np.load("C:\\Users\\mcaousr\\Desktop\\M2C_synthetic.npy")# go from um to m
 
        self.setC2MFromM2C(self.M2C)
 

        self.mask = self.tel.pupil.copy().astype('int')

    def getNextAtmOPD(self):
        self.atm.update()
        return self.atm.OPD
    
    def getNextTurbAsModes(self):
        atm_phase = self.getNextAtmOPD()
        modes_to_send = self.C2M @  atm_phase[self.mask==1]
        return modes_to_send
    

    def rebin(self, arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])        
        out = (arr.reshape(shape).mean(-1).mean(1)) * (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])        
        return out

    def genMask(self, res):

        xx,yy = np.meshgrid(np.arange(res),np.arange(res))
        zz = np.sqrt((xx-(res//2))**2 + (yy-(res//2))**2)
        actMap = np.zeros((res,res)).astype('bool')
        actMap[zz<=(res/2)] = True
        return actMap
    
    def setC2MFromM2C(self, M2C):
        self.C2M =np.load("C:\\Users\\mcaousr\\Desktop\\C2M_synthetic.npy")
    
    def getdmplot(self):   
        import matplotlib.pyplot as plt     
        atm_phase = self.getNextAtmOPD()
        v=self.C2M @ atm_phase[self.mask==1]
        dmMap=self.genMask(11)
        temp=np.zeros((11,11))
        temp[dmMap==1]=v
        return temp