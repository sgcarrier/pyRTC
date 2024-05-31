from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pylablib.devices import Andor
import argparse
import sys
import os 

class AndorIXon(WavefrontSensor):

    def __init__(self, conf) -> None:
        super().__init__(conf)

        # Open it with no temperature control, no gain and the fan on
        self.cam = Andor.AndorSDK2Camera(idx=conf["idx"], temperature="off", fan_mode="low", amp_mode=None)
        time.sleep(0.1)
        #Turnoff cooler
        if self.cam.is_cooler_on():
            print("WARNING: cooler is on, turning it off")
            self.cam.set_cooler(on=False)


        if "exposure" in conf:
            self.setExposure(conf["exposure"])

    def open_shutter(self):
        self.cam.setup_shutter("open")
    def close_shutter(self):
        self.cam.setup_shutter("closed")

    def close_camera(self):
        self.close_shutter()
        self.cam.close()
        


    def setExposure(self, exposure):
        super().setExposure(exposure)
        self.cam.set_exposure(self.exposure)
        return



    def expose(self):
        
        ## Unsure if necessary, I think the pylablib sdk for andor already manages this
        #while self.cam.acquisition_in_progress():
        #    time.sleep(0.1)

        img_raw = self.cam.snap()
        self.img = img_raw
        
        self.data = np.ndarray((self.img.shape[0],self.img.shape[1]), 
                               buffer= self.img, 
                               dtype=np.uint16)

        super().expose()

        return

    def __del__(self):
        super().__del__()
        time.sleep(1e-1)
        self.close_shutter()
        self.cam.close()
        
        return