from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import sys
import os 

import wavekit_py as wkpy

class HASO_SH(WavefrontSensor):

    def __init__(self, conf) -> None:
        super().__init__(conf)

        #confHASO = self.extract_info_from_configfile(conf["confFile"])
        #Init Camera
        try :
            self.camera = wkpy.Camera(config_file_path = conf["confFile"])
            self.camera.connect()
            self.camera.start(0, 1)
        except Exception as e :
            print(str(e))

        
        time.sleep(0.1)

        if "exposure" in conf:
            self.setExposure(conf["exposure"])       


        self.img = wkpy.Image(size = wkpy.uint2D(1,1), bit_depth = 16)

    def extract_info_from_configfile(self, haso_config_file_path):
        try :
            hasoconfig, hasospec, wavelenght  = wkpy.HasoConfig.get_config(haso_config_file_path);
        except Exception as e :
            print(str(e))
            
        return (hasospec.nb_subapertures.X, hasospec.nb_subapertures.Y), (hasospec.ulens_step.X, hasospec.ulens_step.Y), hasospec.micro_lens_focal


    def setExposure(self, exposure):
        super().setExposure(exposure)
        self.camera.set_parameter_value("exposure_duration_us", self.exposure)
        return



    def expose(self):
        
        
        self.img = self.camera.snap_raw_image()
        
        self.data = np.ndarray((self.img.get_size()[0].Y,self.img.get_size()[0].X), 
                               buffer= self.img.get_data(), 
                               dtype=np.uint32)

        super().expose()

        return

    def __del__(self):
        super().__del__()
        time.sleep(1e-1)
        if self.camera is not None:
            self.camera.stop()
            self.camera.disconnect()
        
        return