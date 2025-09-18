from pyRTC.TimeResolvedWavefrontSensor import TimeResolvedWavefrontSensor
from pyRTC.pyRTCComponent import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import sys
import os 
from pyRTC.hardware.Hermes.Hermes import *

# Class for the high-speed MPD SPAD camera
# Since we process cubes of images, we dont inherit WavefrontSensor
class MPDSPADCam(TimeResolvedWavefrontSensor):

    def __init__(self, conf) -> None:
        

        # Default settings for Hermes cam
        self.exposure = setFromConfig(conf, "exposure", 100)
        self.nFrames = setFromConfig(conf, "nFrames", 100)
        self.nIntegFrames = setFromConfig(conf, "nIntegFrames", 300)
        self.nCounters = setFromConfig(conf, "nCounters", 1)

        self.cam = Hermes(Hermes.CameraMode.NORMAL) # TODO add option to become trigger mode

        self.cam.SetCameraPar(Exposure = self.exposure,  # in 10ns
                              NFrames = self.nFrames, 
                              NIntegFrames = self.nIntegFrames , 
                              NCounters = self.nCounters , 
                              Force8bit = Hermes.State.DISABLED, 
                              Half_array = Hermes.State.DISABLED, 
                              Signed_data = Hermes.State.DISABLED)
        self.cam.ApplySettings()
        
        super().__init__(conf)
        return 



    def __del__(self):
        super().__del__()
        time.sleep(1e-1)
        del self.cam  # Hermes SDK managed the deallocation of memory
        return
    
    def enable_sync_mode(self):
        if (int(self.nFrames) > 100) or (int(self.nFrames) < 0):
            print("Change the nFrames parameter first, limit is 100 frames for sync mode")
        self.cam.SetSyncInState(Hermes.State.ENABLED, int(self.nFrames))

    def disable_sync_mode(self):
        self.cam.SetSyncInState(Hermes.State.DISABLED, 0)


    def expose(self):
        
        self.cam.SnapPrepare() # TODO I think this becomes blocking when in sync mode. Otherwise, might have to use IsTriggered()
        self.cam.SnapAcquire()

        # TODO do we want to use other counters?
        self.frames = self.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
        self.data = np.ndarray((self.frames.shape[0],self.frames.shape[1], self.frames.shape[2]), 
                            buffer= np.ascontiguousarray(self.frames), 
                            dtype=np.uint16)

    

        super().expose()
        #self.imageRaw.write(self.data)
        #Check float here
        #self.image.write(data_dark_substracted.astype(self.imageDType))

        return
    
    
