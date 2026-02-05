from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *
from pyRTC.TimeResolvedLoopWithRemoteWFC import *




config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

confTRWFS = conf["trwfs"]
trwfs = MPDSPADCam(conf=confTRWFS)
#trwfs.stop()
trwfs.advancedMode(True)
#trwfs.setExposure(4166) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
#trwfs.setNIntegFrames(3) # maybe increase this?
trwfs.setExposure(690) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
trwfs.setNIntegFrames(3) # maybe increase this?
trwfs.setNFrames(48)
trwfs.applySettingsToCamera() # Appy the settings to the camera previously put



def expose_test():
    trwfs.cam.SnapPrepare()
    trwfs.cam.SnapAcquire()
    data = trwfs.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
    return data.shape[0]

def expose_test2():
    #if trwfs.cam.IsTriggered() or (not trwfs.inSyncMode) :
    t5 = trwfs.cam.ContAcqToMemoryGetBuffer()
    f5 = trwfs.cam.BufferToFrames(t5, trwfs.cam.num_pixels, trwfs.cam.num_counters)[0]
        
    frames = f5.shape[0]
    return frames



def run_test():
    total_frames = 0
    start_time = time.perf_counter()
    trwfs.cam.ContAcqToMemoryStart()
    for i in range(1000):
        total_frames += expose_test2()
    stop_time = time.perf_counter()
    trwfs.cam.ContAcqToMemoryStop()
    print(f"{1/((stop_time-start_time)/1000)} Hz")
    print(f"{total_frames/((stop_time-start_time))} fps")



run_test()


print("Done")