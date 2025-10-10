#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *
from pyRTC.TimeResolvedLoopWithRemoteWFC import *



#%%

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

N = np.random.randint(3000,6000)
#%%

# %% Launch WFS
trwfs = hardwareLauncher("./hardware/MPDSPADCam.py", config, 5640)
trwfs.launch()

#%%
trwfs.run("advancedMode", True)
trwfs.run("setExposure", 4266) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
trwfs.run("setNIntegFrames", 1) # maybe increase this?
trwfs.run("applySettingsToCamera") # Appy the settings to the camera previously put


#%%
trwfs.run("setNFrames", 48)
trwfs.run("enable_sync_mode")
trwfs.run("applySettingsToCamera")
#%%
trwfs.run("advancedMode", True)
trwfs.setExposure(4166) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
trwfs.setNIntegFrames(1) # maybe increase this?
trwfs.applySettingsToCamera() # Appy the settings to the camera previously put


#%%
################## Setup MPD SPAD Camera ##############
confTRWFS = conf["trwfs"]
trwfs = MPDSPADCam(conf=confTRWFS)
#trwfs.stop()

#%%
trwfs.advancedMode(True)
trwfs.setExposure(4166) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
trwfs.setNIntegFrames(1) # maybe increase this?
trwfs.applySettingsToCamera() # Appy the settings to the camera previously put


#%%
trwfs.setNFrames(48)
trwfs.enable_sync_mode()
trwfs.applySettingsToCamera()
#%%
trwfs.start()

#%%
trwfs.takeDark()
trwfs.saveDark("res/sky_latest_dark.npy")

#%%
################## Setup Full Frame Signal ##############

# For TR PWFS
sig = TimeResolvedFullFrameProcessCustomArea(conf=conf)
sig.start()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/TimeResolvedFullFrameProcessCustomArea.py", config, 5641)
slopes.launch()
#plt.imshow(np.sum(sig.read(), axis=0))


#%%
################## Connect to remote WFC ##############

# Dont forget to start the DM on the other PC
confWFC = conf["wfc"]
remote_wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", confWFC, 3000, remoteProcess=True)
remote_wfc.host = "132.246.192.209"
remote_wfc.launch()
#%%
remote_wfc.run("flatten")

#%%
################## Setup loop ##############

loop = TimeResolvedLoopWithRemoteWFC(conf, remote_wfc, settings_name="trloop")

#%%
loop.manual_cam_obj = trwfs
loop.left_limit = 19
loop.right_limit = 53
#%%

loop.computeIM()
loop.flatten()

#%%
for i in range(10):
    loop.timeResolvedIntegratorWithLeak_C()
    #print(np.max(loop.currentCorrection))
#%%
loop.resetCurrentCorrection()
loop.flatten()
#%%
trwfs.record_data_direct(1000, "SPAD_TR_CL_0g3_0L02_11h08_25sept2025")

#%%
loop.setGain(0.00001)
loop.leakyGain = 0.00

#%%
loop.setGain(0.1)
loop.leakyGain = 0.01
#%%
loop.resetCurrentCorrection()
loop.start()

#%%
loop.stop()

#%%

valid_aps = np.load("valid_aps_spad_25sept2025.npy")

#%%
loop.changeWeightsAndUpdate(np.ones_like(loop.frame_weights))

#%%
loop.switchToFF()

# %%
loop.switchToFFwithWeights()

#%%
newCorrection = np.ascontiguousarray((1-loop.leakyGain)*np.array(loop.remoteWFC.getProperty("currentCorrection")), dtype=np.float32)
start_time = time.time()
for i in range(1000):
    loop.TR_norm_correction_function(loop.numModes, loop.numFrames, loop.signalSize,
                                                 loop.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.latest_slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.ref_signal_per_mode_normed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 newCorrection.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
stop_time = time.time()

print(f"{1/((stop_time-start_time)/1000)} Hz")

#%%
loop.flatten()

#%%


def expose_test():
    if trwfs.cam.IsTriggered() or (not trwfs.inSyncMode) :
        trwfs.cam.SnapPrepare()
        trwfs.cam.SnapAcquire()
        trwfs.data = trwfs.cam.SnapGetImageBuffer()[0]  # frames of counter 1 

def expose_test2():
    if trwfs.cam.IsTriggered() or (not trwfs.inSyncMode) :
        t5 = trwfs.cam.ContAcqToMemoryGetBuffer()
        f5 = trwfs.cam.BufferToFrames(t5, trwfs.cam.num_pixels, trwfs.cam.num_counters)[0]

#%%
start_time = time.perf_counter()

for i in range(1000):
    loop.remoteWFC.run("write", loop.convertForTransmission(loop.currentCorrection))

stop_time = time.perf_counter()

print(f"{1/((stop_time-start_time)/1000)} Hz")

#%%

def expose_test():


    # TODO do we want to use other counters?
    data = np.empty((0,), dtype=np.uint8)
    while data.size < (trwfs.nFrames+trwfs._offset)*32*64:
        data = np.concatenate((data, trwfs.cam.ContAcqToMemoryGetBuffer()))
    #self.data = self.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
    #self.frames = self.cam.ContAcqToMemoryGetBuffer()
    #if self.frames.shape[0] != self.nFrames: # Sometimes the snap returns nothing, TODO check to use a flag check maybe?
    #    return
    data = data[0: trwfs.cam.num_counters * trwfs.cam.num_pixels * int(np.floor((data.size / (trwfs.cam.num_counters * trwfs.cam.num_pixels))))]

    all_frames = (trwfs.cam.BufferToFrames(data, trwfs.cam.num_pixels, trwfs.cam.num_counters)[0])
    trwfs.frames = all_frames[trwfs._offset:trwfs._offset+48,:,:]
    trwfs.offset = trwfs.nFrames - ((all_frames.shape[0]-trwfs._offset) % trwfs.nFrames)
    
    #self.data = (self.cam.BufferToFrames(buf, self.cam.num_pixels, self.cam.num_counters)[0])[:48,:,:]
    
    # for idx in trwfs.screamers:
    #     trwfs.frames[:,idx[0], idx[1]] = 0





#%%