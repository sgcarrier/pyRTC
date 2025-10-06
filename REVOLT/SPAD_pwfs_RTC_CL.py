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
################## Setup Full Frame Signal ##############

# For TR PWFS
sig = TimeResolvedFullFrameProcessCustomArea(conf=conf)
sig.start()

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

loop.computeIM()
loop.flatten()

#%%
import timeit

for i in range(500):
    #loop.timeResolvedIntegratorWithLeak()
    saved.append(loop.currentCorrection)
#%%
loop.flatten()
#%%
trwfs.record_data_direct(1000, "SPAD_TR_CL_0g3_0L02_11h08_25sept2025")

#%%
loop.setGain(0.00001)
loop.leakyGain = 0.00

#%%
loop.setGain(0.0)
loop.leakyGain = 0.00
#%%
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
start_time = time.time()
for i in range(1000):
    loop.timeResolvedIntegratorWithLeak()
stop_time = time.time()

print((stop_time-start_time)/1000)

#%%



