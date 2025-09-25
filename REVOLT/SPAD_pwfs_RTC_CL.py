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
# trwfs.setNFrames(1)
trwfs.enable_sync_mode()
trwfs.applySettingsToCamera()
#%%
trwfs.start()

#%%
trwfs.record_data_bypass(1000, "sky_with_dark_with_correction_1h09_24sept2025")

#%%
################## Setup Full Frame Signal ##############

# For TR PWFS
sig = TimeResolvedFullFrameProcessCustomArea(conf=conf)
sig.start()


#%%
################## Setup WFC ##############

# Dont forget to start the DM on the other PC
confWFC = conf["wfc"]
remote_wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", confWFC, 3000, remoteProcess=True)
remote_wfc.host = "132.246.192.209"
remote_wfc.launch()
#%%
a = remote_wfc.getProperty("currentCorrection")

#%%

remote_wfc.run("push", 1, 0.05)



#%%
newCorrection =np.zeros(100).astype(np.float32)
newCorrection[0] = 0.05
newCorrection[1] = -0.00
remote_wfc.run("write", newCorrection)

#%%
remote_wfc.run("flatten")

#%%
################## Setup loop ##############


loop = TimeResolvedLoopWithRemoteWFC(conf, None)



#%%













