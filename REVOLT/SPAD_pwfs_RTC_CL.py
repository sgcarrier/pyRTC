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
trwfs.record_data_direct(1000, "hip14632_trget_with_dark_with_correction_12h50_25sept2025_2")

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

remote_wfc.run("push", 0, 0.05)





#%%
remote_wfc.run("flatten")

#%%
################## Setup loop ##############


loop = TimeResolvedLoopWithRemoteWFC(conf, remote_wfc, settings_name="trloop")



#%%

loop.computeIM()
loop.flatten()


#%%
saved = []
for i in range(500):
    loop.timeResolvedIntegratorWithLeak()
    #saved.append(loop.currentCorrection)
#%%
loop.flatten()
#%%
trwfs.record_data_direct(1000, "trwfs_0g3_ffw_2h08_25sept2025")

#%%
loop.start()

#%%
loop.stop()

#%%




# %%
