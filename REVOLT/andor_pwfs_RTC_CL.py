#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *

from pyRTC.LoopWithRemoteWFC import *

#%%

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

#%%

################## Setup Andor Camera ##############
confWFS = conf["wfs"]
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()
wfs.start()

#%%

# TODO test turning on the Andor cooler
wfs.cam.set_cooler(on=True)
wfs.cam.get_temperature_setpoint(20)
wfs.cam.get_temperature()
#wfs.cam.set_temperature(20)

#%% Close all

wfs.stop()
time.sleep(1)
wfs.close_shutter()
time.sleep(1)
wfs.close_camera()
time.sleep(1)


#%%
################## Setup MPD SPAD Camera ##############
confTRWFS = conf["trwfs"]
trwfs = MPDSPADCam(conf=confTRWFS)
trwfs.start()

#%%
trwfs.record_data(100, "test2")

#%%
################## Setup Full Frame Signal ##############

# For normal PWFS
#sig = FullFrameProcess(conf=conf)
#sig.start()

# For TR PWFS
sig = TimeResolvedFullFrameProcessCustomArea(conf=conf)
sig.start()




#%%
################## Setup WFC ##############
confWFC = conf["wfc"]
remote_wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", confWFC, 3000, remoteProcess=True)
remote_wfc.host = "132.246.192.209"
remote_wfc.launch()
#%%
#a =remote_wfc.run("read")
a = remote_wfc.getProperty("currentCorrection")


#%%

remote_wfc.run("push", 10, 0.01)

#%%
remote_wfc.run("flatten")

#%%
remote_wfc.run("sendToHardware")

#%%
################## Setup loop ##############


loop = LoopWithRemoteWFS(conf, remote_wfc)



#%%













