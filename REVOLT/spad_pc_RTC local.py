#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
#%%

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

#%%

################## Setup Andor Camera ##############
confWFS = conf["wfs"]
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()
wfs.start()
wfs.setExposure(0.0625)

#%%

# TODO test turning on the Andor cooler
wfs.cam.set_cooler(on=True)
wfs.cam.get_temperature_setpoint()
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


#%%
trwfs.plot()

#%%