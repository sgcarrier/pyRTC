#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *


LISTENING_PORT = 3000

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
l_wfs = Listener(wfs, port= int(LISTENING_PORT), host="0.0.0.0")
while l_wfs.running:
    l_wfs.listen()
    time.sleep(1e-3)
# %%

wfs.stop()
time.sleep(1)
wfs.close_shutter()
time.sleep(1)
wfs.close_camera()

# %%
