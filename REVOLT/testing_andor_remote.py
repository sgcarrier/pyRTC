from pyRTC.Pipeline import *
from pyRTC.utils import *
import os


config = '../REVOLT/SPAD_PC_config.yaml'
# %% Launch Loop Class
wfs = hardwareLauncher("./hardware/AndorIXonCam.py", config, 3000, remoteProcess=True)
wfs.host = "132.246.193.79"
wfs.launch()


test_img =  wfs.getProperty("exposure")

print(test_img)
print(np.sum(test_img))


time.sleep(5)
