#%%
import argparse
import sys
import os 

import wavekit_py as wkpy

from pyRTC.utils import *
import matplotlib.pyplot as plt
import ctypes
import time
#%%
conf = read_yaml_file("conf.yaml")
confSHWFS  = conf[  "shwfs"]

try :
    camera = wkpy.Camera(config_file_path = confSHWFS["confFile"])
    camera.connect()
    camera.start(0, 1)
except Exception as e :
    print(str(e))

#%%

img_test = camera.snap_raw_image()


#%%

try:
    slopes = wkpy.HasoSlopes(image= img_test, config_file_path = confSHWFS["confFile"])
except Exception as e:
    print(str(e))
    print("test")

#%%
filter = [False, False, False, False, False]
filter_array = (ctypes.c_byte * 5)()
for i,f in enumerate(filter):
    filter_array[i] = ctypes.c_byte(0) if f else ctypes.c_byte(1)

phase = wkpy.Phase(hasoslopes = slopes, type_ = wkpy.E_COMPUTEPHASESET.ZONAL, filter_ = filter_array)
WF = phase.get_data()
rms, pv, max, min = phase.get_statistics()

#%%
if camera is not None:
    camera.stop()
    camera.disconnect()
# %%
