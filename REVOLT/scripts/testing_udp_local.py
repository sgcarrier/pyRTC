
#%%
import numpy as np
from numba import jit, prange
import time
import ctypes
import numpy as np

from pyRTC.Pipeline import hardwareLauncherUDP, ListenerUDP
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.utils import *


config = "/home/simonc/Documents/Programming/pyRTC/REVOLT/SPAD_PC_config.yaml"
conf = read_yaml_file(config)

confWFC = conf["wfc"]


remote_wfc = hardwareLauncherUDP("../pyRTC/hardware/WavefrontCorrector.py",confWFC, port_cmds=6000, port_resps=6001, remoteProcess=True )
remote_wfc.host = "localhost"
remote_wfc.launch()

#%%

remote_wfc.run("flatten")

#%%
random_cmds = np.random.rand(50)

start_time = time.time()
for i in range(1000):
    remote_wfc.run("write", random_cmds)
stop_time = time.time()

print(f"{1/((stop_time-start_time)/1000):.5f}Hz")

#%%