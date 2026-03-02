import numpy as np
from numba import jit, prange
import time
import ctypes
import numpy as np

from pyRTC.Pipeline import hardwareLauncherUDP, ListenerUDP
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.utils import *


config = "../SPAD_PC_config.yaml"
conf = read_yaml_file(config)

confWFC = conf["wfc"]
dummy_wfc = WavefrontCorrector(conf=confWFC)
dummy_wfc.start()


l_wfc = ListenerUDP(dummy_wfc, port_cmds=6000, port_resps=6001, host="localhost")

print("Started UDP listener")

try:
    while l_wfc.running:
        l_wfc.listen()
        time.sleep(1e-5)
        #print(dummy_wfc.currentCorrection)
except KeyboardInterrupt:
    print("Stopping UDP listener")
    dummy_wfc.stop()

