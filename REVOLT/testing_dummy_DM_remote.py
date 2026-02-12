#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.WavefrontCorrector import *

LISTENING_PORT = 3000

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)


#%%
################## Setup Andor Camera ##############
confWFC = conf["wfc"]
dummy_wfc = WavefrontCorrector(conf=confWFC)
dummy_wfc.start()


#%%
l_wfc = Listener(dummy_wfc, port= int(LISTENING_PORT), host="0.0.0.0")

print("Received Connection")
try:
    while l_wfc.running:
        l_wfc.listen()
        time.sleep(1e-3)
        print(dummy_wfc.currentCorrection)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught! Exiting loop gracefully.")
    dummy_wfc.stop()

