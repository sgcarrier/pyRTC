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
count = 0
count_max = 10000

try:
    while l_wfc.running:
        l_wfc.listen()
        time.sleep(1e-3)
        count += 1
        if count > count_max:
            print(dummy_wfc.correctionVector)
            count = 0
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught! Exiting loop gracefully.")
    l_wfc.stop()
