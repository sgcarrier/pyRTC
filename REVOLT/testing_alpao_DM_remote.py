#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.ALPAODM import *

LISTENING_PORT = 3000

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)


#%%
################## Setup Andor Camera ##############
confWFC = conf["wfc"]
wfc = ALPAODM(conf=confWFC)
wfc.start()
wfc.flatten()

#plt.figure()
#currentShape2D = np.zeros(wfc.layout.shape)
#currentShape2D[wfc.layout] = wfc.currentShape
#plt.imshow(currentShape2D)
#plt.colorbar()
#plt.show()

print("Setup DM")
#%%
try:
    l_wfc = Listener(wfc, port= int(LISTENING_PORT), host="0.0.0.0")
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught! Exiting loop gracefully.")
    wfc.flatten()
    wfc.stop()

print("Received Connection")

try:
    while l_wfc.running:
        l_wfc.listen()
        time.sleep(1e-5)
        #print(wfc.currentShape)
        #print(np.max(np.abs(wfc.currentShape)))
        if np.max(np.abs(wfc.currentShape)) > 0.8:
            print("WARNING ::: Actuator passed threshold 0.8")
            wfc.flatten()
            wfc.stop()
            time.sleep(1)
            break
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught! Exiting loop gracefully.")
    wfc.flatten()
    wfc.stop()

