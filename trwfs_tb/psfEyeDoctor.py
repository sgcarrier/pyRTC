# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *

from pyRTC.utils import *

import matplotlib.pyplot as plt


# %% 
# Load configs
conf = read_yaml_file("conf.yaml")

confDMPSF  = conf[  "dmpsf"]
confDM     = conf[    "wfc"]
confMODPSF = conf[ "modpsf"]
confMOD    = conf[    "fsm"]
confWFS    = conf[    "wfs"]


# %% Create DM PSF Cam
#dmpsf = PGScienceCam(conf=confDMPSF)
#dmpsf.start()

#time.sleep(1)
modpsf = PGScienceCam(conf=confMODPSF)
modpsf.start()

psf = modpsf

#%% Load DM and flatten
wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()


#%%

offset = np.zeros(97)

stepsPerMode = 11
modeSteps = np.linspace(-1, 1, stepsPerMode) * 0.001
numImagesToStack = 5
imStack = np.zeros((psf.imageShape[0], psf.imageShape[1], numImagesToStack))
maxInt = np.zeros(stepsPerMode)
maxModeToSharpen = 25
numPixelsToSharpen = 4

for mode in range(2, maxModeToSharpen):
    for s in range(stepsPerMode):
        wfc.flatten()
        #wfc.write(offset)
        # Apply to DM
        to_apply = offset
        to_apply[mode] =  modeSteps[s]
        print(f"to apply = {to_apply}")
        #wfc.push(mode, modeSteps[s])
        wfc.write(to_apply)

        print(f"Max of current shape:{np.max(np.abs(wfc.currentShape))}")
        time.sleep(0.1)
        # Grab PSF frames 
        for i in range(numImagesToStack):
            psf.expose()
            im = psf.read()
            imStack[:,:, i] = im
        ima = np.mean(imStack, axis=2)
        ima_flat = ima.flatten()
        idx = np.argsort(-ima_flat)
        sorted_values = ima_flat[idx]
        maxInt[s] = np.sum(sorted_values[:numPixelsToSharpen])

        # Average frame stack
    time.sleep(0.5)

    # determine best coeff
    idx = np.argmax(maxInt)
    # Add to the offset 
    offset[mode] = modeSteps[idx]
    
print(offset)
wfc.write(offset)
time.sleep(1)

# %%
psf.expose()
im = psf.read()

print(f"Final Shape: {wfc.currentShape}")

np.save("new_flat.npy", wfc.currentShape)


# %%

time.sleep(1)
wfc.flatten()
wfc.stop()

#dmpsf.stop()

# %%
