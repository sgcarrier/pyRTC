# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *
from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Loop import *

from hardware.TimeResolvedLoop import *


from pyRTC.utils import *


# %% 
################### Load configs ###################
conf = read_yaml_file("conf_simple.yaml")

confDMPSF  = conf[  "dmpsf"]
confDM     = conf[    "wfc"]
confMODPSF = conf[ "modpsf"]
confMOD    = conf[    "fsm"]
confWFS    = conf[    "wfs"]
confSHWFS  = conf[  "shwfs"]
confLOOP   = conf[   "loop"]


# %% 
################### Start PSF Cam ###################
# Start PSF Cam
dmpsf = PGScienceCam(conf=confDMPSF)
dmpsf.start()

#time.sleep(1)
#modpsf = PGScienceCam(conf=confMODPSF)
#modpsf.start()

#%% 
################### Load ALPAO DM and flatten ###################
wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()

#%%
################### Display DM current shape ###################
plt.figure()
currentShape2D = np.zeros(wfc.layout.shape)
currentShape2D[wfc.layout] = wfc.currentShape
plt.imshow(currentShape2D)
plt.colorbar()
plt.show()
# %% 
################### Setup Pi modulator ###################
fsm = PIE517Modulator(conf=confMOD)
pos = {"A": 5.0, "B": 5.0}

fsm.goTo(pos)


time.sleep(1)

print(fsm.getCurrentPos())

#%%
################### Plot FSM path ###################
from pipython import GCSDevice, pitools
max_pos_x = []
max_pos_y = []
for i in range(len(fsm.points)):
    fsm.step()
    time.sleep(0.01)
    im = modpsf.read()
    max_pos_x.append(np.unravel_index(im.argmax(), im.shape)[0])
    max_pos_y.append(np.unravel_index(im.argmax(), im.shape)[1])
    pitools.waitonready(fsm.mod)

plt.figure()
plt.scatter(max_pos_x, max_pos_y)
plt.show()    
print(f"x_range={np.max(max_pos_x) - np.min(max_pos_x)}, y_range={np.max(max_pos_y) - np.min(max_pos_y)}")
# %%
################### Setup Andor Camera ###################
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()

wfs.start()
wfs.setExposure(0.0625)

#%%
################### Start modulation ###################
fsm.start()
#dmpsf.setExposure(1000)

#%%
################### Stop modulation ###################
fsm.stop()
#dmpsf.setExposure(500)


#%%
################### Pupil positioning ###################
from scripts.pupilMask import *
pupil_pos = autoFindAndDisplayPupils(wfs)

#%%
for i in range(48):
    fsm.step()
    time.sleep(1)

#%% 
################### Calculate slopes ################### 
slope = SlopesProcess(conf=conf)
slope.start()


#%%
################### Plot pupils ###################
slope.plotPupils()
overlayCalcPosWithPupilMask(pupil_pos, slope)

#%% 
################### Create loop ###################
loop = Loop(conf=conf)

# %%
################### Compute Interaction Matrix ###################
loop.computeIM()
wfc.flatten()

#%%
################### Plot SVD modes ###################
plt.figure()
u,s,v = np.linalg.svd(loop.IM)
plt.plot(s/np.max(s), label = 'EMPIRICAL')
plt.yscale("log")
plt.ylim(1e-3,1.5)
plt.xlabel("Eigen Mode #", size = 18)
plt.ylabel("Normalizaed Eigenvalue", size = 18)
plt.axvline(x = loop.numActiveModes-1, color = 'r', label = 'axvline - full height')
plt.legend()
plt.show()

#%%
################### Plot IM and CM ###################
plt.imshow(loop.IM, cmap = 'inferno', aspect='auto')
plt.colorbar()
plt.show()
plt.imshow(loop.CM, cmap = 'inferno', aspect='auto')
plt.colorbar()
plt.show()
#%%
loop.start()
time.sleep(10)
loop.stop()
print(np.max(np.abs(wfc.currentShape)))

#%%
wfc.flatten()

# %%
################### Stop all ###################
fsm.stop()
time.sleep(1)
wfs.stop()
time.sleep(1)
wfs.close_shutter()
time.sleep(1)
wfs.close_camera()
time.sleep(1)

wfc.stop() 

time.sleep(1)
slope.stop()

time.sleep(1)
dmpsf.stop()



# %%
