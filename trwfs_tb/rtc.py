# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *
from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Loop import *


from pyRTC.utils import *


# %% 
# Load configs
conf = read_yaml_file("conf.yaml")

confDMPSF  = conf[  "dmpsf"]
confDM     = conf[    "wfc"]
confMODPSF = conf[ "modpsf"]
confMOD    = conf[    "fsm"]
confWFS    = conf[    "wfs"]
confSHWFS  = conf[  "shwfs"]
confLOOP   = conf[   "loop"]


# %% 
# Start PSF Cam
#dmpsf = PGScienceCam(conf=confDMPSF)
#dmpsf.start()

#time.sleep(1)
modpsf = PGScienceCam(conf=confMODPSF)
modpsf.start()

#%% 
# Load ALPAO DM and flatten
wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()

#%%
plt.figure()
currentShape2D = np.zeros(wfc.layout.shape)
currentShape2D[wfc.layout] = wfc.currentShape
plt.imshow(currentShape2D)
plt.colorbar()
plt.show()
# %% 
# Setup Pi modulator
fsm = PIE517Modulator(conf=confMOD)
pos = {"A": 5.0, "B": 5.0}

fsm.goTo(pos)

time.sleep(1)

print(fsm.getCurrentPos())

#%%
from pipython import GCSDevice, pitools
max_pos_x = []
max_pos_y = []
for i in range(48):
    fsm.step()
    time.sleep(0.01)
    im = modpsf.read()
    max_pos_x.append(np.unravel_index(im.argmax(), im.shape)[0])
    max_pos_y.append(np.unravel_index(im.argmax(), im.shape)[1])
    pitools.waitonready(fsm.mod)
#%%
plt.figure()
plt.scatter(max_pos_x, max_pos_y)
plt.show()    
print(f"x_range={np.max(max_pos_x) - np.min(max_pos_x)}, y_range={np.max(max_pos_y) - np.min(max_pos_y)}")
# %%
# Setup Andor Camera
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()

wfs.start()
wfs.setExposure(0.062)

#%%
# Start modulation
fsm.start()
modpsf.setExposure(62500)

#%%
# Stop modulation
fsm.stop()
modpsf.setExposure(500)


#%%
from scripts.pupilMask import *
pupil_pos = autoFindAndDisplayPupils(wfs)


#%% Calculate slopes
slope = SlopesProcess(conf=conf)
slope.start()

#%%
slope.plotPupils()

#%%
overlayCalcPosWithPupilMask(pupil_pos, slope)

#%% 
# Create loop
loop = Loop(conf=conf)

# %%
loop.computeIM()
wfc.flatten()

#%%
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

#%% remove edge actuators 
# IM = loop.IM
# validAct = []
# for i in range(97):
#     if i not in edgeAct:
#         validAct.append(int(i))
# IM_sub = np.delete(IM, edgeAct, axis=1)
# loop.IM[:,edgeAct] = 0
# invIM = np.linalg.pinv(IM_sub, rcond=0)
# loop.CM = np.zeros((loop.numModes, loop.signalSize),dtype=loop.signalDType)
# loop.CM[validAct,:] = invIM
# loop.gCM = loop.gain*loop.CM
# loop.fIM = np.copy(loop.IM)

#%%
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
np.max(np.abs(wfc.currentShape))


# %% Try to get weighted cube
from scripts.modulation_weights import *

im_cube = calcIM_cube(wfc, fsm, loop, maxNumModes=30)

weighting_cube = modWeightsFromIMCube(im_cube=im_cube)


#%%
plt.figure()
im1 = plt.imshow(weighting_cube)
plt.colorbar(im1)
plt.title("Measured weights for each modulation frame and KL mode")
plt.ylabel("Modulation Frame")
plt.xlabel("KL mode")


#%%
from scripts.turbulenceGenerator import *
import matplotlib.pyplot as plt
atm = OOPAO_atm(wfc)

mask = atm.genMask(11)
atm_phase = atm.rebin(atm.getNextAtmOPD(), (11,11))
atm_phase[~mask] = 0

plt.figure()
plt.imshow(atm_phase)
plt.colorbar()


#%%

loop.setTurbulenceGenerator(atm)

#%%
for i in range(5):
    loop.standardIntegratorWithTurbulence()
    time.sleep(1)
np.max(np.abs(wfc.currentShape))

#wfc.flatten()

#%% Get modes from commands

C2M = np.linalg.pinv(wfc.M2C)

modes_to_send = C2M @  atm_phase[mask]

#%% 
C2M = np.linalg.pinv(wfc.M2C)
mask = atm.genMask(11)

for i in range(5):
    atm_phase = atm.rebin(atm.getNextAtmOPD(), (11,11)) *1000
    modes_to_send = C2M @  atm_phase[mask]
    print(np.max(np.abs(modes_to_send)))
    wfc.write(modes_to_send)
    
    time.sleep(1)



#%% 
#Save fits file
from astropy.io import fits as pyfits
hdu = pyfits.PrimaryHDU(wfs.read())
hdu.writeto("0mod_flat.fits", overwrite=True)
# %%
# Stop all
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
modpsf.stop()

# %%
def genMask( res, radius):
    xx,yy = np.meshgrid(np.arange(res),np.arange(res))
    zz = np.sqrt((xx-(res//2))**2 + (yy-(res//2))**2)
    actMap = np.zeros((res,res)).astype('bool')
    actMap[zz<=(radius)] = True
    return actMap

new_mask = genMask(11, 4.5)
plt.figure()
plt.imshow(new_mask)
plt.show()
# %%
