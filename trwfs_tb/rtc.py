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
conf = read_yaml_file("conf.yaml")

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
dmpsf.setExposure(1000)

#%%
################### Stop modulation ###################
fsm.stop()
dmpsf.setExposure(500)


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
#loop = Loop(conf=conf)
loop = TimeResolvedLoop(conf=conf, fsm=fsm)


#%%
loop.calcFrameWeights()

#%%
plt.figure()
im1 = plt.imshow(loop.frame_weights)
plt.colorbar(im1)
plt.title("Measured weights\nfor each modulation frame and KL mode")
plt.ylabel("Modulation Frame")
plt.xlabel("KL mode")
plt.show()

#%%
loop.plotWeights()
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
time.sleep(5)
loop.stop()
print(np.max(np.abs(wfc.currentShape)))

#%%
dmpsf.takeModelPSF()

#%%
wfc.flatten()
loop.resetCurrentCorrection()



#%%
import wavekit_py as wkpy
confSHWFS  = conf[  "shwfs"]
try :
    camera = wkpy.Camera(config_file_path = confSHWFS["confFile"])
    camera.connect()
    camera.start(0, 1)
    camera.set_parameter_value("exposure_duration_us", 40000)
except Exception as e :
    print(str(e))

#%%
def grabHASOImage(camera, confSHWFS):
    img_test = camera.snap_raw_image()

    try:
        slopes = wkpy.HasoSlopes(image= img_test, config_file_path = confSHWFS["confFile"])
    except Exception as e:
        print(str(e))
        print("Error: Most likely there is not enough light to determine slopes")
        return None

    filter = [False, False, False, False, False]
    filter_array = (ctypes.c_byte * 5)()
    for i,f in enumerate(filter):
        filter_array[i] = ctypes.c_byte(0) if f else ctypes.c_byte(1)

    phase = wkpy.Phase(hasoslopes = slopes, type_ = wkpy.E_COMPUTEPHASESET.ZONAL, filter_ = filter_array)
    WF = phase.get_data()
    return WF[0]


#%%
# Grab ref wavefront 
REF_WF = grabHASOImage(camera, confSHWFS)
plt.imshow(REF_WF)
plt.colorbar()

#%%
################### Turbulence ###################
from scripts.turbulencePreGen import *
turb = np.load("res/turb_coeff_Jun21_with_floating.npy")


turb /= 1
turb_no_piston = turb[:,1:]
turb_no_piston_first_5_modes = turb_no_piston
#turb_no_piston_first_5_modes[:, 5:] = 0
atm = DummyAtm(turb_no_piston_first_5_modes)

t = atm.getNextTurbAsModes()
t_cmd =ModaltoZonalWithFlat(t, 
                     wfc.f_M2C,
                     wfc.flat)


print(np.max(np.abs(t_cmd)))
#%%
t = atm.getNextTurbAsModes()

t_cmd = wfc.f_M2C@t
phase_screen = np.zeros((11,11))
phase_screen[atm.mask] =t_cmd 
plt.imshow(phase_screen)
plt.colorbar()

#%%

loop.setTurbulenceGenerator(atm)


#%%
wfs.activateNoise = True
wfs.total_photon_flux = 1000
#%%
iterations = 50
strehls = np.zeros(iterations)
rms_plot = np.zeros(iterations)
for i in range(iterations):
    loop.timeResolvedIntegratorWithTurbulence()
    #loop.standardIntegratorWithTurbulence()
    time.sleep(1)
    strehls[i] = dmpsf.strehl_ratio
    read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
    read_WF_valid = read_WF[~np.isnan(read_WF)] 
    rms_plot[i] = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid)))) 
    print(np.max(np.abs(wfc.currentShape)))
    print(f"it = {i}")


#%%
plt.plot(strehls)
plt.xlabel("Loop cycles")
plt.ylabel("Strehl")

#%%
plt.plot(rms_plot)
plt.xlabel("Loop cycles")
plt.ylabel("RMS (um)")

#%%
plt.plot(rms_plot_normal*1000, label="Normal")
plt.plot(rms_plot_TR*1000, label="TR")
plt.xlabel("Loop cycles")
plt.ylabel("RMS (nm)")
plt.title("Wavefront RMS in closed loop \n48000 photons/modulation. 68 KL Modes")
plt.legend()
#%%
wfc.flatten()
loop.resetCurrentCorrection()


#%%
loop.frame_weights = np.ones(loop.frame_weights.shape)
loop.IM = np.sum(loop.IM_cube * loop.frame_weights[np.newaxis, :, :], axis=1)
loop.computeCM()
#%%
slopes = loop.wfsShm.read()
for i in range(10-1):
    slopes+= loop.wfsShm.read()
slopes /= 10
print(f"Perceived modes 0,1 : {np.dot(loop.gCM, slopes)[0:5]}")



# %% Try to get weighted cube
from scripts.modulation_weights import *

im_cube = calcIM_cube(wfc, fsm, loop, maxNumModes=30)

weighting_cube = modWeightsFromIMCube(im_cube=im_cube)


#%%
plt.figure()
im1 = plt.imshow(weighting_cube)
plt.colorbar(im1)
plt.title("Measured weights for each modulation frame and orthogonal mirror modes")
plt.ylabel("Modulation Frame")
plt.xlabel("Mirror mode")

#%%
np.save("frame_weights_20june2024.npy", weighting_cube)

#%%
import scripts.turbulenceGenerator
from scripts.turbulenceGenerator import OOPAO_atm
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
wfc.flatten()
loop.timeResolvedIntegratorWithTurbulence()
print(np.max(np.abs(wfc.currentShape)))
time.sleep(1)
wfc.flatten()
#%%
wfc.push(0, 0.01)
num_of_iterations = 5
strehl_cl = np.zeros(num_of_iterations)

for i in range(num_of_iterations):
    loop.timeResolvedIntegratorWithTurbulence()
    strehl_cl[i] = dmpsf.strehl_ratio
    print(np.max(np.abs(wfc.currentShape)))


#wfc.flatten()

#%% Get modes from commands

C2M = np.linalg.pinv(wfc.M2C)
atm_phase = atm.rebin(atm.getNextAtmOPD(), (11,11))
modes_to_send = C2M @  atm_phase[mask]
#modes_to_send = atm.getNextTurbAsModes()
modes_to_send[-1:] = 0
recon_phase = wfc.M2C @ modes_to_send

display_recon_phase = np.zeros((11,11))
display_recon_phase[mask] =recon_phase
plt.figure()
plt.imshow(atm_phase)
plt.colorbar()

plt.figure()
plt.imshow(display_recon_phase)
plt.colorbar()

#%%


#%% 
C2M = np.linalg.pinv(wfc.M2C)
mask = atm.genMask(11)

for i in range(5):
    atm_phase = atm.rebin(atm.getNextAtmOPD(), (11,11))
    modes_to_send = C2M @  atm_phase[mask]
    #modes_to_send[-5:] = 0
    print(np.max(np.abs(modes_to_send)))
    wfc.write(modes_to_send)
    time.sleep(1)
    wfc.flatten()



#%%
for i in range(5):
    modes_to_send = atm.getNextTurbAsModes()
    print(np.max(np.abs(modes_to_send)))
    wfc.write(modes_to_send)
    
    time.sleep(1)

#%% 
################### Save WFS fits ###################
from astropy.io import fits as pyfits
hdu = pyfits.PrimaryHDU(wfs.read())
hdu.writeto("0mod_flat.fits", overwrite=True)

#%%
from astropy.io import fits as pyfits
hdul = pyfits.open('turb_coeff_Jun21_with_floating.fits')
data = hdul[0].data

np.save("res/turb_coeff_Jun21_with_floating.npy", data)

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


if camera is not None:
    camera.stop()
    camera.disconnect()

wfc.stop()


time.sleep(1)
slope.stop()

time.sleep(1)
dmpsf.stop()


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