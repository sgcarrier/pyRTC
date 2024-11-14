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

#loop.grabRefTRSlopes()
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
time.sleep(1)
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
import ctypes
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

def grabHASOCoeffs(camera, confSHWFS, coeffs):
    img_test = camera.snap_raw_image()

    try:
        slopes = wkpy.HasoSlopes(image= img_test, config_file_path = confSHWFS["confFile"])
    except Exception as e:
        print(str(e))
        print("Error: Most likely there is not enough light to determine slopes")
        return None

    coeffs = wkpy.ModalCoef(modal_normalization=wkpy.E_ZERNIKE_NORM.STD, nb_coeffs_total=coeffs, hasoslopes=slopes)
    values = coeffs.get_coefs_values()
    return np.array(values[0])

#%%
wfs.activateNoise = False
wfs.total_photon_flux = 0
loop.turbulenceGenerator = None
time.sleep(1)
for i in range(10):
    loop.timeResolvedIntegratorWithTurbulence()
    time.sleep(0.1)


#%%
# Grab ref wavefront 
REF_WF = grabHASOImage(camera, confSHWFS)
plt.imshow(REF_WF)
plt.colorbar()

#%%
# Grab ref coeffs 
NUM_MODES = loop.numModes
REF_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
plt.bar(list(range(NUM_MODES)), REF_coeff)

#%%
# Grab current coeffs 
CUR_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
plt.bar(list(range(NUM_MODES)), REF_coeff-CUR_coeff)




#%%
# Get the conversion matrix from KL space to HASO space
fsm.stop()
fsm.resetPos()
wfc.flatten()
REF_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)

HASO_resp = np.zeros((NUM_MODES, NUM_MODES))

poke_amp = 0.02
#For each mode
for i in range(NUM_MODES):
    wfc.push(i, poke_amp)
    #Add some delay to ensure one-to-one
    time.sleep(0.1)
    #Burn the first new image since we were moving the DM during the exposure
    push_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
    #push_coeff -= REF_coeff

    time.sleep(1)

    wfc.push(i, -poke_amp)
    #Add some delay to ensure one-to-one
    time.sleep(0.1)
    #Burn the first new image since we were moving the DM during the exposure
    pull_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
    #pull_coeff -= REF_coeff


    #Compute the normalized difference
    HASO_resp[i,:] = (push_coeff-pull_coeff)/(2*poke_amp)


#%%
plt.imshow(HASO_resp, cmap = 'inferno', aspect='auto')
plt.colorbar()
plt.show()
#%%
wfc.flatten()
loop.resetCurrentCorrection()

#%%
################### Turbulence ###################
from scripts.turbulencePreGen import *
turb = np.load("res/turb_coeff_Jun21_with_floating.npy")


turb /= 1
turb_no_piston = turb[:,1:]
turb_no_piston_first_5_modes = turb_no_piston
#turb_no_piston_first_5_modes[:, 1:] = 0
atm = DummyAtm(turb_no_piston_first_5_modes)

t = atm.getNextTurbAsModes()
t_cmd =ModaltoZonalWithFlat(t, 
                     wfc.f_M2C,
                     wfc.flat)


print(np.max(np.abs(t_cmd)))
#%%
atm.currentPos = 0
t = atm.getNextTurbAsModes()

t_cmd = wfc.f_M2C@t
phase_screen = np.zeros((11,11))
phase_screen[atm.mask] =t_cmd 
plt.imshow(phase_screen)
plt.colorbar()

#%% Animate
import matplotlib.animation as animation
t = atm.getNextTurbAsModes()

t_cmd = wfc.f_M2C@t
phase_screen = np.zeros((11,11))
phase_screen[atm.mask] =t_cmd 
fig = plt.figure()
img = plt.imshow(phase_screen)
ann = plt.annotate(str(0), (0,0))
plt.colorbar(img)
def animate(i):
    t = atm.atm[i,:]
    t_cmd = wfc.f_M2C@t
    phase_screen = np.zeros((11,11))
    phase_screen[atm.mask] =t_cmd 
    img.set_data(phase_screen)
    img.set_clim(np.min(phase_screen), np.max(phase_screen))
    ann.set_text(str(i))

anim = animation.FuncAnimation(fig, animate, frames= 100*10, interval=1000/10)

anim.save("test_anime.gif", fps=10)

#%% Animate with atmo
import matplotlib.animation as animation



fig = plt.figure()
img = plt.imshow(REF_WF)
ann = plt.annotate(str(0), (0,0))
plt.colorbar(img)
def animate(i):
    t = atm.atm[i,:]
    loop.wfcShm.write(t)
    read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
    read_WF_valid = read_WF[~np.isnan(read_WF)] 
    rms_val = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
    img.set_data(read_WF)
    img.set_clim(np.min(read_WF_valid), np.max(read_WF_valid))
    ann.set_text(f"f={i},rms={int(rms_val*1000)}nm")

anim = animation.FuncAnimation(fig, animate, frames= 10*10, interval=1000/10)
wfc.flatten()
loop.resetCurrentCorrection()
anim.save("atmo.gif", fps=10)
#%%

loop.setTurbulenceGenerator(atm)


#%%
wfs.activateNoise = True
wfs.total_photon_flux = 10
#%%
iterations = 100
loop.setGain(0.3)
strehls = np.zeros(iterations)
rms_plot = np.zeros(iterations)
REF_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
saved_coeffs = np.zeros((iterations, NUM_MODES))
saved_WFs = np.zeros((iterations, REF_WF.shape[0], REF_WF.shape[1]))
img_slopes = []
current_wfc_shape = np.zeros((iterations, wfc.layout.shape[0], wfc.layout.shape[1]))
turb_modes  = np.zeros((iterations, NUM_MODES))
corr_modes  = np.zeros((iterations, NUM_MODES))
for i in range(iterations):
    #loop.turbulenceGenerator.currentPos +=10
    loop.timeResolvedIntegratorWithTurbulence()
    #loop.standardIntegratorWithTurbulence()
    time.sleep(0.1)
    #strehls[i] = dmpsf.strehl_ratio
    read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
    read_WF_valid = read_WF[~np.isnan(read_WF)] 
    saved_WFs[i,:,:] = read_WF
    rms_plot[i] = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
    CUR_coeff = (REF_coeff - grabHASOCoeffs(camera, confSHWFS, NUM_MODES))
    saved_coeffs[i,:] = CUR_coeff
    current_wfc_shape[i,wfc.layout] = wfc.currentShape
    print(np.max(np.abs(wfc.currentShape)))
    print(f"it = {i}, RMS={rms_plot[i]}")
    img_slopes.append(loop.latest_slopes)
    turb_modes[i,:] = loop.turbModes
    corr_modes[i,:] = loop.latest_correction


#%%
wfc.flatten()
loop.resetCurrentCorrection()

#%%
import pickle


HASO_resp_inv = np.linalg.inv(HASO_resp.T)
coeffs_in_KL_space = np.zeros((iterations, NUM_MODES))

for i in range(iterations):
    coeffs_in_KL_space[i,:] = HASO_resp_inv @ saved_coeffs[i, :]


saveFilename = f"norm_low_photon_{wfs.total_photon_flux*48}photons_{loop.gain}g.pickle"
data = {"REF_coeff":REF_coeff,
        "saved_coeffs":saved_coeffs,
        "current_wfc_shape":current_wfc_shape,
        "img_slopes":img_slopes,
        "turb_modes":turb_modes,
        "corr_modes":corr_modes,
        "saved_WFs":saved_WFs,
        "rms_plot":rms_plot,
        "HASO_resp":HASO_resp,
        "HASO_resp_inv":HASO_resp_inv,
        "coeffs_in_KL_space":coeffs_in_KL_space,
        "slope_validSubAps":slope.validSubAps}

with open(saveFilename, 'wb') as handle:
    pickle.dump(data, handle)

#%%
import matplotlib.animation as animation
#saveFilename = f"fixed_normal_{wfs.total_photon_flux*48}photons_{loop.gain}g.pickle"

with open(saveFilename, 'rb') as handle:
    loaded_data = pickle.load(handle)

fig, axs = plt.subplots(3, 2)
i=0
num_modes_display = len(loaded_data["coeffs_in_KL_space"][i,:])

im_dm = axs[0,0].imshow(loaded_data["current_wfc_shape"][i,:,:])
axs[0,0].set_title("DM Shape (flat sub.)")
im_dm_cb =fig.colorbar(im_dm, ax=axs[0,0])

im_WF = axs[0,1].imshow(loaded_data["saved_WFs"][i,:,:])
axs[0,1].set_title("HASO SH wavefront (um)")
im_WF_cb =fig.colorbar(im_WF, ax=axs[0,1])

rms_line, = axs[1,0].plot(loaded_data["rms_plot"][:i])
rms_line.axes.set_xlim(0, len(loaded_data["rms_plot"]))
rms_line.axes.set_ylim(0, np.max(loaded_data["rms_plot"]))
axs[1,0].set_title("RMS of wavefront")

modal_haso_bar = axs[1,1].stem(list(range(num_modes_display)), loaded_data["coeffs_in_KL_space"][i,:], bottom=0)
axs[1,1].set_title("Modal comp from HASO in KL space")


mask = loaded_data["slope_validSubAps"]
img_slopes = np.zeros((mask.shape[1],mask.shape[0]))
img_slopes[mask.T] = np.sum(loaded_data["img_slopes"][i], axis=1)

im_slopes = axs[2,0].imshow(img_slopes.T)
axs[2,0].set_title("Slopes")
im_slopes_cb =fig.colorbar(im_slopes, ax=axs[2,0])

if i == 0 :
    prev = 0
else:
    prev = loaded_data["turb_modes"][i-1,:]
turb_modes_bar = axs[2,1].stem(list(range(num_modes_display)), loaded_data["turb_modes"][i,:] - prev, 'r',markerfmt='ro', label="Perfect corr")
corr_modes_bar = axs[2,1].stem(list(range(num_modes_display)), prev + loaded_data["corr_modes"][i,:], 'g', markerfmt='go', label="Turb - Corr modes")
axs[2,1].set_title("Modal turb and correction")
axs[2,1].legend()

def update_stem(h_stem, x=None, y=None, bottom=None):
    if x is None:
        x = h_stem[0].get_xdata()
    else:
        h_stem[0].set_xdata(x)
        h_stem[2].set_xdata([np.min(x), np.max(x)])

    if y is None:
        y = h_stem[0].get_ydata()
    else:
        h_stem[0].set_ydata(y)

    if bottom is None:
        bottom = h_stem[2].get_ydata()[0]
    else:
        h_stem[2].set_ydata([bottom, bottom])

    h_stem[1].set_paths([np.array([[xx, bottom], 
                                   [xx, yy]]) for (xx, yy) in zip(x, y)])

def animate(i):

    im_dm.set_data(loaded_data["current_wfc_shape"][i,:,:])
    im_dm.set_clim(np.min(loaded_data["current_wfc_shape"][i,:,:]), np.max(loaded_data["current_wfc_shape"][i,:,:]))

    im_WF.set_data(loaded_data["saved_WFs"][i,:,:])
    read_WF_valid = loaded_data["saved_WFs"][i,:,:][~np.isnan(loaded_data["saved_WFs"][i,:,:])] 
    wf_min = np.min(read_WF_valid) 
    wf_max = np.max(read_WF_valid) 
    im_WF.set_clim(wf_min, wf_max)


    mask = loaded_data["slope_validSubAps"]
    img_slopes = np.zeros((mask.shape[1],mask.shape[0]))
    img_slopes[mask.T] = np.sum(loaded_data["img_slopes"][i], axis=1)

    im_slopes.set_data(img_slopes.T)
    im_slopes.set_clim(np.min(img_slopes.T), np.max(img_slopes.T))
    
    rms_line.set_xdata(list(range(i)))
    rms_line.set_ydata(loaded_data["rms_plot"][:i])


    if i == 0 :
        prev = 0
    else:
        prev = loaded_data["turb_modes"][i-1,:]

    update_stem(modal_haso_bar, y=loaded_data["coeffs_in_KL_space"][i,:])
    axs[1,1].set_ylim(np.min(loaded_data["coeffs_in_KL_space"][i,:]), np.max(loaded_data["coeffs_in_KL_space"][i,:]))
    update_stem(turb_modes_bar, y=loaded_data["turb_modes"][i,:] - prev)
    update_stem(corr_modes_bar, y=prev + loaded_data["corr_modes"][i,:])

    min_both  = np.min([np.min(loaded_data["turb_modes"][i,:] - prev),  np.min(prev + loaded_data["corr_modes"][i,:])])
    max_both  = np.max([np.max(loaded_data["turb_modes"][i,:] - prev),  np.max(prev + loaded_data["corr_modes"][i,:])])
    axs[2,1].set_ylim(min_both, max_both)


anim = animation.FuncAnimation(fig, animate, frames= 50, interval=1000/5)

anim.save("recording_norm_low.gif", fps=5)

#%%
plt.plot(strehls)
plt.xlabel("Loop cycles")
plt.ylabel("Strehl")

#%%
plt.plot(rms_plot)
plt.xlabel("Loop cycles")
plt.ylabel("RMS (um)")


#%%
import matplotlib.animation as animation
mask = slope.validSubAps
img_slopes_sum = np.zeros((mask.shape[1],mask.shape[0], 50))
for i in range(len(img_slopes)):
    img_slopes_sum[mask.T,i] =np.sum(img_slopes[i], axis=1)

fig = plt.figure()
img = plt.imshow(img_slopes_sum[:,:, 0])
ann = plt.annotate(str(0), (0,0))
plt.colorbar(img)
def animate(i):
    img.set_data(img_slopes_sum[:,:,i])
    img.set_clim(np.min(img_slopes_sum[:,:,i]), np.max(img_slopes_sum[:,:,i]))
    ann.set_text(str(i))

anim = animation.FuncAnimation(fig, animate, frames= 50, interval=1000/5)

anim.save("test_img_norm.gif", fps=5)

#%%
import matplotlib.animation as animation
mask = slope.validSubAps
img_slopes = loop.getTRSlopes()
img_slopes_sum = np.zeros((mask.shape[1],mask.shape[0], loop.numFrames))
for i in range(loop.numFrames):
    img_slopes_sum[mask.T,i] =img_slopes[:,i]

fig = plt.figure()
img = plt.imshow(img_slopes_sum[:,:, 0])
ann = plt.annotate(str(0), (0,0))
plt.colorbar(img)
def animate(i):
    img.set_data(img_slopes_sum[:,:,i])
    img.set_clim(np.min(img_slopes_sum[:,:,i]), np.max(img_slopes_sum[:,:,i]))
    ann.set_text(str(i))

anim = animation.FuncAnimation(fig, animate, frames=loop.numFrames, interval=1000/5)

anim.save("slopes_modulation.gif", fps=5)

#%%
remove_first_its = 30
HASO_resp_inv = np.linalg.inv(HASO_resp.T)
coeffs_in_KL_space = np.zeros((iterations-remove_first_its, NUM_MODES))

for i in range(iterations-remove_first_its):
    coeffs_in_KL_space[i,:] = HASO_resp_inv @ saved_coeffs[remove_first_its+i, :]

rms_for_each_modes = np.zeros((NUM_MODES))
for i in range(NUM_MODES):
    rms_for_each_modes[i] = np.sqrt(np.mean(coeffs_in_KL_space[:,i]**2))

plt.plot(rms_for_each_modes)

#%%
plt.plot(rms_plot_norm_10*1000, label="Normal")
plt.plot(rms_plot_tr_10*1000, label="TR")
plt.xlabel("Loop cycles")
plt.ylabel("RMS (nm)")
plt.title(f"Wavefront RMS in closed loop \n {wfs.total_photon_flux*48} photons. 68 KL Modes, {loop.gain} loop gain")
plt.legend()

#%%
norm_data_file = "norm_low_photon_480photons_0.3g.pickle"
tr_data_file = "tr_low_photon_480photons_0.5g.pickle"

with open(norm_data_file, 'rb') as handle:
    loaded_data_norm = pickle.load(handle)
rms_plot_norm = loaded_data_norm["rms_plot"]


with open(tr_data_file, 'rb') as handle:
    loaded_data_tr = pickle.load(handle)
rms_plot_tr = loaded_data_tr["rms_plot"]
cutoff = 30
plt.plot(rms_plot_norm*1000, label=f"Normal {np.mean(rms_plot_norm[cutoff:])}")
plt.plot(rms_plot_tr*1000, label=f"TR {np.mean(rms_plot_tr[cutoff:])}")
plt.xlabel("Loop cycles")
plt.ylabel("RMS (nm)")
plt.title(f"Wavefront RMS in closed loop \n {wfs.total_photon_flux*48} photons. 68 KL Modes, {loop.gain} loop gain")
plt.legend()
#%%
wfc.flatten()
loop.resetCurrentCorrection()


#%%
loop.frame_weights = np.ones(loop.frame_weights.shape)/48
loop.IM = np.sum(loop.IM_cube * loop.frame_weights[np.newaxis, :, :], axis=1)
loop.computeCM()
#%%
slopes = loop.wfsShm.read()
for i in range(10-1):
    slopes+= loop.wfsShm.read()
slopes /= 10
print(f"Perceived modes 0,1 : {np.dot(loop.gCM, slopes)[0:5]}")

#%%
fsm.currentPos = None
slopes_TR = loop.getTRSlopes()
signal_per_mode = slopes_TR @ loop.frame_weights
new_corr = np.diag(loop.gCM.astype(np.float64) @ signal_per_mode)
print(f"Perceived modes : {new_corr[0:5]}")



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

#%%
wfs.activateNoise = True
wfs.total_photon_flux = 25
#%%
fsm.currentPos = None
image = slope.readImage()
signal_TR =  np.zeros((image.shape[0], image.shape[1], 48))
for s in range(fsm.numOfTRFrames):
    fsm.step()
    signal_TR[:, :,s] = slope.readImage()

plt.imshow(np.sum(signal_TR, axis=2))
plt.colorbar()

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


if camera is not None:
    camera.stop()
    camera.disconnect()


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