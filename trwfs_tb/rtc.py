# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *
from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess
from hardware.FullFrameProcess import FullFrameProcess
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
modpsf = PGScienceCam(conf=confMODPSF)
modpsf.start()

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
################### Calculate slopes ################### 
#slope = SlopesProcess(conf=conf)
slope = FullFrameProcess(conf=conf)
#slope.takeRefFullFrame()
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
# %%
################### Compute Interaction Matrix ###################
loop.computeIM()
wfc.flatten()

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
saved_cc = wfc.currentCorrection
print(saved_cc)
print(np.max(np.abs(wfc.currentShape)))

#%%
dmpsf.takeModelPSF()

#%%
wfc.flatten()
loop.resetCurrentCorrection()



#%% Take darks
dmpsf.takeDark()
dmpsf.saveDark("res/dmpsf_dark.npy")
modpsf.takeDark()
modpsf.saveDark("res/modpsf_dark.npy")
wfs.takeDark()
wfs.saveDark("res/andor_dark.npy")

#%%
import wavekit_py as wkpy

confSHWFS  = conf[  "shwfs"]
try :
    camera = wkpy.Camera(config_file_path = confSHWFS["confFile"])
    camera.connect()
    camera.start(0, 1)
    camera.set_parameter_value("exposure_duration_us", 15000)
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


def grabHASOStrehl(camera, confSHWFS):
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

    phase = wkpy.Phase(hasoslopes = slopes, type_ = wkpy.E_COMPUTEPHASESET.MODAL_ZERNIKE, filter_ = filter_array, nb_coeffs=200)
    strehl = wkpy.HasoField(config_file_path = confSHWFS["confFile"], hasoslopes=slopes, phase=phase, curv_radius=3.5, wavelength=635.0, oversampling=1)

    SR = strehl.strehl(config_file_path=confSHWFS["confFile"],flat_experimental_intensity=False, flat_theoretical_intensity=False, through_focus=False, oversample=True, defocus=0)
    
    return SR

#%%
loop.setGain(0.1)
wfs.activateNoise = False
wfs.total_photon_flux = 0
wfs.activateRONoise = False
loop.turbulenceGenerator = None
time.sleep(1)
for i in range(30):
    #loop.standardIntegratorWithTurbulence()
    loop.timeResolvedIntegratorWithTurbulence()
    time.sleep(0.1)

fsm.stop()



#%%
dmpsf.takeModelPSF()
dmpsf.box_size = 50
modpsf.takeModelPSF()
modpsf.box_size = 150
#%%
# Grab ref wavefront 

REF_WF = grabHASOImage(camera, confSHWFS)
for i in range(49):
    REF_WF += grabHASOImage(camera, confSHWFS)

REF_WF /= 50
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
wfc.flatten()
loop.resetCurrentCorrection()

#%%
################### Turbulence ###################
from scripts.turbulencePreGen import *
turb = np.load("res/turb_coeff_Jun21_with_floating.npy")


turb *= 1
turb_no_piston = turb[:,1:]
turb_no_piston_first_5_modes = turb_no_piston

# Remove some modes if needed
MODES_TO_USE = 30
turb_no_piston_first_5_modes[:, MODES_TO_USE:] = 0
loop.numActiveModes = MODES_TO_USE

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

#%%
atm.setSpeed(1)
loop.setTurbulenceGenerator(atm)


#%%
wfs.activateNoise = True
wfs.activateRONoise = False
wfs.total_photon_flux = 10



#%%

def runLoop(iterations, gain, photons_per_frame, switching_point=0, type_of_loop="tr", turb_wheel=None, delay=0):
    # Reset wheel
    if turb_wheel is not None:
        c = turb_wheel
        c('PA 60000')
        time.sleep(1)
        c('BG')
        time.sleep(3)
        print(f"Curent position = {c('TPA')}")
        time.sleep(1)
        c('PR ' + str(stepSize))
        time.sleep(1)


    # SEt Photons
    wfs.activateNoise = True
    wfs.activateRONoise = False
    wfs.total_photon_flux = photons_per_frame

    #Prep all arrays for recording loop
    fsm.stop()
    fsm.currentPos = None
    strehls = np.zeros(iterations)
    psf_img = np.zeros((iterations, 480, 640 ))
    strehls_mod = np.zeros(iterations)
    rms_plot = np.zeros(iterations)
    rss_plot = np.zeros(iterations)
    rms_strehl = np.zeros(iterations)
    REF_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
    saved_coeffs = np.zeros((iterations, NUM_MODES))
    saved_WFs = np.zeros((iterations, REF_WF.shape[0], REF_WF.shape[1]))
    img_slopes = []
    current_wfc_shape = np.zeros((iterations, wfc.layout.shape[0], wfc.layout.shape[1]))
    turb_modes  = np.zeros((iterations, NUM_MODES))
    corr_modes  = np.zeros((iterations, NUM_MODES))
    wavelenth = 0.635 # in um
    residual_modes = np.zeros((iterations, NUM_MODES))

    loop = TimeResolvedLoop(conf=conf, fsm=fsm)
    loop.setDelay(delay)
    # Reset flat 
    wfc.flatten()
    loop.resetCurrentCorrection()

    #Set gain
    loop.setGain(gain)
    switched = False

    if switching_point == 0:
        match type_of_loop:
            case "norm":
                loop.changeWeightsAndUpdate(np.ones(loop.frame_weights.shape))
            case "tr":
                pass
            case "ff":
                loop.switchToFF()
                loop.FF_active = True
            case "ff_w":
                loop.switchToFFwithWeights()
                loop.FF_weighted_active = True
    else:
        #Start with norm with default 0.3 gain
        loop.changeWeightsAndUpdate(np.ones(loop.frame_weights.shape))
        loop.setGain(0.3)

    # Run loop
    MAX_ACT_LIMIT = 0.35
    for i in range(iterations):
        if np.max(np.abs(wfc.currentShape)) > MAX_ACT_LIMIT:
            wfc.flatten()
            loop.resetCurrentCorrection()
            break
        try:
            #loop.turbulenceGenerator.currentPos +=10
            if (not switched) and (switching_point != 0) :
                if i > switching_point :  # switch to selected method
                    switched=True
                    saved_current_correction = loop.currentCorrection.copy()
                    loop = TimeResolvedLoop(conf=conf, fsm=fsm)
                    loop.setDelay(delay)
                    loop.setGain(gain)
                    match type_of_loop:
                        case "norm":
                            loop.changeWeightsAndUpdate(np.ones(loop.frame_weights.shape))
                        case "tr":
                            pass
                        case "ff":
                            loop.switchToFF()
                            loop.FF_active = True
                        case "ff_w":
                            loop.switchToFFwithWeights()
                            loop.FF_weighted_active = True
                    loop.currentCorrection = saved_current_correction
            
            loop.timeResolvedIntegratorWithTurbulence()
            
            #loop.standardIntegratorWithTurbulence()
            time.sleep(0.1)
            if (not switched) and (switching_point != 0) :
                if i <= switching_point :
                    residual_modes[i,:] = calc_TR_Residual(CM=loop.CM, 
                                                slopes_TR=loop.latest_slopes,
                                                weights=loop.frame_weights,
                                                ref_signal_per_mode_normed = loop.ref_signal_per_mode_normed)
            else:
                if type_of_loop == "norm" or type_of_loop == "tr":
                    residual_modes[i,:] = calc_TR_Residual(CM=loop.CM, 
                                                    slopes_TR=loop.latest_slopes,
                                                    weights=loop.frame_weights,
                                                    ref_signal_per_mode_normed = loop.ref_signal_per_mode_normed)
                elif type_of_loop == "ff":
                    residual_modes[i,:] = calc_TRFF_residual(CM=loop.CM, 
                                    slopes_TR=loop.latest_slopes.flatten(),
                                        ref_signal_normed = loop.ref_signal_normed)
                elif type_of_loop == "ff_w":
                    residual_modes[i,:] = calc_TRFF_residual_weighted(CM=loop.CM, 
                                    slopes_TR=loop.latest_slopes,
                                    weights=loop.frame_weights,
                                        ref_signal_per_mode_normed = loop.ref_signal_per_mode_normed)
            strehls[i] = dmpsf.strehl_ratio_ref()
            psf_img[i,:,:] = dmpsf.readLong()
            strehls_mod[i] = modpsf.strehl_ratio_ref()
            read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
            read_WF_valid = read_WF[~np.isnan(read_WF)] 
            saved_WFs[i,:,:] = read_WF
            rms_plot[i] = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
            rss_plot[i] = np.sqrt(np.sum(np.square(read_WF_valid - np.mean(read_WF_valid))))
            rms_strehl[i] = np.exp( -(2*np.pi*rms_plot[i]/wavelenth)**2)
            #CUR_coeff = (REF_coeff - grabHASOCoeffs(camera, confSHWFS, NUM_MODES))
            #saved_coeffs[i,:] = CUR_coeff
            #current_wfc_shape[i,wfc.layout] = wfc.currentShape
            print(np.max(np.abs(wfc.currentShape)))
            print(f"it = {i}, RMS={rms_plot[i]:.4f}, RMS_SR={rms_strehl[i]:.4f}, SR={strehls[i]:.4f}, mod_SR={strehls_mod[i]:.4f}")
            #img_slopes.append(loop.latest_slopes)
            #turb_modes[i,:] = loop.turbModes
            corr_modes[i,:] = loop.latest_correction
            if turb_wheel is not None:
                c('BGA')
            time.sleep(0.2)
        except KeyboardInterrupt:
            fsm.stop()
            print("Stopped loop")
            break
    
    # Save data
    data = {"photon_per_frame": wfs.total_photon_flux,
        "gain": loop.gain,
        "strehls" : strehls,
        "strehls_mod" : strehls_mod,
        "rms_strehl": rms_strehl,
        "corr_modes":corr_modes,
        "saved_WFs":saved_WFs,
        "rms_plot":rms_plot,
        "rss_plot":rss_plot,
        "slope_validSubAps":slope.validSubAps,
        "psf_img": psf_img,
        "residual_modes": residual_modes}

    print(f"Wheel final position = {c('TPA')}")

    return data

#%%

#data_ff_100_05g_d0_1m2 = runLoop(iterations=100, gain=0.5, photons_per_frame=100, switching_point=40, type_of_loop="ff", turb_wheel=c, delay=0)
#data_ff_50_05g_d0_1m2 = runLoop(iterations=100, gain=0.5, photons_per_frame=50, switching_point=40, type_of_loop="ff", turb_wheel=c, delay=0)
#data_ff_25_05g_d0_1m2 = runLoop(iterations=100, gain=0.5, photons_per_frame=25, switching_point=40, type_of_loop="ff", turb_wheel=c, delay=0)
#data_ff_10_05g_d0_1m2 = runLoop(iterations=100, gain=0.5, photons_per_frame=10, switching_point=40, type_of_loop="ff", turb_wheel=c, delay=0)
data_ff_5_05g_d0_1m2_offset = runLoop(iterations=100, gain=0.5, photons_per_frame=5, switching_point=40, type_of_loop="ff", turb_wheel=c, delay=0)
#data_ff_5_07g_1m2 = runLoop(iterations=100, gain=0.7, photons_per_frame=5, switching_point=40, type_of_loop="ff", turb_wheel=c)
#data_norm_100_03g_d2_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=100, type_of_loop="norm", turb_wheel=c, delay=2)
#data_norm_100_03g_d1_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=100, type_of_loop="norm", turb_wheel=c, delay=1)
#data_norm_100_03g_d0_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=100, type_of_loop="norm", turb_wheel=c, delay=0)
#data_norm_100_05g_d0_1m2 = runLoop(iterations=100, gain=0.5,  photons_per_frame=100, switching_point=40, type_of_loop="norm", turb_wheel=c, delay=0)
#data_norm_50_03g_d0_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=50, type_of_loop="norm", turb_wheel=c, delay=0)
#data_norm_25_03g_d0_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=25, type_of_loop="norm", turb_wheel=c, delay=0)
#data_norm_10_03g_d0_1m2 = runLoop(iterations=100, gain=0.3, photons_per_frame=10, type_of_loop="norm", turb_wheel=c, delay=0)
data_norm_5_03g_d0_1m2_offset = runLoop(iterations=100, gain=0.3, photons_per_frame=5, type_of_loop="norm", turb_wheel=c, delay=0)
#data_norm_5_05g_1m2 = runLoop(iterations=100, gain=0.5, photons_per_frame=5,switching_point=40, type_of_loop="norm", turb_wheel=c)
#data_ff_5_0g_1m2 = runLoop(iterations=100, gain=1.1, photons_per_frame=5, type_of_loop="ff", turb_wheel=c)

#%%
data_norm_100_03g_1m2 = data.copy()
#%%
strehl_stacked = calc_avg_strehl(data_ff_100_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_ff_50_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_ff_25_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_ff_10_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")

strehl_stacked = calc_avg_strehl(data_norm_100_03g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_norm_100_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_norm_50_03g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_norm_25_03g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")
strehl_stacked = calc_avg_strehl(data_norm_10_03g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")



strehl_stacked = calc_avg_strehl(data_ff_5_05g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")

strehl_stacked = calc_avg_strehl(data_norm_5_03g_d0_1m2["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")


strehl_stacked = calc_avg_strehl(data_ff_5_05g_d0_1m2_offset["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")

strehl_stacked = calc_avg_strehl(data_norm_5_03g_d0_1m2_offset["psf_img"][50:, :,:], dmpsf)
print(f"Stacked strehl = {strehl_stacked}")


#%%

plt.figure()
plt.plot(data_norm_5_03g_1m2["strehls"], label="norm")
plt.plot(data_tr_5_03g_1m2["strehls"], label="tr")
plt.legend()
#%%

fig, (ax1,ax2) = plt.subplots(2,1)
title = "Residual mode analysis, photon=" + str(data["photon_per_frame"])
fig.suptitle(title)
per_mode = np.std(data["residual_modes"][50:,:], axis=0)

for i in range(5):
    ax1.plot(data["residual_modes"][:,i], label="Mode={i}")
ax1.set_title("Residual modes for each cycle (um)")
ax1.axhline(0, color="r", ls="--")

ax2.bar(list(range(len(per_mode))), per_mode, label="Residual std per mode (last 50)")
ax2.set_title("std")
plt.legend()

#%%
import pickle

total_data = {"dmpsf_model": dmpsf.model,
              "modpsf_model": modpsf.model, 
              "haso_ref": REF_WF,
              "data_ff_100_05g_d0_1m2":data_ff_100_05g_d0_1m2,
              "data_ff_50_05g_d0_1m2":data_ff_50_05g_d0_1m2,
              "data_ff_25_05g_d0_1m2":data_ff_25_05g_d0_1m2,
              "data_ff_10_05g_d0_1m2":data_ff_10_05g_d0_1m2,
              "data_ff_5_05g_d0_1m2":data_ff_5_05g_d0_1m2,
              "data_norm_100_03g_d0_1m2":data_norm_100_03g_d0_1m2,
              "data_norm_100_05g_d0_1m2":data_norm_100_05g_d0_1m2,
              "data_norm_50_03g_d0_1m2":data_norm_50_03g_d0_1m2,
              "data_norm_25_03g_d0_1m2":data_norm_25_03g_d0_1m2,
              "data_norm_10_03g_d0_1m2": data_norm_10_03g_d0_1m2,
              "data_norm_5_03g_d0_1m2": data_norm_5_03g_d0_1m2,
              "data_ff_5_05g_d0_1m2_offset": data_ff_5_05g_d0_1m2_offset,
              "data_norm_5_03g_d0_1m2_offset":data_norm_5_03g_d0_1m2_offset,

}
with open("29aug2025_data_norm_vs_ff_switch40_1m2_0delay_turbwheel_mult_photons.pickle", 'wb') as handle:
    pickle.dump(total_data, handle)

#%%

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(data_norm_5_03g_1m2["rms_plot"]*1000, label="HASO RMS")
ax1.set_title("WFE RMS (nm)")

ax2.plot(data_norm_5_03g_1m2["strehls"], label="From DM PSF cam")
ax2.plot(data_norm_5_03g_1m2["rms_strehl"], "--", label="From HASO RMS")
ax2.plot(data_norm_5_03g_1m2["strehls_mod"], "-*", label="From MOD PSF cam")
ax2.set_title("Strehl")
plt.legend()
fig.suptitle(f'Closed-loop performance,{data_norm_5_03g_1m2["gain"]}g', fontsize=16)

#%%
c('PA 0')
time.sleep(1)
c('BG')
time.sleep(3)
print(f"Curent position = {c('TPA')}")
time.sleep(1)
c('PR ' + str(stepSize))
time.sleep(1)
#%%
fsm.stop()
fsm.currentPos = None
iterations = 100
loop.setGain(0.5)
strehls = np.zeros(iterations)
psf_img = np.zeros((iterations, 480, 640 ))
strehls_mod = np.zeros(iterations)
rms_plot = np.zeros(iterations)
rms_strehl = np.zeros(iterations)
REF_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
saved_coeffs = np.zeros((iterations, NUM_MODES))
saved_WFs = np.zeros((iterations, REF_WF.shape[0], REF_WF.shape[1]))
img_slopes = []
current_wfc_shape = np.zeros((iterations, wfc.layout.shape[0], wfc.layout.shape[1]))
turb_modes  = np.zeros((iterations, NUM_MODES))
corr_modes  = np.zeros((iterations, NUM_MODES))
wavelenth = 0.635 # in um
for i in range(iterations):
    try:
        #loop.turbulenceGenerator.currentPos +=10
        #loop.timeResolvedIntegratorWithTurbulence()
        #c('BGA')
        #loop.standardIntegratorWithTurbulence()
        time.sleep(0.1)
        strehls[i] = dmpsf.strehl_ratio_ref()
        psf_img[i,:,:] = dmpsf.readLong()
        strehls_mod[i] = modpsf.strehl_ratio_ref()
        read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
        read_WF_valid = read_WF[~np.isnan(read_WF)] 
        saved_WFs[i,:,:] = read_WF
        rms_plot[i] = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
        rms_strehl[i] = np.exp( -(2*np.pi*rms_plot[i]/wavelenth)**2)
        #CUR_coeff = (REF_coeff - grabHASOCoeffs(camera, confSHWFS, NUM_MODES))
        #saved_coeffs[i,:] = CUR_coeff
        #current_wfc_shape[i,wfc.layout] = wfc.currentShape
        print(np.max(np.abs(wfc.currentShape)))
        print(f"it = {i}, RMS={rms_plot[i]:.4f}, RMS_SR={rms_strehl[i]:.4f}, SR={strehls[i]:.4f}, mod_SR={strehls_mod[i]:.4f}")
        #img_slopes.append(loop.latest_slopes)
        #turb_modes[i,:] = loop.turbModes
        corr_modes[i,:] = loop.latest_correction
        c('BGA')
        time.sleep(0.1)
    except KeyboardInterrupt:
        fsm.stop()
        print("Stopped loop")
        break


#%%
wfc.flatten()
loop.resetCurrentCorrection()

#%%
def calc_avg_strehl( psfs, cam):
    psf= np.mean(psfs, axis=0)
    psf_diff = cam.model

    # Find the max for PSF
    ix, iy = np.unravel_index(np.argmax(psf), psf.shape)
    # Box the PSF
    psf_box = psf[ix-cam.box_size:ix+cam.box_size, iy-cam.box_size:iy+cam.box_size]
    # Find the max for diffraction PSF
    ix, iy = np.unravel_index(np.argmax(psf_diff), psf_diff.shape)
    # Box the diff PSF
    psf_diff_box = psf_diff[ix-cam.box_size:ix+cam.box_size, iy-cam.box_size:iy+cam.box_size]
    # Normalize diffraction PSF box
    psf_diff_box = psf_diff_box / np.max(psf_diff_box)
    # Normalize PSF box
    psf_box_norm = psf_box / np.sum(psf_box) * np.sum(psf_diff_box)
    # Oversample to get true peak (using scipy.ndimage.zoom with bicubic interpolation)
    psf_box_norm_oversampled = zoom(psf_box_norm, 4, order=3)  # order=3 for bicubic
    # Calculate Strehl ratio
    strehl_ratio = np.max(psf_box_norm_oversampled)
    # Handle invalid SR values
    if not np.isfinite(strehl_ratio) or strehl_ratio == 0:
        strehl_ratio = np.nan
    return strehl_ratio


#strehl_stacked = calc_avg_strehl(loop_data_tr_10_03g_10m_wind["psf_img"][50:, :,:], dmpsf)
#print(f"Stacked strehl = {strehl_stacked}")


#%%
@jit(nopython=True)
def calc_TR_Residual(CM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    signal_per_mode = slopes_TR @ weights 
    signal_per_mode_normed = signal_per_mode / np.sum(signal_per_mode, axis=0)
    #TODO Might be able to optimize this with einsum
    #new_corr = np.diag(gCM.astype(np.float64) @ (signal_per_mode_normed - ref_signal_per_mode_normed))
    nModes = weights.shape[1]
    new_corr = np.array([np.dot(CM.astype(np.float64)[i,:],  (signal_per_mode_normed - ref_signal_per_mode_normed)[:,i]) for  i in range(nModes)])

    return new_corr

@jit(nopython=True)
def calc_TRFF_residual(CM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       ref_signal_normed=np.array([[]], dtype=np.float64),):
    signal_normed = slopes_TR / np.sum(slopes_TR)
    #TODO Might be able to optimize this with einsum
    new_corr = CM.astype(np.float64) @ (signal_normed - ref_signal_normed)
    return new_corr
#%%
#saveFilename = f"tr_2KL_very_low_photon_14feb25_new_noise{wfs.total_photon_flux*48}photons_{loop.gain}g.pickle"
data = {"photon_per_frame": wfs.total_photon_flux,
        "gain": loop.gain,
        "strehls" : strehls,
        "strehls_mod" : strehls_mod,
        "rms_strehl": rms_strehl,
        "corr_modes":corr_modes,
        "saved_WFs":saved_WFs,
        "rms_plot":rms_plot,
        "slope_validSubAps":slope.validSubAps,
        "psf_img": psf_img}

turbulence_data_1m2 = data.copy()
#%%

from OOPAO.tools.displayTools           import displayMap
mod_cube = np.zeros((128, 128, 48))

signal_TR = loop.getTRSlopes()

# fsm.currentPos = None
# signal_TR =  np.zeros((loop.signalSize, 48))
# for s in range(loop.numFrames):
#     fsm.step()
#     signal_TR[:,s] = loop.wfsShm.read()


#%%
from OOPAO.tools.displayTools           import displayMap
mod_cube = np.zeros((128, 128, 48))
for f in range(48):
    mod_cube[slope.p1mask, f] = loop.latest_slopes[:loop.signalSize//4,f]
    mod_cube[slope.p2mask, f] = loop.latest_slopes[loop.signalSize//4:loop.signalSize//2,f]
    mod_cube[slope.p3mask, f] = loop.latest_slopes[loop.signalSize//2:loop.signalSize//4*3,f]
    mod_cube[slope.p4mask, f] = loop.latest_slopes[loop.signalSize//4*3:loop.signalSize,f]

plt.imshow(np.sum(mod_cube[:,:,:], axis=2))
displayMap(mod_cube)
#%%
mode_chosen = 0

fig,ax=plt.subplots()
#plt.plot(turb_modes[:,mode_chosen], "r-", label=f"turb")
plt.plot(-corr_modes[:,mode_chosen], "k-", label=f"corr")

# And a corresponding grid
ax.grid(which='both')

plt.legend()


#%%


fig,ax=plt.subplots()
for mode_chosen in range(30):
    #diff = turb_modes[:,mode_chosen] +  corr_modes[:,mode_chosen]
    diff = corr_modes[:,mode_chosen]
    plt.plot(diff, label=f"{mode_chosen}")

# And a corresponding grid
ax.grid(which='both')
fig.suptitle(f'DM coeffs for each mode per iteration', fontsize=16)
fig.legend()


#%%

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(np.diff(rms_plot)*1000, label="HASO RMS")
ax1.set_title("WFE RMS (nm)")

# ax2.plot(strehls, label="From DM PSF cam")
# ax2.plot(rms_strehl, "--", label="From HASO RMS")
# ax2.plot(strehls_mod, "-*", label="From MOD PSF cam")
# ax2.set_title("Strehl")
plt.legend()
fig.suptitle(f'Closed-loop performance,{loop.gain}g', fontsize=16)
#%%

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(rms_plot_norm_10_03g*1000)
ax1.set_title("WFE RMS (nm)")

ax2.plot(strehl_norm_10_03g)
ax2.set_title("Strehl")

fig.suptitle(f'Closed-loop performance NORM 10 0.3g', fontsize=16)

#%%

fig, (ax1,ax2) = plt.subplots(2,1)


#ax1.plot(turb_rms*1000, "x--", label="Turb")
#ax1.plot(np.abs(np.diff(turb_rms, prepend=turb_rms[0]))*1000, "o--", label="Perfect")

ax1.plot(data_norm_10_03g["rms_plot"]*1000, "r", label="Modulated - 0.3g")
#ax1.plot(rms_plot_norm_10_01g*1000, "r--",label="Modulated - 0.1g")
#ax1.plot(rms_plot_tr_2_01g*1000, "b",label="TR - 0.1g")
ax1.plot(data_tr_10_03g["rms_plot"]*1000, "b--", label="TR - 0.3g")
#ax1.plot(rms_plot_ff_10_03g*1000, "g",label="TR-Cube - 0.3g")
ax1.plot(data_ff_10_05g["rms_plot"]*1000, "g--",label="TR-Cube - 0.5g")
ax1.set_title("WFE RMS (nm)")

#ax2.plot(turb_sr, "x--", label="Turb")
#ax2.plot(strehl_norm_2_01g, "r",label="Modulated - 0.1g")
ax2.plot(data_norm_10_03g["strehls"], "r--", label="Modulated - 0.3g")
#ax2.plot(strehl_tr_2_01g, "b",label="TR - 0.1g")
ax2.plot(data_tr_10_03g["strehls"],"b--", label="TR - 0.3g")
#ax2.plot(strehl_ff_10_03g,"g", label="TR-Cube - 0.3g")
ax2.plot(data_ff_10_05g["strehls"], "g--",label="TR-Cube - 0.5g")
ax2.set_title("Strehl")

ax1.legend()

#ax1.axvline(x=30, color='red', linestyle='--')
#ax2.axvline(x=30, color='red', linestyle='--')

fig.suptitle(f'Closed-loop performance, fixed turb: {wfs.total_photon_flux*48} photons,  gain={loop.gain}', fontsize=16)

#fig.suptitle(f'Closed-loop performance, fixed turb: No restrict photons,  gain={loop.gain}', fontsize=16)


#%%
import pickle


HASO_resp_inv = np.linalg.inv(HASO_resp.T)
coeffs_in_KL_space = np.zeros((iterations, NUM_MODES))

for i in range(iterations):
    coeffs_in_KL_space[i,:] = HASO_resp_inv @ saved_coeffs[i, :]


saveFilename = f"tr_2KL_very_low_photon_14feb25_new_noise{wfs.total_photon_flux*48}photons_{loop.gain}g.pickle"
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
fig,ax=plt.subplots()
plt.plot(rms_plot_norm_50_03g_30KL_3xspeed*1000, "r-", label=f"Normal - {50*48} photons - G=0.3")
plt.plot(rms_plot_norm_20_03g_30KL_3xspeed*1000, "k--", label=f"Normal - {20*48} photons - G=0.3")
plt.plot(rms_plot_tr_50_03g_30KL_3xspeed*1000, "b-" , label=f"TR - {50*48} photons - G=0.3")
#plt.plot(rms_plot_tr_10_03_5xspeed*1000, "b--",label="TR - 2400 photons - G=0.3")
plt.xlabel("Loop cycles")
plt.ylabel("RMS (nm)")
plt.title(f"Wavefront RMS in closed loop 3x speed turbulence \n 30 KL of turb/corr, reduced pupil")
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1000, 100)
minor_ticks = np.arange(0, 1000, 50)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

plt.legend()

#%%
norm_data_file = "norm_low_photon_7feb25_alt_480photons_0.2g.pickle"
tr_data_file = "tr_low_photon_7feb25_alt_480photons_0.3g.pickle"

with open(norm_data_file, 'rb') as handle:
    loaded_data_norm = pickle.load(handle)
#rms_plot_norm = np.mean(loaded_data_norm["coeffs_in_KL_space"][50:,:]**2, axis=0)
rms_plot_norm = loaded_data_norm["rms_plot"]

with open(tr_data_file, 'rb') as handle:
    loaded_data_tr = pickle.load(handle)
#rms_plot_tr = np.mean(loaded_data_tr["coeffs_in_KL_space"][50:,:]**2, axis=0)
rms_plot_tr = loaded_data_tr["rms_plot"]
cutoff = 50
plt.plot(rms_plot_norm*1000, label=f"Normal {np.mean(rms_plot_norm[cutoff:])}")
plt.plot(rms_plot_tr*1000, label=f"TR {np.mean(rms_plot_tr[cutoff:])}")
plt.xlabel("Loop cycles")
plt.ylabel("RMS (nm)")
plt.title(f"Wavefront RMS in closed loop \n {wfs.total_photon_flux*48} photons. 5 KL Modes or turbulence and correction, 0.3 loop gain")
plt.legend()
#%%
wfc.flatten()
loop.resetCurrentCorrection()


#%%
loop.frame_weights = np.ones(loop.frame_weights.shape)
loop.makeIM(loop.push_cube,
                    loop.pull_cube,  
                    loop.ref_slopes,
                    loop.pokeAmp,
                    loop.frame_weights)

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

atm.numActiveModes = 30

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
for i in range(1):
    modes_to_send = atm.getNextTurbAsModes() 
    print(np.max(np.abs(modes_to_send)))
    wfc.write(modes_to_send)
    
    time.sleep(1)

#%%
wfc.flatten()

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
wfs.activateRONoise = False
wfs.total_photon_flux = 200
#%%
fsm.currentPos = None
image = slope.readImage() #slope.signal2D.read() #slope.readImage()
signal_TR =  np.zeros((image.shape[0], image.shape[1], 48))
for s in range(fsm.numOfTRFrames):
    fsm.step()
    signal_TR[:, :,s] = slope.readImage()

plt.imshow(np.sum(signal_TR, axis=2))
plt.colorbar()





#%%

import numpy as np
from scipy.ndimage import zoom
def strehl_ratio_ref(psf, psf_diff, box_size):
    '''
    This function computes the strehl ratio (SR) from a given PSF and the diffraction reference PSF.
    Inputs:
    - psf: PSF for SR estimation [2d array]
    - psf_diff: Reference diffraction PSF [2d array]
    - box_size: Box size for SR estimation [pixels]
    Outputs:
    - sr: Strehl Ratio []
    '''
    # Find the max for PSF
    ix, iy = np.unravel_index(np.argmax(psf), psf.shape)
    # Box the PSF
    psf_box = psf[ix-box_size:ix+box_size, iy-box_size:iy+box_size]
    # Find the max for diffraction PSF
    ix, iy = np.unravel_index(np.argmax(psf_diff), psf_diff.shape)
    # Box the diff PSF
    psf_diff_box = psf_diff[ix-box_size:ix+box_size, iy-box_size:iy+box_size]
    # Normalize diffraction PSF box
    psf_diff_box = psf_diff_box / np.max(psf_diff_box)
    # Normalize PSF box
    psf_box_norm = psf_box / np.sum(psf_box) * np.sum(psf_diff_box)
    # Oversample to get true peak (using scipy.ndimage.zoom with bicubic interpolation)
    psf_box_norm_oversampled = zoom(psf_box_norm, 4, order=3)  # order=3 for bicubic
    # Calculate Strehl ratio
    sr = np.max(psf_box_norm_oversampled)
    # Handle invalid SR values
    if not np.isfinite(sr) or sr == 0:
        sr = np.nan
    return sr

#%%
st = strehl_ratio_ref(dmpsf.readLong(), dmpsf.model, 50)
print(st)



#%% Linearity curves

wfs.activateNoise = True
wfs.activateRONoise = False
wfs.total_photon_flux = 0

selected_KL = 0
amps = np.arange(-0.05, 0.05, 0.01)
amp_resp = np.zeros(amps.shape)

for i in range(len(amps)):
    print(f"{i}/{len(amps)}")
    fsm.stop()
    wfc.push(selected_KL, amps[i])
    time.sleep(0.5)
    slopes_TR = loop.getTRSlopes()

    if loop.FF_active:
        if loop.ref_signal_normed is not None:
            newCorrection = updateCorrectionTRFF(correction=np.zeros((loop.numModes)), 
                                            gCM=loop.gCM, 
                                            slopes_TR=slopes_TR.flatten(),
                                            ref_signal_normed = loop.ref_signal_normed)
        else:
            print("Error: ref signal never defined, skipping loop")
    else:
        if loop.ref_signal_per_mode_normed is not None:
            newCorrection = updateCorrectionTR(correction=np.zeros((loop.numModes)), 
                                            gCM=loop.gCM, 
                                            slopes_TR=slopes_TR,
                                            weights=loop.frame_weights,
                                            ref_signal_per_mode_normed = loop.ref_signal_per_mode_normed)
        else:
            print("Error: weighted ref signal never defined, skipping loop")

    amp_resp[i] = newCorrection[selected_KL]

wfc.flatten()
loop.resetCurrentCorrection()

plt.figure()
plt.plot(amps, -amp_resp)
plt.plot(amps, amps, "--")
plt.show

plt.figure()
plt.plot(amps, amps+amp_resp)
plt.plot(amps, [0]*len(amps), "--")
plt.show
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

time.sleep(1)
modpsf.stop()



#%%
slopeinfo = {}

slopeinfo["p1mask"] = slope.p1mask
slopeinfo["p2mask"] = slope.p2mask
slopeinfo["p3mask"] = slope.p3mask
slopeinfo["p4mask"] = slope.p4mask
slopeinfo["valid"] = slope.validSubAps


with open("slope_info.pickle", 'wb') as handle:
    pickle.dump(slopeinfo, handle)

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
# %%
