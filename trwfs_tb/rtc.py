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
# modpsf = PGScienceCam(conf=confMODPSF)
# modpsf.start()

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
pos = {"A": 3, "B": 3}
fsm.goTo(pos)
time.sleep(1)
img_q1 = wfs.read().astype(np.float64)
time.sleep(1)
fsm.resetPos()
time.sleep(1)
pos = {"A": 7, "B": 3}
fsm.goTo(pos)
time.sleep(1)
img_q2 = wfs.read().astype(np.float64)
time.sleep(1)
fsm.resetPos()
time.sleep(1)
pos = {"A": 7, "B": 7}
fsm.goTo(pos)
time.sleep(1)
img_q3 = wfs.read().astype(np.float64)
time.sleep(1)
fsm.resetPos()
time.sleep(1)
pos = {"A": 3, "B": 7}
fsm.goTo(pos)
time.sleep(1)
img_q4 = wfs.read().astype(np.float64)
time.sleep(1)
fsm.resetPos()

#%%
img_high_mod = img_q1 + img_q2 + img_q3 + img_q4
img_high_mod_bin = np.zeros(img_high_mod.shape)
img_high_mod_bin[img_high_mod>(np.max(img_high_mod)*0.02)] = 1

#%%
pos_ret = findAllPupils2(img_high_mod_bin, quadrant_size=16)


#%%
f, ax = plt.subplots()
pos_ret = [(5, 5, 5), (5, 26, 5), (26, 5, 5), (27, 26, 5)]
ax.imshow(img_high_mod, cmap='gray', interpolation='nearest')
for i in range(4):
    cir = plt.Circle((pos_ret[i][0], pos_ret[i][1]), pos_ret[i][2], color='red', fill=False)
    ax.add_artist(cir)

displayOffset(img_high_mod_bin, pos_ret)
plt.show()
#%%
def displayOffsetnoshow(params):
    p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = params
    pupilLocs = [(np.round(p0_x), np.round(p0_y), radius_pupil), (np.round(p1_x), np.round(p1_y), radius_pupil), (np.round(p2_x), np.round(p2_y), radius_pupil), (np.round(p3_x), np.round(p3_y), radius_pupil)]
    pupils = []
    pupilMask = np.zeros(img.shape)
    xx,yy = np.meshgrid(np.arange(pupilMask.shape[0]),np.arange(pupilMask.shape[1]))
    for i, pupil_loc in enumerate(pupilLocs):
        px, py, r = pupil_loc
        zz = np.sqrt((xx-px)**2 + (yy-py)**2)
        pupils.append(zz < r)
        pupilMask += pupils[-1]*(i+1)
    p1mask = pupilMask == 1
    p2mask = pupilMask == 2
    p3mask = pupilMask == 3
    p4mask = pupilMask == 4

    xx,yy = np.meshgrid(np.arange(2*r),np.arange(2*r))
    zz = np.sqrt((xx-r)**2 + (yy-r)**2)
    p1image = np.zeros((2*r,2*r))
    p2image = np.zeros((2*r,2*r))
    p3image = np.zeros((2*r,2*r))
    p4image = np.zeros((2*r,2*r))
    p1image[zz<r] = img_bin[p1mask]
    p2image[zz<r] = img_bin[p2mask]
    p3image[zz<r] = img_bin[p3mask]
    p4image[zz<r] = img_bin[p4mask]
    #x_slopes = (p1 + p2) - (p3 + p4)
    #y_slopes = (p1 + p3) - (p2 + p4)
    x_sum = np.sum(np.abs(((p1image + p2image) - (p3image + p4image))))
    y_sum = np.sum(np.abs(((p1image + p3image) - (p2image + p4image))))
    return x_sum+ y_sum

#%%
img = img_high_mod
img_bin = img_high_mod_bin
radius_pupil = 5
pos_init  = [4, 5, 5, 27, 26, 5, 26, 26]
for pos in range(8):
    for i in [-1, 0, 1]:
        cur_pos = pos_init
        cur_pos[pos] += i
        ret_val= displayOffsetnoshow(tuple(cur_pos))
        print(f"val={ret_val}, with pos = {cur_pos}")
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

def findPupilPosAndRadius_forced_radius(img, size, r_forced=5):
    # img_bin = np.zeros(img.shape) 
    # img_bin[img>(np.max(img)*0.1)] = 1
    img_bin = img
    image = img_bin[:size,:size]
    regions = measure.regionprops(measure.label(image))
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = r_forced

    def cost(params):
        x0, y0 = params
        coords = draw.disk((np.round(y0), np.round(x0)), r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0 = optimize.fmin(cost, (x0, y0))
    print(( x0, y0, r_forced))
    return int(np.round(x0)), int(np.round(y0)), int(r_forced)

def findAllPupils2(img, quadrant_size):
    pos = []
    x0, y0, r0 = findPupilPosAndRadius_forced_radius(img[:quadrant_size,:quadrant_size], quadrant_size)
    pos.append((x0, y0, r0))
    x1, y1, r1 = findPupilPosAndRadius_forced_radius(img[quadrant_size:,:quadrant_size], quadrant_size)
    y1 += quadrant_size
    pos.append((x1, y1, r1))
    x2, y2, r2 = findPupilPosAndRadius_forced_radius(img[:quadrant_size,quadrant_size:], quadrant_size)
    x2 += quadrant_size
    pos.append((x2, y2, r2))
    x3, y3, r3 = findPupilPosAndRadius_forced_radius(img[quadrant_size:,quadrant_size:], quadrant_size)
    x3 += quadrant_size
    y3 += quadrant_size
    pos.append((x3, y3, r3))

    return pos
#%%
def autoFindAndDisplayPupils2(wfs, quadrant_size):

    numImages = 20
    img = wfs.read().astype(np.float64)
    for i in range(numImages-1):
        img += wfs.read().astype(np.float64)
    img /= numImages

    img_bin = np.zeros(img.shape)
    img_bin[img>(np.max(img)*0.04)] = 1
    plt.imshow(img_bin)
    plt.colorbar()
    pos = findAllPupils2(img_bin, quadrant_size)
    print(pos)

    displayOffset(img_bin, pos)

    print(pos)
    f, ax = plt.subplots()
    ax.imshow(img_bin, cmap='gray', interpolation='nearest')
    for i in range(4):
        cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
        ax.add_artist(cir)
    plt.show()

    return pos


pupil_pos = autoFindAndDisplayPupils2(wfs, quadrant_size=16)

#%%
numImages = 20
img = wfs.read().astype(np.float64)
for i in range(numImages-1):
    img += wfs.read().astype(np.float64)
img /= numImages

img_bin = np.zeros(img.shape)
img_bin[img>(np.max(img)*0.05)] = 1


#%%
import scipy
quads = np.zeros((4, 64, 64))
quads[0,:,:] = img_bin[0:64,0:64]
quads[1,:,:] = img_bin[64:, 0:64]
quads[2,:,:] = img_bin[0:64, 64:]
quads[3,:,:] = img_bin[64:, 64:]

avg_offset = np.zeros((4, 4, 2))
for q in range(4):
    for i in range(4):
        co = scipy.signal.convolve(quads[q,:,:], quads[i,:,:], mode="same", method="direct")
        center = np.unravel_index(co.argmax(axis=None), co.shape)
        avg_offset[q,i,:] += [center[0], center[1]]

plt.imshow(co)
print(avg_offset)

#TODO prob need to substract 64/2 from offsets found
#%%

#  - 11,11 #22,20 # 21,22
#  - 53,11 #107,20 #  106,22
#  - 10,53 #22,105 # 21,107
#  - 53,54 #107,106 # 106,107
pos[0] = (22, 22, 17)
pos[2] = (23, 106, 17)
pos[1] = (107, 21, 17)
pos[3] = (108, 106, 17)


f, ax = plt.subplots()
ax.imshow(img, cmap='gray', interpolation='nearest')
for i in range(4):
    cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
    ax.add_artist(cir)
plt.show()
displayOffset(img, pos)

#%%
for i in range(48):
    fsm.step()
    time.sleep(1)

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
#%%
loop.computeIM()
#loop.calcFrameWeights()

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
saved_cc = wfc.currentCorrection
print(saved_cc)
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
wfs.activateRONoise = False
loop.turbulenceGenerator = None
time.sleep(1)
for i in range(5):
    #loop.standardIntegratorWithTurbulence()
    loop.timeResolvedIntegratorWithTurbulence()
    time.sleep(0.1)

#%%
wfc.flatten()
loop.resetCurrentCorrection()


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

    time.sleep(0.5)

    wfc.push(i, -poke_amp)
    #Add some delay to ensure one-to-one
    time.sleep(0.1)
    #Burn the first new image since we were moving the DM during the exposure
    pull_coeff = grabHASOCoeffs(camera, confSHWFS, NUM_MODES)
    #pull_coeff -= REF_coeff


    #Compute the normalized difference
    HASO_resp[i,:] = (push_coeff-pull_coeff)/(2*poke_amp)
    time.sleep(0.5)

#%%
plt.imshow(HASO_resp, cmap = 'inferno', aspect='auto')
plt.colorbar()
plt.show()
#%%
wfc.flatten()
loop.resetCurrentCorrection()


#%%
modal_gain = np.ones(68) *0.2
modal_gain[0] = 0.3
modal_gain[1] = 0.3
loop.gain = modal_gain
loop.gCM = loop.gain[:, np.newaxis]*loop.CM

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

anim = animation.FuncAnimation(fig, animate, frames= 100, interval=1000/10)

anim.save("test_anime3.gif", fps=10)

#%% Animate with atmo
import matplotlib.animation as animation



fig = plt.figure()
img = plt.imshow(REF_WF)
ann = plt.annotate(str(0), (0,0))
plt.colorbar(img)
turb_rms = np.zeros(100)
def animate(i):
    t = atm.atm[i,:]
    loop.wfcShm.write(t)
    read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
    read_WF_valid = read_WF[~np.isnan(read_WF)] 
    rms_val = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
    img.set_data(read_WF)
    img.set_clim(np.min(read_WF_valid), np.max(read_WF_valid))
    ann.set_text(f"f={i},rms={int(rms_val*1000)}nm")
    turb_rms[i]= rms_val

anim = animation.FuncAnimation(fig, animate, frames= 10*10, interval=1000/10)
wfc.flatten()
loop.resetCurrentCorrection()
anim.save("atmo.gif", fps=10)
#%%
atm.setSpeed(1)
loop.setTurbulenceGenerator(atm)


#%%
wfs.activateNoise = True
wfs.activateRONoise = False
wfs.total_photon_flux = 100
#%%
fsm.stop()
fsm.currentPos = None
iterations = 70
loop.setGain(0.5)
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
    try:
        #loop.turbulenceGenerator.currentPos +=10
        loop.timeResolvedIntegratorWithTurbulence()
        #loop.standardIntegratorWithTurbulence()
        time.sleep(0.1)
        strehls[i] = dmpsf.strehl_ratio
        read_WF = ((REF_WF - grabHASOImage(camera, confSHWFS)))
        read_WF_valid = read_WF[~np.isnan(read_WF)] 
        saved_WFs[i,:,:] = read_WF
        rms_plot[i] = np.sqrt(np.mean(np.square(read_WF_valid - np.mean(read_WF_valid))))
        CUR_coeff = (REF_coeff - grabHASOCoeffs(camera, confSHWFS, NUM_MODES))
        saved_coeffs[i,:] = CUR_coeff
        current_wfc_shape[i,wfc.layout] = wfc.currentShape
        print(np.max(np.abs(wfc.currentShape)))
        print(f"it = {i}, RMS={rms_plot[i]:.4f}, SR={strehls[i]:.4f}")
        #img_slopes.append(loop.latest_slopes)
        turb_modes[i,:] = loop.turbModes
        corr_modes[i,:] = loop.latest_correction
    except KeyboardInterrupt:
        fsm.stop()
        print("Stopped loop")
        break


#%%
wfc.flatten()
loop.resetCurrentCorrection()

#%%
mode_chosen = 0

fig,ax=plt.subplots()
plt.plot(turb_modes[:,mode_chosen], "r-", label=f"turb")
plt.plot(-corr_modes[:,mode_chosen], "k-", label=f"corr")

# And a corresponding grid
ax.grid(which='both')

plt.legend()


#%%


fig,ax=plt.subplots()
for mode_chosen in range(30):
    diff = turb_modes[:,mode_chosen] +  corr_modes[:,mode_chosen]
    plt.plot(diff, label=f"{mode_chosen}")

# And a corresponding grid
ax.grid(which='both')
fig.suptitle(f'Residual coeffs for each mode per iteration', fontsize=16)
fig.legend()

#%%

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(rms_plot*1000)
ax1.set_title("WFE RMS (nm)")

ax2.plot(strehls)
ax2.set_title("Strehl")

fig.suptitle(f'Closed-loop performance', fontsize=16)

#%%

fig, (ax1,ax2) = plt.subplots(2,1)

#ax1.plot(rms_plot_norm_10*1000, label="Modulated - 0.5g")
#ax1.plot(rms_plot_norm_500_03g*1000, label="Modulated - 0.3g")
ax1.plot(rms_plot_tr_100_05g*1000, label="TR - 0.5g")
#ax1.plot(rms_plot_FF_10*1000, label="TR-Cube - 0.5g")
ax1.plot(rms_plot_ff_100_05g*1000, label="TR-Cube - 0.5g")
ax1.set_title("WFE RMS (nm)")

#ax2.plot(strehl_norm_10, label="Modulated - 0.5g")
#ax2.plot(strehl_norm_500_03g, label="Modulated - 0.3g")
ax2.plot(strehl_tr_100_05g, label="TR - 0.5g")
#ax2.plot(strehl_FF_10, label="TR-Cube - 0.5g")
ax2.plot(strehl_ff_100_05g, label="TR-Cube - 0.5g")
ax2.set_title("Strehl")

ax1.legend()

ax1.axvline(x=30, color='red', linestyle='--')
ax2.axvline(x=30, color='red', linestyle='--')

fig.suptitle(f'Closed-loop performance: {wfs.total_photon_flux*48} photons, gain={loop.gain}', fontsize=16)

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