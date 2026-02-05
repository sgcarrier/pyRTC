#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *
from pyRTC.TimeResolvedLoopWithRemoteWFC import *
from pyRTC.SlopesProcess import *


#%%

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

#N = np.random.randint(3000,6000)
#%%

#%%
################## Setup MPD SPAD Camera ##############
confTRWFS = conf["trwfs"]
trwfs = MPDSPADCam(conf=confTRWFS)
#trwfs.stop()

#%%
trwfs.advancedMode(True)
#trwfs.setExposure(4166) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
#trwfs.setNIntegFrames(3) # maybe increase this?
trwfs.setExposure(1379) # 41666.66 ns = 24 Khz exact # 10ns steps in advanced mode
trwfs.setNIntegFrames(3) # maybe increase this?
trwfs.setNFrames(48)
trwfs.applySettingsToCamera() # Appy the settings to the camera previously put


#%%
trwfs.setNFrames(48)
trwfs.enable_sync_mode()
trwfs.applySettingsToCamera()
#%%
trwfs.start()

#%%
trwfs.takeDark()
trwfs.saveDark("res/sky_latest_dark.npy")

#%%
################## Setup Full Frame Signal ##############

# For TR PWFS
sig = TimeResolvedFullFrameProcessCustomArea(conf=conf)
sig.start()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/TimeResolvedFullFrameProcessCustomArea.py", config, 5641)
slopes.launch()
#plt.imshow(np.sum(sig.read(), axis=0))

#%%

conf["slopes"]["signalType"] = "slopes"
sig = SlopesProcess(conf=conf, force_name_camera="trwfs")
sig.start()

#%%

sig.plotPupils()

def overlayCalcPosWithPupilMask(pos, img):
    f, ax = plt.subplots()
    ax.imshow(img, cmap='gray', interpolation='nearest')
    for i in range(4):
        cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
        ax.add_artist(cir)
    plt.show()
locs = []
current_locs = sig.pupilLocs
for i in range(4):
    locs.append((current_locs[i][0], current_locs[i][1], sig.pupilRadius))
img =trwfs.data 
for i in range(50):
    img +=trwfs.data 
    time.sleep(0.01)

img = np.sum(img, axis=0)
overlayCalcPosWithPupilMask(locs, img)

#%%
################## Connect to remote WFC ##############

# Dont forget to start the DM on the other PC
confWFC = conf["wfc"]
remote_wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", confWFC, 3000, remoteProcess=True)
remote_wfc.host = "132.246.192.209"
remote_wfc.launch()
#%%
remote_wfc.run("flatten")

#%%
################## Setup loop ##############


#%%
loop = LoopWithRemoteWFS(conf, remote_wfc)


#%%
loop.numItersIM = 10
loop.hardwareDelay  = 0.01
loop.computeIM()
remote_wfc.run("flatten")
loop.hardwareDelay  = 0.001

#%%

for i in range(50):
    loop.leakyIntegrator()

#%%
remote_wfc.run("flatten")

#%%
loop.setGain(0.01)
loop.leakyGain = 0.01
loop.start()
#%%
loop.stop()
#%%
remote_wfc.run("flatten")

#%%

loop = TimeResolvedLoopWithRemoteWFC(conf, remote_wfc, settings_name="trloop")

#%%
loop.manual_cam_obj = trwfs
loop.left_limit = 19
loop.right_limit = 53
#%%
loop.numModes = 5
loop.pushPullRef_cube()

weighting_cube = loop.modWeightsFromPushPullRef(loop.push_cube,
                                                loop.pull_cube,
                                                loop.ref_slopes,
                                                loop.pokeAmp)

plt.imshow(weighting_cube)

loop.flatten()
#%%
plt.plot(weighting_cube[:,0] )
plt.plot(weighting_cube[:,1] )

#%%
time_in_quad = [0,0,0,0]
for i in range(48):
    to_test = [quad1[i], quad2[i], quad3[i] , quad4[i]]
    time_in_quad[np.argmax(to_test)] += 1

#%%
img = trwfs.data
for i  in range(1000):
    time.sleep(0.001)
    img += trwfs.data
img_cut = img[:,:,19:53]
quad2 = np.sum(img_cut[:,0:16, 0:17], axis=(1,2)) / np.sum(img_cut)
quad1 = np.sum(img_cut[:,16:, 0:17], axis=(1,2)) / np.sum(img_cut)
quad3 = np.sum(img_cut[:,0:16, 17:], axis=(1,2)) / np.sum(img_cut)
quad4 = np.sum(img_cut[:,16:, 17:], axis=(1,2)) / np.sum(img_cut)



plt.plot(quad1, label="1")
plt.plot(quad2, label="2")
plt.plot(quad3, label="3")
plt.plot(quad4, label="4")
plt.legend()
#%%

X_offset = np.sum((quad1+quad2) - (quad3+quad4))
Y_offset = np.sum((quad1+quad4) - (quad2+quad3))

print(f"Xoff={X_offset}, Yoff={Y_offset}")
#%%

loop.computeIM()
loop.flatten()

#%%
for i in range(10):
    loop.timeResolvedIntegratorWithLeak_C()
    #print(np.max(loop.currentCorrection))
#%%
loop.resetCurrentCorrection()
loop.flatten()
#%%
trwfs.record_data_direct(1000, "SPAD_FFW_CL_0g1_0l03_20mode_20nov2025")

#%%
loop.setGain(0.3)
loop.leakyGain = 0.02

#%%
loop.setGain(0.0001)
loop.leakyGain = 0.00
#%%
loop.resetCurrentCorrection()
loop.start()

#%%
loop.stop()

#%%
loop.flatten()

#%%

valid_aps = np.load("valid_aps_spad_25sept2025.npy")

#%%
loop.changeWeightsAndUpdate(np.ones_like(loop.frame_weights))

#%%
loop.switchToFF()

# %%
loop.switchToFFwithWeights()

#%%
newCorrection = np.ascontiguousarray((1-loop.leakyGain)*np.array(loop.remoteWFC.getProperty("currentCorrection")), dtype=np.float32)
start_time = time.time()
for i in range(1000):
    loop.TR_norm_correction_function(loop.numModes, loop.numFrames, loop.signalSize,
                                                 loop.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.latest_slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 loop.ref_signal_per_mode_normed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 newCorrection.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
stop_time = time.time()

print(f"{1/((stop_time-start_time)/1000)} Hz")


#%%
start_time = time.time()
for i in range(1000):
    loop.getTRSlopes_bypass()
stop_time = time.time()

print(f"{1/((stop_time-start_time)/1000)} Hz")

#%%
loop.flatten()

#%%


def expose_test():
    trwfs.cam.SnapPrepare()
    trwfs.cam.SnapAcquire()
    data = trwfs.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
    return data.shape[0]

def expose_test2():
    #if trwfs.cam.IsTriggered() or (not trwfs.inSyncMode) :
    t5 = trwfs.cam.ContAcqToMemoryGetBuffer()
    f5 = trwfs.cam.BufferToFrames(t5, trwfs.cam.num_pixels, trwfs.cam.num_counters)[0]
        
    frames = f5.shape[0]
    #print(f5.dtype)
    return frames

#%%
def run_test():
    total_frames = 0
    start_time = time.perf_counter()
    trwfs.cam.ContAcqToMemoryStart()
    for i in range(10000):
        total_frames += expose_test2()
    stop_time = time.perf_counter()
    trwfs.cam.ContAcqToMemoryStop()
    print(f"{1/((stop_time-start_time)/10000)} Hz")
    print(f"{total_frames/((stop_time-start_time))} fps")

#%%

run_test()



#%%

def expose_test():


    # TODO do we want to use other counters?
    data = np.empty((0,), dtype=np.uint8)
    while data.size < (trwfs.nFrames+trwfs._offset)*32*64:
        data = np.concatenate((data, trwfs.cam.ContAcqToMemoryGetBuffer()))
    #self.data = self.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
    #self.frames = self.cam.ContAcqToMemoryGetBuffer()
    #if self.frames.shape[0] != self.nFrames: # Sometimes the snap returns nothing, TODO check to use a flag check maybe?
    #    return
    data = data[0: trwfs.cam.num_counters * trwfs.cam.num_pixels * int(np.floor((data.size / (trwfs.cam.num_counters * trwfs.cam.num_pixels))))]

    all_frames = (trwfs.cam.BufferToFrames(data, trwfs.cam.num_pixels, trwfs.cam.num_counters)[0])
    trwfs.frames = all_frames[trwfs._offset:trwfs._offset+48,:,:]
    trwfs.offset = trwfs.nFrames - ((all_frames.shape[0]-trwfs._offset) % trwfs.nFrames)
    
    #self.data = (self.cam.BufferToFrames(buf, self.cam.num_pixels, self.cam.num_counters)[0])[:48,:,:]
    
    # for idx in trwfs.screamers:
    #     trwfs.frames[:,idx[0], idx[1]] = 0





#%%

current_shape = remote_wfc.getProperty('currentShape')
np.max(np.abs(current_shape))
# %%
