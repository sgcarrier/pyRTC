#%%
import argparse
import sys
import os 

import wavekit_py as wkpy

from pyRTC.utils import *
import matplotlib.pyplot as plt
import ctypes
import time
#%%
conf = read_yaml_file("conf.yaml")
confSHWFS  = conf[  "shwfs"]
confDM     = conf[    "wfc"]

#%%
try :
    camera = wkpy.Camera(config_file_path = confSHWFS["confFile"])
    camera.connect()
    camera.start(0, 1)
except Exception as e :
    print(str(e))

#%% 
# Load ALPAO DM and flatten
from pyRTC.hardware.ALPAODM import *

wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()

#%%

def getWF(camera, conf):
    img = camera.snap_raw_image()

    try:
        slopes = wkpy.HasoSlopes(image= img, config_file_path = conf["confFile"])
    except Exception as e:
        print(str(e))

    filter = [False, False, False, False, False]
    filter_array = (ctypes.c_byte * 5)()
    for i,f in enumerate(filter):
        filter_array[i] = ctypes.c_byte(0) if f else ctypes.c_byte(1)

    phase = wkpy.Phase(hasoslopes = slopes, type_ = wkpy.E_COMPUTEPHASESET.ZONAL, filter_ = filter_array)
    WF = phase.get_data()
    rms, pv, max, min = phase.get_statistics()

    return WF[0], rms, pv



#%%
wfc.flatten()
time.sleep(0.1)
wf_flat, rms, pv = getWF(camera, confSHWFS)

#mask = ~np.isnan(wf)

#%%
plt.imshow(wf_flat)
plt.colorbar()



#%%
numModes = wfc.numModes
pokeAmp = 0.1
numItersIM = 5

#Reset the correction
#Plus amplitude
wfc.flatten()
#Post a new shape to be made
#Add some delay to ensure one-to-one
time.sleep(1)
#Burn the first new image since we were moving the DM during the exposure
wf, _, _ = getWF(camera, confSHWFS)
#Average out N new WFS frames
tmp_plus = np.zeros_like(wf)
for n in range(numItersIM):
    wf, _, _ = getWF(camera, confSHWFS)
    tmp_plus += wf
tmp_plus /= numItersIM

#Minus amplitude

#Post a new shape to be made
#Add some delay to ensure one-to-one
time.sleep(0.1)
#Burn the first new image since we were moving the DM during the exposure
wf, _, _ = getWF(camera, confSHWFS)
#Average out N new WFS frames
tmp_minus = np.zeros_like(wf)
for n in range(numItersIM):
    wf, _, _ = getWF(camera, confSHWFS)
    tmp_minus += wf
tmp_minus /= numItersIM

#Compute the normalized difference
flat= (tmp_plus-tmp_minus)



IM = np.zeros((numModes+1, wf_flat.shape[0], wf_flat.shape[1]))

IM[0,:,:] = flat

def pushPullIM():
    
    #For each mode
    for i in range(numModes):
        #Reset the correction
        #Plus amplitude
        wfc.push(i, pokeAmp)
        #Post a new shape to be made
        #Add some delay to ensure one-to-one
        time.sleep(0.1)
        #Burn the first new image since we were moving the DM during the exposure
        wf, _, _ = getWF(camera, confSHWFS)
        #Average out N new WFS frames
        tmp_plus = np.zeros_like(wf)
        for n in range(numItersIM):
            wf, _, _ = getWF(camera, confSHWFS)
            tmp_plus += wf
        tmp_plus /= numItersIM

        #Minus amplitude
        wfc.push(i, -pokeAmp)
        #Post a new shape to be made
        #Add some delay to ensure one-to-one
        time.sleep(0.1)
        #Burn the first new image since we were moving the DM during the exposure
        wf, _, _ = getWF(camera, confSHWFS)
        #Average out N new WFS frames
        tmp_minus = np.zeros_like(wf)
        for n in range(numItersIM):
            wf, _, _ = getWF(camera, confSHWFS)
            tmp_minus += wf
        tmp_minus /= numItersIM

        #Compute the normalized difference
        IM[i+1,:,:] = (tmp_plus-tmp_minus)/(2*pokeAmp)

    return



#%%
plt.imshow(flat)
plt.colorbar()
#%%
pushPullIM()
wfc.flatten()

#%%
plt.imshow(IM[60,:,:])
plt.colorbar()
#%%
for i in range(10):
    mode_img = IM[i,:,:]
    avg = np.mean(mode_img[~np.isnan(mode_img)])
    rms= np.sqrt(np.mean(np.square(mode_img[~np.isnan(mode_img)]-avg))) 
    print(rms)
#%%
np.save("poke_influence_97_20june_with_floating.npy", IM)


#%%
data = np.load("IM_haso_new_modal_5_accurate.npy")
#Save fits file
from astropy.io import fits as pyfits
hdu = pyfits.PrimaryHDU(data)
hdu.writeto("poke_influence_97_19juin.fits", overwrite=True)


#%%
data = np.load("testing.npy")
#%%
plt.imshow(data[1,:,:])
plt.colorbar()
# %%
for i in range(10):
    mode_img = data[i,:,:]
    avg = np.mean(mode_img[~np.isnan(mode_img)])
    rms= np.sqrt(np.mean(np.square(mode_img[~np.isnan(mode_img)]-avg))) 
    print(rms)
#%%
if camera is not None:
    camera.stop()

    camera.disconnect()

# %% [markdown]
# 