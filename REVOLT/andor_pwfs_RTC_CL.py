#%%
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *

from pyRTC.LoopWithRemoteWFC import *

#%%

config = '../REVOLT/SPAD_PC_config.yaml'
conf = read_yaml_file(config)

#%%

################## Setup Andor Camera ##############
confWFS = conf["wfs"]
wfs = AndorIXon(conf=confWFS)
wfs.setExposure(0.001987) #500 Hz
wfs.open_shutter()
#wfs.start()

#%%
wfs.cam.get_EMCCD_gain()

#%%
wfs.cam.set_EMCCD_gain(300) # 150 for the lab

#%%
# TODO test turning on the Andor cooler
wfs.cam.set_cooler(on=True)
wfs.cam.set_temperature(-10)

#%%
print(wfs.cam.get_temperature_setpoint())
print(wfs.cam.get_temperature())


#%%
wfs.start()


#%% Close all

wfs.stop()
time.sleep(1)
wfs.close_shutter()
time.sleep(1)
wfs.close_camera()
time.sleep(1)

#%%
################## Setup Full Frame Signal ##############

# For normal PWFS
sig = FullFrameProcess(conf=conf)
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
img = sig.readImage()
overlayCalcPosWithPupilMask(locs, img)

#%%
################## Setup WFC ##############
confWFC = conf["wfc"]
remote_wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", confWFC, 3000, remoteProcess=True)
remote_wfc.host = "132.246.192.209"
remote_wfc.launch()
#%%
#a =remote_wfc.run("read")
a = remote_wfc.getProperty("currentCorrection")


#%%

remote_wfc.run("push", 98, 0.01)

#%%
remote_wfc.run("flatten")

#%%
#%%

newCorrection = np.zeros(99)
newCorrection[0] = 0.01
#newCorrection[1] = 0.0
remote_wfc.run("write", newCorrection*100000)

#%%
################## Setup loop ##############

remote_wfc.run("saveShape", "res/best_flat_4.npy")

#%%
loop = LoopWithRemoteWFS(conf, remote_wfc)


#%%
loop.computeIM()

#%%
for i in range(100):
    loop.leakyIntegrator()

#%%
loop.flatten()

#%%
current_flat = remote_wfc.getProperty('currentShape')


#%%


#%%

def save_images_to_fits( data, filename, headers=None, overwrite=True):
    """
    Save AxNxWxH image array to a FITS file.

    Parameters:
    -----------
    data_cube : numpy.ndarray
        Array with shape (A, N, W, H) where:
        A = number of acquisitions
        W, H = image dimensions
    filename : str
        Output FITS filename
    headers : dict, optional
        Header dictionary to add to FITS file
    overwrite : bool
        Whether to overwrite existing file
    """

    # Create primary HDU
    primary_hdu = fits.PrimaryHDU(data)

    # Add headers if provided
    if headers:
        for key, value in headers.items():
            primary_hdu.header[key] = value

    # Create HDU list and write
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(filename, overwrite=overwrite)
    hdul.close()


#%%
num_acq=1000
data = np.zeros((num_acq, 128,128))
for i in range(num_acq):
    data[i,:,:] = wfs.read()



#%%

save_images_to_fits(data, "andor_pwfs_ol_sky_gain300_0g00001_10h31_24sept2025.fits")







# %%
