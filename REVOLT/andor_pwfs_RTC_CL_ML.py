#%%
# Imports
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
from pyRTC.hardware.AndorIXonCam import *
from pyRTC.hardware.MPDSPADCam import *
from pyRTC.FullFrameProcess import *
from pyRTC.FullFrameProcessCustomArea import *
from pyRTC.TimeResolvedFullFrameProcessCustomArea import *
from pyRTC.SlopesProcess import *
from pyRTC.LoopWithRemoteWFC import *

#%%
# Config 
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
wfs.cam.set_EMCCD_gain(150) # 150 for the lab

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

# For full frame PWFS
#sig = FullFrameProcess(conf=conf)
#sig.start()

conf["slopes"]["signalType"] = "slopes"
sig = SlopesProcess(conf=conf)
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

remote_wfc.run("push", 1, 0.01)

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

remote_wfc.run("saveShape", "res/best_flat_slopes_08oct2025_lab.npy")

#%%
loop = LoopWithRemoteWFS(conf, remote_wfc)


#%%
#Compute IM
loop.computeIM()

#%%

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
save_images_to_fits(data, "andor_flat_slopes_25sept2025_2.fits")

#%%
remote_wfc.run("saveShape", "res/spad_slopes_19nov2025_2.npy")


#%%
#Initiate slowdata
slowdata= np.zeros((num_acq, 128,128))
slowdatacount = 0

# %%
#slow loop & collect Data
#Don't include integration in config for "loop" object, using Simon's integrator for now 
#the way this is set up for now -> should have atm turb gen in integrator so it moves by one step when this is run 

loop.leakyIntegrator()
slowdata[i,:,:]=wfs.read()
slowdatacount += 1

#%%
#save slow data
slowdata_filename = "mlData/slowdata_date_time.fits"
save_images_to_fits(slowdata, slowdata_filename)


# %%
#Delayed loop & collect Data
#Don't include integration in config for "loop" object, using Simon's integrator for now 
#the way this is set up for now -> should have atm turb gen in integrator so it moves by one step for each for loop iteration 

delaydata= np.zeros((num_acq, 128,128))
num_acq=1000

for i in range(num_acq):
    #introduce delay as needed: 
    #time.sleep(1)
    loop.leakyIntegrator()
    delaydata[i,:,:] = wfs.read()


delaydata_filename = "mlData/delaydata_date_time.fits"
save_images_to_fits(delaydata, delaydata_filename)

