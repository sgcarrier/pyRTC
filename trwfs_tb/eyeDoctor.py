# %%
import aotools
import aotools.functions.zernike
import matplotlib.pyplot as plt
import numpy as np


import matplotlib 
matplotlib.use('TkAgg')
# %%
# First we generate the modal basis we will use
numOfModes = 97 #Number of modes
res = 11 # Resolution of the pupil. Here I use the number of actuators accross the diameter
bases,_,_,_ = aotools.functions.karhunenLoeve.make_kl(numOfModes, res, stf="kolmogorov")

max_basis_value = 1
bases = bases / np.max(np.abs(bases)) * max_basis_value
#for i in range(bases.shape[0]):
#    bases[i,:,:] = bases[i,:,:] / np.max(np.abs(bases[i,:,:])) * max_basis_value


mask = np.abs(bases[0,:,:]) > 0
print(mask)

M2C = np.zeros((np.sum(mask), numOfModes))
for m in range(numOfModes):
    M2C[:,m] = bases[m,mask]


#np.save("M2C_KL_97.npy", M2C)



#plt.figure()
#plt.imshow(bases[3,:,:])
#plt.colorbar()
#plt.show(block=True)
#print(np.sum(np.abs(bases[0,:,:]) > 0))

# %%
# initialize the DM
from pyRTC.hardware.ALPAODM import *

conf = read_yaml_file("conf.yaml")

confWFC = conf["wfc"]
wfc = ALPAODM(conf=confWFC)
nActuators = wfc.dm.Get('NBOfActuator')
print(f"DM Reset on close is {wfc.dm.Get('LastCommand')}")

wfc.start()

#wfc.setM2C(M2C)

wfc.flatten()


# %%

# Initialize the PSF camera 
from hardware.PGScienceCam import *
confDMPSF = conf["dmpsf"]
psf = PGScienceCam(conf=confDMPSF)
psf.start()
psf.setExposure(200) # in us



stepsPerMode = 11
modeSteps = np.linspace(-1, 1, stepsPerMode) * 0.001


# TODO change this to use the listening services instead of calling them directly

offset = np.zeros(97)
wfc.start()

wfc.setM2C(M2C)
wfc.flatten()


print(wfc.read())
numImagesToStack = 5
imStack = np.zeros((psf.imageShape[0], psf.imageShape[1], numImagesToStack))
maxInt = np.zeros(stepsPerMode)
maxModeToSharpen = 25
numPixelsToSharpen = 8
# %%
plt.ion()
fig = plt.figure()
psf.expose()
im = psf.read()
img = plt.imshow(im, cmap='hot')
img.autoscale()
cbar = plt.colorbar(img)
plt.show()


for mode in range(2, maxModeToSharpen):
    for s in range(stepsPerMode):
        wfc.flatten()
        #wfc.write(offset)
        # Apply to DM
        to_apply = offset
        to_apply[mode] =  modeSteps[s]
        print(f"to apply = {to_apply}")
        #wfc.push(mode, modeSteps[s])
        wfc.write(to_apply)

        print(f"Max of current shape:{np.max(np.abs(wfc.currentShape))}")
        time.sleep(0.1)
        # Grab PSF frames 
        for i in range(numImagesToStack):
            psf.expose()
            im = psf.read()
            imStack[:,:, i] = im
        ima = np.mean(imStack, axis=2)
        img.set_data(ima)
        img.set_clim([np.min(ima), np.max(ima)])
        cbar.update_normal(img)
        #img.autoscale()
        plt.pause(0.01)
        #plt.show()
        plt.draw()
        #plt.show(block=False)
        ima_flat = ima.flatten()
        idx = np.argsort(-ima_flat)
        sorted_values = ima_flat[idx]
        maxInt[s] = np.sum(sorted_values[:numPixelsToSharpen])

        # Average frame stack
    time.sleep(0.5)

    # determine best coeff
    idx = np.argmax(maxInt)
    # Add to the offset 
    offset[mode] = modeSteps[idx]
    
print(offset)
wfc.write(offset)
time.sleep(1)
plt.close()
plt.figure()
#plt.set_aspect('equal')

psf.expose()
im = psf.read()

im_final = plt.imshow(np.log10(im+1e-9), aspect='equal')
fig.tight_layout()
plt.colorbar(im_final)

print(f"Final Shape: {wfc.currentShape}")

np.save("new_flat.npy", wfc.currentShape)

plt.show(block=True)
time.sleep(1)
wfc.flatten()
wfc.stop()

psf.stop()
# %%
