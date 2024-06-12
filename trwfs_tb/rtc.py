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
# Find center of pupils
import scipy.ndimage as ndi
img  = wfs.read()
img_bin = np.zeros(img.shape) 
img_bin[img>(np.max(img)*0.2)] = 1

img0 = img_bin[0:64,0:64]
cy0, cx0 = ndi.center_of_mass(img0)

img1 = img_bin[64:,0:64]
cy1, cx1 = ndi.center_of_mass(img1)
cy1 += 65

img2 = img_bin[0:64,64:]
cy2, cx2 = ndi.center_of_mass(img2)
cx2 += 65

img3 = img_bin[64:,64:]
cy3, cx3 = ndi.center_of_mass(img3)
cy3 += 65
cx3 += 65

cy0 = int(cy0)
cy1 = int(cy1)
cy2 = int(cy2)
cy3 = int(cy3)
cx0 = int(cx0)
cx1 = int(cx1)
cx2 = int(cx2)
cx3 = int(cx3)
centers = [(int(cy0), cx0),
           (cy1, cx1),
           (cy2, cx2),
           (cy3, cx3)]
print(centers)
avg_diameter = np.mean([ np.sum(img_bin[cy0, :])/2,
        np.sum(img_bin[cy1, :])/2,
        np.sum(img_bin[cy2, :])/2,
        np.sum(img_bin[cy3, :])/2,
        np.sum(img_bin[:, cx0])/2,
        np.sum(img_bin[:, cx1])/2,
        np.sum(img_bin[:, cx2])/2,
        np.sum(img_bin[:, cx3])/2])
print(avg_diameter/2)

#%%

from skimage import io, color, measure, draw, img_as_bool
import numpy as np
from scipy import optimize

def findPupilPos(img):
    # img_bin = np.zeros(img.shape) 
    # img_bin[img>(np.max(img)*0.1)] = 1
    img_bin = img
    image = img_bin[:64,:64]
    regions = measure.regionprops(measure.label(image))
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length / 2.

    def cost(params):
        x0, y0, r = params
        coords = draw.disk((y0, x0), r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))
    print(( x0, y0, r))
    return int(np.round(x0)), int(np.round(y0)), int(np.round(r))

def findAllPupils(img):
    pos = []
    x0, y0, r0 = findPupilPos(img[:64,:64])
    pos.append((x0, y0, r0))
    x1, y1, r1 = findPupilPos(img[64:,:64])
    y1 += 64
    pos.append((x1, y1, r1))
    x2, y2, r2 = findPupilPos(img[:64,64:])
    x2 += 64
    pos.append((x2, y2, r2))
    x3, y3, r3 = findPupilPos(img[64:,64:])
    x3 += 64
    y3 += 64
    pos.append((x3, y3, r3))

    return pos

def displayOffset(img, pupilLocs):
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
    p1image[zz<r] = img[p1mask]
    p2image[zz<r] = img[p2mask]
    p3image[zz<r] = img[p3mask]
    p4image[zz<r] = img[p4mask]
    plt.figure()
    #x_slopes = (p1 + p2) - (p3 + p4)
    #y_slopes = (p1 + p3) - (p2 + p4)
    x_sum = np.sum(np.abs(((p1image + p2image) - (p3image + p4image))))
    plt.imshow((p1image + p2image) - (p3image + p4image))
    plt.figure()
    y_sum = np.sum(np.abs(((p1image + p3image) - (p2image + p4image))))
    plt.imshow((p1image + p3image) - (p2image + p4image))
    plt.show()
    print(f"xsum = {x_sum}, ysum={y_sum}")

    


numImages = 20
img = wfs.read().astype(np.float64)
for i in range(numImages-1):
    img += wfs.read().astype(np.float64)
img /= numImages

img_bin = np.zeros(img.shape)
img_bin[img>(np.max(img)*0.1)] = 1
pos = findAllPupils(img_bin)
print(pos)
# pos[0] = (pos[0][0], pos[0][1],pos[0][2]+1)
# pos[1] = (pos[1][0], pos[1][1],pos[1][2]+1)
# pos[2] = (pos[2][0], pos[2][1],pos[2][2]+1)
# pos[3] = (pos[3][0], pos[3][1],pos[3][2]+1)

# pos[0] = (pos[0][0]+1, pos[0][1],pos[0][2])
# pos[1] = (pos[1][0], pos[1][1],pos[1][2])
# pos[2] = (pos[2][0], pos[2][1]+1,pos[2][2])
# pos[3] = (pos[3][0]+1, pos[3][1]+1,pos[3][2])

# pos[0] = (pos[0][0], pos[0][1],pos[0][2])
# pos[1] = (pos[1][0], pos[1][1],pos[1][2])
# pos[2] = (pos[2][0], pos[2][1],pos[2][2])
# pos[3] = (pos[3][0], pos[3][1],pos[3][2])

displayOffset(img_bin, pos)

print(pos)
f, ax = plt.subplots()
ax.imshow(img_bin, cmap='gray', interpolation='nearest')
for i in range(4):
    cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
    ax.add_artist(cir)
plt.show()


#%%
def findEdgePixelIdx(dmRes):
    xx,yy = np.meshgrid(np.arange(dmRes),np.arange(dmRes))
    zz = np.sqrt((xx-(dmRes//2))**2 + (yy-(dmRes//2))**2)
    actMap = np.ones((dmRes,dmRes))*-1
    actMap[zz<=(dmRes/2)] = list(range(np.sum(zz<=(dmRes/2))))
    edgesID_all = actMap[zz>=(dmRes//2)]
    edgesIdx = (edgesID_all[edgesID_all != -1]).astype("int")
    return actMap, edgesIdx

print(findEdgePixelIdx(11))
_, edgeAct = findEdgePixelIdx(11)
#%% Calculate slopes
slope = SlopesProcess(conf=conf)
slope.start()

#%%
slope.plotPupils()

#%%
f, ax = plt.subplots()
ax.imshow(slope.pupilMask, cmap='gray', interpolation='nearest')
for i in range(4):
    cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
    ax.add_artist(cir)
plt.show()

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
IM = loop.IM
validAct = []
for i in range(97):
    if i not in edgeAct:
        validAct.append(int(i))
IM_sub = np.delete(IM, edgeAct, axis=1)
loop.IM[:,edgeAct] = 0
invIM = np.linalg.pinv(IM_sub, rcond=0)
loop.CM = np.zeros((loop.numModes, loop.signalSize),dtype=loop.signalDType)
loop.CM[validAct,:] = invIM
loop.gCM = loop.gain*loop.CM
loop.fIM = np.copy(loop.IM)

#%%
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
np.max(np.abs(wfc.currentShape))



# %%
def pushPullIM_cube(loop, fsm):
    IM_cube = np.zeros((loop.signalSize, 48, loop.numModes),dtype=loop.signalDType)

    ref_slopes = np.zeros((loop.signalSize, 48))
    fsm.currentPos = None

    # for s in range(len(fsm.points)):
    #     fsm.step()
    #     time.sleep(0.1)
    #     #Average out N new WFS frames
    #     tmp_plus =  np.zeros((loop.signalSize))
    #     for n in range(loop.numItersIM):
    #         tmp_plus += loop.wfsShm.read()
    #     tmp_plus /= loop.numItersIM

    #     #Compute the normalized difference
    #     ref_slopes[:,s] =tmp_plus


    #For each mode
    for i in range(10):
        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.1)
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(loop.IM[:,i])
            for n in range(loop.numItersIM):
                tmp_plus += loop.wfsShm.read()
            tmp_plus /= loop.numItersIM


            #plus_signal = tmp_plus/np.sum(tmp_plus) #- ref_slopes[:,s]/np.sum(ref_slopes[:,s])


        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.1)
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(loop.IM[:,i])
            for n in range(loop.numItersIM):
                tmp_minus += loop.wfsShm.read()
            tmp_minus /= loop.numItersIM


            # #Minus amplitude
            # correction[i] = -loop.pokeAmp
            # #Post a new shape to be made
            # loop.wfcShm.write(correction)
            # #Add some delay to ensure one-to-one
            # time.sleep(loop.hardwareDelay)
            # #Burn the first new image since we were moving the DM during the exposure
            # loop.wfsShm.read()
            # #Average out N new WFS frames
            # tmp_minus = np.zeros_like(loop.IM[:,i])
            # for n in range(loop.numItersIM):
            #     tmp_minus += loop.wfsShm.read()
            # tmp_minus /= loop.numItersIM

        #Compute the normalized difference
        IM_cube[:,s,i] = (tmp_plus-tmp_minus)/(2*loop.pokeAmp)

    return IM_cube

def pushPullIM_cube_2(loop, fsm):
    IM_cube = np.zeros((loop.signalSize, 48, loop.numModes),dtype=loop.signalDType)

    ref_slopes = np.zeros((loop.signalSize, 48))
    fsm.currentPos = None

    for s in range(len(fsm.points)):
        fsm.step()
        time.sleep(0.1)
        #Average out N new WFS frames
        ref_slopes[:,s] =  np.zeros((loop.signalSize))
        for n in range(loop.numItersIM):
            ref_slopes[:,s] += loop.wfsShm.read()
        ref_slopes[:,s] /= loop.numItersIM



    #For each mode
    for i in range(10):
        #Reset the correction
        correction = loop.flat.copy()
        #Plus amplitude
        correction[i] = loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        tmp_plus =  np.zeros((loop.signalSize, 48))
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.01)
            #Average out N new WFS frames
            for n in range(loop.numItersIM):
                tmp_plus[:,s] += loop.wfsShm.read()
            tmp_plus[:,s] /= loop.numItersIM


            tmp_plus[:,s] = tmp_plus[:,s] - ref_slopes[:,s]



        #minus amplitude
        correction[i] = -loop.pokeAmp
        #Post a new shape to be made
        loop.wfcShm.write(correction)
        #Add some delay to ensure one-to-one
        time.sleep(loop.hardwareDelay)
        #Burn the first new image since we were moving the DM during the exposure
        loop.wfsShm.read()

        fsm.currentPos = None
        tmp_minus =  np.zeros((loop.signalSize, 48))
        for s in range(len(fsm.points)):
            fsm.step()
            time.sleep(0.01)
            #Average out N new WFS frames
            for n in range(loop.numItersIM):
                tmp_minus[:,s] += loop.wfsShm.read()
            tmp_minus[:,s] /= loop.numItersIM

            tmp_minus[:,s] = tmp_minus[:,s] - ref_slopes[:,s]


            # #Minus amplitude
            # correction[i] = -loop.pokeAmp
            # #Post a new shape to be made
            # loop.wfcShm.write(correction)
            # #Add some delay to ensure one-to-one
            # time.sleep(loop.hardwareDelay)
            # #Burn the first new image since we were moving the DM during the exposure
            # loop.wfsShm.read()
            # #Average out N new WFS frames
            # tmp_minus = np.zeros_like(loop.IM[:,i])
            # for n in range(loop.numItersIM):
            #     tmp_minus += loop.wfsShm.read()
            # tmp_minus /= loop.numItersIM

        #Compute the normalized difference
        IM_cube[:,:,i] = (tmp_plus-tmp_minus)/(2*loop.pokeAmp)

    return IM_cube

#%%
# Try to get weighted cube

#slope.takeRefSlopes()
wfc.flatten()

fsm.stop()
fsm.currentPos = None
pos = {"A": 5.0, "B": 5.0}
fsm.goTo(pos)
time.sleep(1)
i_cube = pushPullIM_cube_2(loop, fsm)
#%%
weighting_cube = np.zeros((48, 10))
for i in range(10):
    avg_val = np.mean(i_cube[:, :, i], axis=0)

    weighting_cube[:,i] = np.sqrt(((np.mean((i_cube[:, :, i]-avg_val[np.newaxis,:])**2, axis=0))))
    weighting_cube[:,i] = (weighting_cube[:,i]  / np.sum(np.abs(weighting_cube[:,i])))*48

#%%
plt.figure()
im1 = plt.imshow(weighting_cube)
plt.colorbar(im1)
plt.ylabel("Modulation Frame")
plt.xlabel("KL mode")

#%% 
#Save fits file
from astropy.io import fits as pyfits
hdu = pyfits.PrimaryHDU(wfs.read())
hdu.writeto("pyr_test.fits", overwrite=True)
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
modpsf.stop()
time.sleep(1)
slope.stop()





# %%
