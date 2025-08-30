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