# Find center of pupils
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def findCenterOfPupilsSimple(wfs):
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

def findPupilPosAndRadius(img, size):
    # img_bin = np.zeros(img.shape) 
    # img_bin[img>(np.max(img)*0.1)] = 1
    img_bin = img
    image = img_bin[:size,:size]
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

def findAllPupils(img, quadrant_size):
    pos = []
    x0, y0, r0 = findPupilPosAndRadius(img[:quadrant_size,:quadrant_size], quadrant_size)
    pos.append((x0, y0, r0))
    x1, y1, r1 = findPupilPosAndRadius(img[quadrant_size:,:quadrant_size], quadrant_size)
    y1 += quadrant_size
    pos.append((x1, y1, r1))
    x2, y2, r2 = findPupilPosAndRadius(img[:quadrant_size,quadrant_size:], quadrant_size)
    x2 += quadrant_size
    pos.append((x2, y2, r2))
    x3, y3, r3 = findPupilPosAndRadius(img[quadrant_size:,quadrant_size:], quadrant_size)
    x3 += quadrant_size
    y3 += quadrant_size
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

    
def autoFindAndDisplayPupils(wfs, quadrant_size):

    numImages = 20
    img = wfs.read().astype(np.float64)
    for i in range(numImages-1):
        img += wfs.read().astype(np.float64)
    img /= numImages

    img_bin = np.zeros(img.shape)
    img_bin[img>(np.max(img)*0.3)] = 1
    pos = findAllPupils(img_bin, quadrant_size)
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

    return pos


def overlayCalcPosWithPupilMask(pos, slope):
    f, ax = plt.subplots()
    ax.imshow(slope.pupilMask, cmap='gray', interpolation='nearest')
    for i in range(4):
        cir = plt.Circle((pos[i][0], pos[i][1]), pos[i][2], color='red', fill=False)
        ax.add_artist(cir)
    plt.show()