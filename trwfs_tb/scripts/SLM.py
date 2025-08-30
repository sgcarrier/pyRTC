# %% 
import slmpy
import numpy as np
import time
import matplotlib.pyplot as plt

#main display =0, check windows display settings
slm = slmpy.SLMdisplay(monitor=1)

resX, resY = slm.getSize()


# %% 
def get_petals_tel(resX, resY):

    base  = np.zeros((resY, resX, 4))

    res = min(resX, resY)
    if resX>resY:
        offsetX = abs(resX-resY)//2
        offsetY = 0
    elif resX<resY:
        offsetX = 0
        offsetY = abs(resX-resY)//2
    else:
        offsetX = 0
        offsetY = 0

    x, y = np.meshgrid(np.linspace(0,res,res),np.linspace(0,res,res))

    base_temp = np.zeros((res, res, 4))

    idx0 = ((x+y)<res) & (x>=y)
    #idx0 = (x<res//2) & (y < res//2)
    base[offsetY:offsetY+res,offsetX:offsetX+res,0] = np.where( idx0, 1, base_temp[:,:,0]) 
    idx1 = ((x+y)>=res-1) & (x>=y)
    #idx1 = (x>res//2) & (y < res//2)
    base[offsetY:offsetY+res,offsetX:offsetX+res,1] = np.where( idx1, 1, base_temp[:,:,1])
    idx2 = ((x+y)<res) & (x<=y)
    #idx2 = (x<res//2) & (y > res//2)

    base[offsetY:offsetY+res,offsetX:offsetX+res,2] = np.where( idx2, 1, base_temp[:,:,2])
    idx3 = ((x+y)>=res-1) & (x<=y)
    #idx3 = (x>res//2) & (y > res//2)

    base[offsetY:offsetY+res,offsetX:offsetX+res,3] = np.where( idx3, 1, base_temp[:,:,3])
    return base

petals = get_petals_tel(resX, resY)
# %% 

def get_spiders(angle, thickness_spider, offset_X=None, offset_Y=None):

    resolution = 600
    if thickness_spider > 0:

        max_offset = thickness_spider/2
        if offset_X is None:
            offset_X = np.zeros(len(angle))

        if offset_Y is None:
            offset_Y = np.zeros(len(angle))

        if np.max(np.abs(offset_X)) >= max_offset or np.max(np.abs(offset_Y)) > max_offset:
            print('The spider offsets are too large! Weird things could happen!')
        for i in range(len(angle)):
            angle_val = (angle[i]+90) % 360
            x = np.linspace(0, resolution, resolution)
            [X, Y] = np.meshgrid(x, x)
            X += offset_X[i]
            Y += offset_Y[i]
            map_dist = np.abs(X*np.cos(np.deg2rad(angle_val)) + Y*np.sin(np.deg2rad(-angle_val)))
            if 0 <= angle_val < 90:
                map_dist[:resolution//2, :] = thickness_spider
            if 90 <= angle_val < 180:
                map_dist[:, :resolution//2] = thickness_spider
            if 180 <= angle_val < 270:
                map_dist[resolution//2:, :] = thickness_spider
            if 270 <= angle_val < 360:
                map_dist[:, resolution//2:] = thickness_spider
            map_dist[map_dist> thickness_spider/2] = 1
            return map_dist
    else:
        print('Thickness is <=0, returning 0')
    return 0 

spiders = get_spiders([45, 45+90, 45+180, 45+270], 40)
# %%


def simple_spider(res, thickness):

    x, y = np.meshgrid(np.linspace(0,res,res),np.linspace(0,res,res))

    spider = np.round(np.abs(x-y)<=thickness).astype('uint8')
    spider += np.flip(spider, axis=1)
    spider[spider>0] = 1
    return spider 


spider = simple_spider(300, 20)
# %%


final_image = np.zeros((300, 400), dtype='uint8')

final_image[:, 50:350] = spider *200

# %%
slm.updateArray(final_image)


#%%
def simple_spider_crit(x_off, y_off, x_max, y_max, thickness):


    x, y = np.meshgrid(np.linspace(0,x_max,x_max),np.linspace(0,y_max,y_max))

    spider = np.round(np.abs((x-x_off)-(y-y_off))<=thickness).astype('uint8')
    spider += np.flip(spider, axis=1)
    spider[spider>0] = 1
    return spider 

spider = simple_spider_crit(400, 300, 800, 600, 20)

plt.imshow(spider)

#%%
X,Y = np.meshgrid(np.linspace(0,resX,resX),np.linspace(0,resY,resY))
testIMG = np.round((2**8-1)*(0.5+0.5*np.sin(2*np.pi*X/50))).astype('uint8')

X,Y = np.meshgrid(np.linspace(0,resX-1,resX),np.linspace(0,resY-1,resY))

pupil_size = 250
pupil_crit = ((X-(resX/2))**2 + (Y-(resY/2))**2 <= pupil_size**2)
grating_crit = ((Y % 2) == 0 )
spacing_middle_crit = (Y!=300)


gratingIMG = (grating_crit & spacing_middle_crit).astype('uint8') *(133//2)
pupilIMG = (pupil_crit).astype('uint8')

final = gratingIMG
final[(pupilIMG==1) & (spider==0)] = 0

plt.imshow(final)

#%%
slm.updateArray(final)


#%%
image_path = "res/flat_slm_lsh0702233/CAL_LSH0702233_630nm.bmp"

from PIL import Image
# Open the BMP image using Pillow
pil_image = Image.open(image_path)

# Convert the Pillow Image object to a NumPy array
# The array will have dimensions (height, width, channels) for color images
# or (height, width) for grayscale images.
flat_img = np.array(pil_image)

slm.updateArray(flat_img)

#%%
slm.close()
# %%



import h5py
import numpy as np

# Specify the path to your .mat file
file_path = "res/flat_slm_lsh0702233/alpha_slm_lsh0702233.mat"

# Open the .mat file in read mode
with h5py.File(file_path, 'r') as f:
    # List the top-level keys (variables) in the .mat file
    print("Keys in .mat file:", list(f.keys()))

    data = f['alpha_tab'][:] 

#%%

wavelength = 635e-9

alpha_tab = data
alpha_tab[:,0] *= 1e-9 
w1_diff = np.abs(alpha_tab[:,0] - wavelength)
sort_idx = np.argsort(w1_diff)

alpha = np.uint8((((wavelength - \
                                alpha_tab[sort_idx[0], 0])) * \
                               ((alpha_tab[sort_idx[1], 1] - \
                                alpha_tab[sort_idx[0], 1])) / \
                               ((alpha_tab[sort_idx[1], 0] - \
                                alpha_tab[sort_idx[0], 0]))) + \
                              alpha_tab[sort_idx[0], 1])
# %%
