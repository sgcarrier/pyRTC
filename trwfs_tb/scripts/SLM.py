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

final_image[:, 50:350] = spider *1

# %%
slm.updateArray(final_image)


#%%
slm.close()
# %%
