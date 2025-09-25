#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib


#%%

f = h5py.File('sky_with_dark_with_correction_1h05_24sept2025.hdf5', 'r')


#%%

dataset = np.copy(f['images'])

#%% Remove Screamers
dataset[:,:,20, 12] = 0


#%%
all_stacked = np.sum(np.sum(dataset, axis=0),axis=0)

plt.imshow(all_stacked)
plt.title("All acquisitons and frames stacked")
#%%
frame_number = 15
all_stacked_frame = np.sum(dataset[:,frame_number, :,:],axis=0)

plt.imshow(all_stacked_frame)
plt.title(f"All acquisitons of frame {frame_number} stacked")
# %%

from mpl_toolkits.axes_grid1 import ImageGrid

all_stacked_frame = np.sum(dataset[:,:,:,5:40], axis=0)
imgs = []
for i in range(48):
    imgs.append(all_stacked_frame[i,:,:])

fig = plt.figure(figsize=(22., 8.0))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 12),  # creates 2x2 grid of Axes
                 axes_pad=0.01,  # pad between Axes in inch.
                 )

for ax, im in zip(grid, imgs):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("1000 acquisitons stacked, each frame", fontsize=36)

plt.show()
# %%

# %%


flat_signal = np.load("res/flat.npy")
layout = np.array([[0,0,0,0,0,0,271,272,273,274,275,276,277,0,0,0,0,0,0],
                                [0,0,0,0,0,262,263,264,265,266,267,268,269,270,0,0,0,0,0],
                                [0,0,0,0,251,252,253,254,255,256,257,258,259,260,261,0,0,0,0],
                                [0,0,0,238,239,240,241,242,243,244,245,246,247,248,249,250,0,0,0],
                                [0,0,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,0,0],
                                [0,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,0],
                                [187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205],
                                [168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186],
                                [149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167],
                                [130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148],
                                [111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129],
                                [92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110],
                                [73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91],
                                [0,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,0],
                                [0,0,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,0,0],
                                [0,0,0,28,29,30,31,32,33,34,35,36,37,38,39,40,0,0,0],
                                [0,0,0,0,17,18,19,20,21,22,23,24,25,26,27,0,0,0,0],
                                [0,0,0,0,0,8,9,10,11,12,13,14,15,16,0,0,0,0,0],
                                [0,0,0,0,0,0,1,2,3,4,5,6,7,0,0,0,0,0,0]]) > 0

img_layout = np.zeros_like(layout, dtype=flat_signal.dtype)
img_layout[layout] = flat_signal
img_layout_flipped = (img_layout).T
plt.imshow(img_layout_flipped)
flat_singal_flipped = img_layout_flipped[layout]
np.save("res/flat_flipped_T.npy", flat_singal_flipped)


# %%
m2c_signal_loaded = np.load("res/m2c.npy")

#remove_piston
m2c_signal = m2c_signal_loaded[:, 1:]
#%%
layout = np.array([[0,0,0,0,0,0,271,272,273,274,275,276,277,0,0,0,0,0,0],
                                [0,0,0,0,0,262,263,264,265,266,267,268,269,270,0,0,0,0,0],
                                [0,0,0,0,251,252,253,254,255,256,257,258,259,260,261,0,0,0,0],
                                [0,0,0,238,239,240,241,242,243,244,245,246,247,248,249,250,0,0,0],
                                [0,0,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,0,0],
                                [0,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,0],
                                [187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205],
                                [168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186],
                                [149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167],
                                [130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148],
                                [111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129],
                                [92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110],
                                [73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91],
                                [0,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,0],
                                [0,0,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,0,0],
                                [0,0,0,28,29,30,31,32,33,34,35,36,37,38,39,40,0,0,0],
                                [0,0,0,0,17,18,19,20,21,22,23,24,25,26,27,0,0,0,0],
                                [0,0,0,0,0,8,9,10,11,12,13,14,15,16,0,0,0,0,0],
                                [0,0,0,0,0,0,1,2,3,4,5,6,7,0,0,0,0,0,0]]) > 0

m2c_flipped = np.zeros_like(m2c_signal)

for i in range(m2c_signal.shape[1]):
    img_layout = np.zeros_like(layout, dtype=m2c_signal.dtype)
    flat_signal = m2c_signal[:,i]
    img_layout[layout] = flat_signal
    img_layout_flipped =(img_layout).T
    m2c_flipped[:,i] = img_layout_flipped[layout]


np.save("res/m2c_flipped_T.npy", m2c_flipped)

#%%



