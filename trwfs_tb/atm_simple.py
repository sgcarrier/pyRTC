from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Loop import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface

from hardware.TimeResolvedLoop import *
from scripts.turbulenceGenerator import OOPAO_atm

from pyRTC.utils import *

from OOPAO.Atmosphere import Atmosphere
#from OOPAO.DeformableMirror import DeformableMirror
#from OOPAO.MisRegistration import MisRegistration
#from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
import numpy as np

################### Load configs ###################
conf = read_yaml_file("atm_simple.yaml")

confDM     = conf[    "wfc"]
confLOOP   = conf[   "loop"]
sim = OOPAOInterface(conf=conf, param=None)
wfs, dm, psf = sim.get_hardware()

################### Load ALPAO DM and flatten ###################
#wfc = ALPAODM(conf=confDM)
#wfc.start()
#wfc.flatten()

sim_atm=OOPAO_atm()

dm_commands=sim_atm.getNextTurbAsModes() #here are actuators are the model basis. Should rename function in the future I think
#can create a loop to get dm commands and save them to .npy file if things look correct
plt.plot(dm_commands)
sim_atm.getdmplot()
tel = Telescope(   resolution          =  22,
                                diameter            =  1,
                                samplingTime        =  1,
                                centralObstruction  =  0)
mask=tel.pupil.copy().astype('int')

print(np.sum(mask[:]))
frames=[]
for i in range (500):
    sim_atm.getNextTurbAsModes()
    dm_commands=sim_atm.getNextTurbAsModes()
    # create movie of plots 
    frame=sim_atm.getdmplot()
    # get
    frames.append(frame)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig, ax = plt.subplots()

# # This function updates the plot for each frame
# def update(frame):
#     ax.clear()
#     ax.title.set_text(f'Dm Commands {frame}')
#     ax.imshow(frame)  # Display the image
#     # add colorbars
    
# ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
# ani.save('dm_commands.mp4', writer='ffmpeg', fps=10)

####### Fixed Colorbar #######

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Create the initial plot with the first frame from your list
# fig, ax = plt.subplots()
# im = ax.imshow(frames[0], cmap='viridis', vmin=-0.5, vmax=0.5)
# cbar = fig.colorbar(im, ax=ax)  # Create a fixed colorbar

# # Update function that simply updates the image data
# def update(frame):
#     im.set_data(frame)
#     return [im]

# ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
# ani.save('dm_movie_fixed_colorbar.mp4', writer='ffmpeg', fps=10)
# plt.show()


# #### Varying color bar #########
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Create the initial plot with the first frame
# fig, ax = plt.subplots()
# im = ax.imshow(frames[0], cmap='viridis')
# cbar = fig.colorbar(im, ax=ax)

# # Update function adjusts the image data and color limits
# def update(frame):
#     im.set_data(frame)
#     # Set color limits based on the current frame's min and max values
#     im.set_clim(frame.min(), frame.max())
#     cbar.update_normal(im)
#     return [im]

# ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
# ani.save('dm_movie_varying_colorbar.mp4', writer='ffmpeg', fps=10)
# plt.show()


####### Mean Frame ########
mean_frame = np.mean(frames, axis=0)  # shape is the same as a single frame

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(mean_frame, cmap='viridis')
cbar2 = fig2.colorbar(im2, ax=ax2)
cbar2.set_label("Mean DM Command")
ax2.set_title("Time-Averaged DM Shape")
plt.show()