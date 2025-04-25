# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *
from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Loop import *

from hardware.TimeResolvedLoop import *

from pyRTC.utils import *

from scripts.turbulenceGenerator import OOPAO_atm
# %% 
################### Load configs ###################
conf = read_yaml_file("conf.yaml")

confDM     = conf[    "wfc"]
confMOD    = conf[    "fsm"]
confWFS    = conf[    "wfs"]
confLOOP   = conf[   "loop"]


#%% 
################### Load ALPAO DM and flatten ###################
wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()


#%%
################### Display DM current shape ###################
plt.figure()
currentShape2D = np.zeros(wfc.layout.shape)
currentShape2D[wfc.layout] = wfc.currentShape
plt.imshow(currentShape2D)
plt.colorbar()
plt.show()

# %% 
################### Setup Pi modulator ###################
fsm = PIE517Modulator(conf=confMOD)
pos = {"A": 5.0, "B": 5.0}

fsm.goTo(pos)

################### Setup Andor Camera ###################
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()

wfs.start()

### do plt.imshow(wfs.read())
#%%
# pos = {"A": 3.0, "B": 3.0}
# fsm.goTo(pos)
time.sleep(1)
wfc.flatten()
img_flat = wfs.read().astype(np.float64)

#%% ########### Run our own turbulence on the DM.

sim_atm=OOPAO_atm()

dm_commands=sim_atm.getNextTurbAsModes()
wfc.write(dm_commands)
img_turb= wfs.read().astype(np.float64)

plt.imshow(img_turb-img_flat)


# %%
################### Stop all ###################
fsm.stop()
time.sleep(1)
wfs.stop()
time.sleep(1)
wfs.close_shutter()
time.sleep(1)
wfs.close_camera()
time.sleep(1)

wfc.stop() 
# %%

# %% MAKE FITS file ###########
from astropy.io import fits
def getcube(numFrames,filename,overwrite=True):
    """
    Gets you the data cube for dark and flat. Need to turn the laser on and then run this for you to get flat.
    """
    wfc.flatten()
    frames=[]
    for i in range(numFrames):
        wfc.flatten()
        frames.append(wfs.read().astype(np.float32))
    cube=np.stack(frames, axis=0)
    master=np.mean(cube, axis=0)
    hdu=fits.PrimaryHDU(data=cube)
    hdu.writeto(filename,overwrite=overwrite)
    print(f"Data Cube saved to {filename}")
    return cube,master

# %%



def get_turb_data(numFrames,fitsfile):
    frames=[]
    sim_atm=OOPAO_atm()
    for i in range(numFrames):
        dm_commands=sim_atm.getNextTurbAsModes()
        wfc.write(dm_commands.astype(np.float32))
        plt.pause(0.1)
        frames.append(wfs.read().astype(np.float32))

    cube=np.stack(frames,axis=0)
    master=np.mean(cube, axis=0)
    hdu=fits.PrimaryHDU(data=cube)
    hdu.writeto(fitsfile,overwrite=True)
    print(f"Data Cube saved to {fitsfile}")

    return cube




import matplotlib.animation as animation
 
def create_difference_movie(datacube1, darkframe, output_filename='difference_withmod_movie_28-03-25.gif', fps=20):

    """

    Creates and saves a movie that plots the difference between two data cubes frame by frame.

    """

    # if datacube1.shape != datacube2.shape:

    #     raise ValueError("Both data cubes must have the same shape. Remembger turb data is 10000 by 68.")
    # for i in range(500):
    #     diff_cube=datacube1[i,:,:]-datacube2

    diff_cube = datacube1 - darkframe

    n_frames = diff_cube.shape[0]


    fig, ax = plt.subplots()

    im = ax.imshow(diff_cube[0], cmap='viridis', animated=True)

    ax.set_title("Frame 0")
    cbar = fig.colorbar(im, ax=ax)
    # Update function for animation.

    def update(frame):

        im.set_array(diff_cube[frame])

        ax.set_title(f"Frame {frame}")
        im.set_clim(diff_cube.min(), diff_cube.max())
        cbar.update_normal(im)

        return im,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    # Save the animation. You can change the writer if ffmpeg is not installed.

    ani.save(output_filename, writer='ffmpeg', fps=fps)

    plt.close(fig)

    print(f"Movie saved as {output_filename}")
 
        
# %%

######### get Flat ########
