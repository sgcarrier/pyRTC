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
conf = read_yaml_file("conf_simple.yaml")

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
wfs.setExposure(0.0325)

### do plt.imshow(wfs.read())
#%%
# pos = {"A": 3.0, "B": 3.0}
# fsm.goTo(pos)
time.sleep(1)
# wfc.flatten()
img_flat = wfs.read().astype(np.float64)


#%% 
############## Put Turbulence on the DM ##############
from scripts.turbulencePreGen import *
turb = np.load("res/turb_coeff_Jun21_with_floating.npy")


turb *= 1
turb_no_piston = turb[:,1:]
turb_no_piston_first_5_modes = turb_no_piston

# Remove some modes if needed
MODES_TO_USE = 5
turb_no_piston_first_5_modes[:, MODES_TO_USE:] = 0

atm = DummyAtm(turb_no_piston_first_5_modes)

t = atm.getNextTurbAsModes()
t_cmd =ModaltoZonalWithFlat(t, 
                     wfc.f_M2C,
                     wfc.flat)


print(np.max(np.abs(t_cmd)))
#%%
atm.currentPos = 0
t = atm.getNextTurbAsModes()

t_cmd = wfc.f_M2C@t
phase_screen = np.zeros((11,11))
phase_screen[atm.mask] =t_cmd 
plt.imshow(phase_screen)
plt.colorbar()
row=15
wfc.write(turb[0,row:])

img_trub = wfs.read().astype(np.float64)

plt.imshow(img_trub-img_flat)



#%% ########### Run our own turbulence on the DM.

sim_atm=OOPAO_atm()

dm_commands=sim_atm.getNextTurbAsModes


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
        frames.append(wfs.read().astype(np.float64))
    cube=np.stack(frames, axis=0)
    master=np.mean(cube, axis=0)
    hdu=fits.PrimaryHDU(data=cube)
    hdu.writeto(filename,overwrite=overwrite)
    print(f"Data Cube saved to {filename}")
    return cube,master

# %%
from scripts.turbulencePreGen import *
turb = np.load("res/turb_coeff_Jun21_with_floating.npy")


turb *= 1
turb_no_piston = turb[:,1:]
turb_no_piston_first_5_modes = turb_no_piston



def get_turb_data(numFrames,fitsfile,filename=turb_no_piston):
    frames=[]
    filename[:,0:2]=0
    for i in range(numFrames):
        wfc.write(filename[i,:])
        plt.pause(0.1)
        frames.append(wfs.read().astype(np.float64))

    cube=np.stack(frames,axis=0)
    master=np.mean(cube, axis=0)
    hdu=fits.PrimaryHDU(data=cube)
    hdu.writeto(fitsfile,overwrite=True)
    print(f"Data Cube saved to {fitsfile}")

    return cube




import matplotlib.animation as animation
 
def create_difference_movie(datacube1, darkframe, output_filename='flat_diff_movie_19-03-25.gif', fps=20):

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

    # Update function for animation.

    def update(frame):

        im.set_array(diff_cube[frame])

        ax.set_title(f"Frame {frame}")

        return im,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)

    # Save the animation. You can change the writer if ffmpeg is not installed.

    ani.save(output_filename, writer='ffmpeg', fps=fps)

    plt.close(fig)

    print(f"Movie saved as {output_filename}")
 
        
# %%
