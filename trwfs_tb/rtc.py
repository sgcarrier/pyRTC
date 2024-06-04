# %% 
from hardware.AndorIXonCam import AndorIXon
from pyRTC.hardware.ALPAODM import *
from hardware.PIE517Modulator import PIE517Modulator
from hardware.PGScienceCam import *
from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess


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


# %% Create DM PSF Cam
#dmpsf = PGScienceCam(conf=confDMPSF)
#dmpsf.start()

#time.sleep(1)
modpsf = PGScienceCam(conf=confMODPSF)
modpsf.start()

#%% Load DM and flatten
wfc = ALPAODM(conf=confDM)
wfc.start()
wfc.flatten()


# %% 
# Setup Pi modulator
fsm = PIE517Modulator(conf=confMOD)
pos = {"A": 5.0, "B": 5.0}

fsm.goTo(pos)

time.sleep(1)

print(fsm.getCurrentPos())


# %%
# Setup Andor Camera
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()

wfs.start()
wfs.setExposure(0.062)

# %% Run a few modulations
for i in range(48*5):
    fsm.step()
    time.sleep(0.1)

# %%
from pipython import GCSDevice, pitools
# %%
fsm.frequency = 16
fsm.sampling = 1/25000
numPoints = int(1.0 / (fsm.frequency * fsm.sampling) )
# %%
fsm.offsetX = 5
fsm.offsetY = 5
fsm.amplitudeX = 1.5*1.288
fsm.amplitudeY = 1.5
# #Define sine and cosine waveforms for wave tables
fsm.mod.WAV_SIN_P(table=fsm.wavetables[0], 
                firstpoint=0, 
                numpoints=numPoints, 
                append='X',
                center=numPoints / 2, 
                amplitude=fsm.amplitudeX, 
                offset=fsm.offsetX- fsm.amplitudeX/2, 
                seglength=numPoints)
# %%
fsm.mod.WAV_SIN_P(table=fsm.wavetables[1], 
                firstpoint=numPoints // 4 + fsm.phaseOffset, 
                numpoints=numPoints, append='X',
                center=numPoints / 2, 
                amplitude=fsm.amplitudeY, 
                offset=fsm.offsetY - fsm.amplitudeY/2, 
                seglength=numPoints)
pitools.waitonready(fsm.mod)



# %%
#Connect wave generators to wave tables 
if fsm.mod.HasWSL(): 
    fsm.mod.WSL(fsm.wavegens, fsm.wavetables)

# %%
startpos = (fsm.offsetX, fsm.offsetY + fsm.amplitudeY / 2)
# %%
pos = {"A": startpos[0], "B": startpos[1]}
#pos = {"A": 5.0, "B": 5.25}
fsm.goTo(pos)


# %%
#Start wave generators {}'.format(self.wavegens))
fsm.mod.WGO(fsm.wavegens, mode=[1] * len(fsm.wavegens))

# %%
#Stop wave
fsm.mod.WGO(fsm.wavegens, mode=[0] * len(fsm.wavegens))
pos = {"A": 5.0, "B": 5.0}
fsm.goTo(pos)
print(fsm.getCurrentPos())


#%%
fsm.start()

#%%
fsm.stop()


#%% Setup SH WFS
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
#shwfs = HASO_SH(confSHWFS) 
#shwfs.start()


#%%
#slope = SlopesProcess(conf)
#slope.start()


# %%
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
fsm.stop()



# %%
