# %% 
#from hardware.AndorIXonCam import AndorIXon
#from pyRTC.hardware.ALPAODM import *
#from hardware.PIE517Modulator import PIE517Modulator
#from hardware.PGScienceCam import *
#from hardware.HASO_SH import HASO_SH
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Loop import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface

from hardware.TimeResolvedLoop import *


from pyRTC.utils import *


# %% 
################### Load configs ###################
conf = read_yaml_file("atm_simple.yaml")

confDM     = conf[    "wfc"]
confLOOP   = conf[   "loop"]
sim = OOPAOInterface(conf=conf, param=None)
wfs, dm, psf = sim.get_hardware()
#%% 
################### Load ALPAO DM and flatten ###################
#wfc = ALPAODM(conf=confDM)
#wfc.start()
#wfc.flatten()





# %%
