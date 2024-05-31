# Need to copy over the Lib64 folder from the ALPAO python SDK to the python packages directory
from pyRTC.hardware.ALPAODM import *

conf = read_yaml_file("conf.yaml")

#pid = os.getpid()
#set_affinity((conf["wfc"]["affinity"])%os.cpu_count()) 
#decrease_nice(pid)

confWFC = conf["wfc"]
wfc = ALPAODM(conf=confWFC)


print(wfc.dm.Get('NBOfActuator'))

print("Done")