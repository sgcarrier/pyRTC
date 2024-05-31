from trwfs_tb.hardware.PIE517Modulator import PIE517Modulator
from pyRTC.utils import *
import time

conf = read_yaml_file("conf.yaml")

fsm = PIE517Modulator(conf['fsm'])

pos = {"A": 5.5, "B": 5.0}


fsm.goTo(pos)

time.sleep(1)

print(fsm.getCurrentPos())