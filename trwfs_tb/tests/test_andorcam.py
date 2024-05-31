from trwfs_tb.hardware.AndorIXonCam import AndorIXon
from pyRTC.utils import *
import matplotlib.pyplot as plt

import time

conf = read_yaml_file("conf.yaml")

confWFS = conf["wfs"]
wfs = AndorIXon(conf=confWFS)
wfs.open_shutter()

wfs.start()
#wfs.expose()

im = wfs.read()

plt.figure()

imobj = plt.imshow(im)
plt.show(block=True)



wfs.stop()

time.sleep(1)

wfs.close_shutter()



