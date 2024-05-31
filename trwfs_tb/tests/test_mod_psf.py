# initialize the DM
from pyRTC.hardware.ALPAODM import *

conf = read_yaml_file("conf.yaml")

# Initialize the PSF camera 
from PGScienceCam import *
confWFC = conf["mod"]
psf = PGScienceCam(conf=confWFC)
psf.start()
psf.setExposure(500) # in us


plt.ion()
fig = plt.figure()
psf.expose()
im = psf.read()
img = plt.imshow(im, cmap='hot')
img.autoscale()
cbar = plt.colorbar(img)
plt.show(block=True)