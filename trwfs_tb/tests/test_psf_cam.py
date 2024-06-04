from hardware.PGScienceCam import *

import matplotlib.pyplot as plt

conf = read_yaml_file("trwfs_tb/conf.yaml")

#pid = os.getpid()
#set_affinity((conf["wfc"]["affinity"])%os.cpu_count()) 
#decrease_nice(pid)

confWFC = conf["dmpsf"]
psf = PGScienceCam(conf=confWFC)
psf.start()

#psf.expose()

time.sleep(0.1)

im = psf.read()

plt.figure()

imobj = plt.imshow(im)
plt.show(block=True)

#time.sleep(10)

psf.stop()

#for i in range(200):
#    print(f"frame{i}")
#    psf.expose()
#    im = psf.read()
#    imobj.set_data(im)
#    plt.draw()
#    plt.show(block=False)
#    time.sleep(1)

