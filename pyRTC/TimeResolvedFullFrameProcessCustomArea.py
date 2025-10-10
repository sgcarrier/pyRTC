
"""
Slopes Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

class TimeResolvedFullFrameProcessCustomArea(pyRTCComponent):

    def __init__(self, conf) -> None:

        self.confWFS = conf["trwfs"]
        self.name = "TR_FF"
        self.imageShape = (self.confWFS["nFrames"], self.confWFS["width"], self.confWFS["height"])
        self.nFrames = self.confWFS["nFrames"]
        self.conf = conf["tr_ff"]

        #Read wfs images's metadata and open a stream to the shared memory
        self.wfsMeta = ImageSHM("trwfs_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.imageDType = float_to_dtype(self.wfsMeta[3])
        self.wfsShm = ImageSHM("trwfs", self.imageShape, self.imageDType)

        self.signalDType = np.float32
        # self.signal = ImageSHM("signal", self.imageShape, self.signalDType)
        self.imageNoise = setFromConfig(self.conf,"imageNoise", 0)

        self.wfsType = self.conf["type"]
        self.signalType = self.conf["signalType"]
        self.validSubAps = None

        #Initialize Valid Subaperture Mask
        self.image2DShape = (self.confWFS["width"], self.confWFS["height"])
        self.validSubAps = np.ones(self.image2DShape, dtype=bool)
        self.validSubApsFile = setFromConfig(self.conf, "validSubApsFile", "")
        self.loadValidSubAps()

        #Initialize the reference slopes

        self.signalSize = (np.sum(self.validSubAps))
        self.refSlopesFile = setFromConfig(self.conf, "refSlopesFile", "")
        self.refSlopes = np.zeros((self.nFrames, self.signalSize,), dtype=self.signalDType)
        self.loadRefSlopes()
        self.signal = ImageSHM("signal", (self.nFrames, self.signalSize,), self.signalDType)
        self.signal2D = ImageSHM("signal2D", (self.validSubAps.shape[0], self.validSubAps.shape[1]), self.signalDType)

        super().__init__(self.conf)

    def read(self):
        return self.signal.read()

    def readImage(self):
        return self.wfsShm.read()

    def setValidSubAps(self, validSubAps):
        self.validSubAps = validSubAps.astype(self.validSubAps)
        return

    def saveValidSubAps(self,filename=''):
        if filename == '':
            filename = self.validSubApsFile
        np.save(filename, self.validSubAps)
        return

    def loadValidSubAps(self,filename=''):
        #If no file given, first try reference slopes file
        if filename == '':
            filename = self.validSubApsFile
        #If we are still without a file, set zeros
        if filename == '':
            self.validSubAps = np.ones_like(self.validSubAps)
        else: #If we have a filename
            self.validSubAps = np.load(filename).astype(self.validSubAps.dtype)
        return


    def takeRefSlopes(self):
        #Reset reference slopes to zero
        self.setRefSlopes(np.zeros_like(self.refSlopes))
        refSlopes = np.zeros_like(self.refSlopes)
        #Average self.refSlopeCount slopes measurements
        for i in range(self.refSlopeCount):
            cur_slopes = self.read().astype(refSlopes.dtype)
            refSlopes += self.computeSignal2D(cur_slopes)
        refSlopes /= self.refSlopeCount
        self.setRefSlopes(refSlopes)
        return

    def setRefSlopes(self, refSlopes):
        self.refSlopes = refSlopes.astype(self.signalDType)
        return

    def saveRefSlopes(self,filename=''):
        if filename == '':
            filename = self.refSlopesFile
        np.save(filename, self.refSlopes)
        return

    def loadRefSlopes(self,filename=''):
        #If no file given, first try reference slopes file
        if filename == '':
            filename = self.refSlopesFile
        #If we are still without a file, set zeros
        if filename == '':
            self.refSlopes = np.zeros_like(self.refSlopes)
        else: #If we have a filename
            self.refSlopes = np.load(filename)
        return

    def computeSignal(self):
        cube = self.readImage().astype(self.signalDType)
        if self.signalType == "full_frame":
            validSignal = cube[:, self.validSubAps] / np.sum(cube[:, self.validSubAps])

            self.signal.write(validSignal)
            self.signal2D.write(self.computeSignal2D(validSignal))
        return

    def computeImageNoise(self):
        img = self.readImage()
        if img[img < 0].size > 0:
            self.imageNoise = compute_fwhm_dark_subtracted_image(img)/2
        else:
            print("Image is not dark subtracted")
        return


    def computeSignal2D(self, signal, validSubAps=None):
        if validSubAps is None and isinstance(self.validSubAps, np.ndarray):
            validSubAps = self.validSubAps
        else:
            return -1
        curSignal2D = np.zeros(validSubAps.shape)
        curSignal2D[validSubAps] = np.sum(signal, axis=0)
        return curSignal2D

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity(conf["tr_ff"]["affinity"]%os.cpu_count())
    decrease_nice(pid)

    slopes = TimeResolvedFullFrameProcessCustomArea(conf=conf)
    slopes.start()

    l = Listener(slopes, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)
