"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sys import platform

class TimeResolvedWavefrontSensor(pyRTCComponent):

    def __init__(self, conf) -> None:

        self.name = conf["name"]
        self.cubeShape = (conf["nFrames"],conf["width"], conf["height"])
        self.imageShape = (conf["width"], conf["height"])
        self.cubeRawDType = np.uint16
        self.cubeDType = np.int32
        
        self.cubeRaw = ImageSHM("trwfsRaw", self.cubeShape, self.cubeRawDType)
        self.cube = ImageSHM("trwfs", self.cubeShape, self.cubeDType)

        self.data = np.zeros(self.cubeShape, dtype=self.cubeRawDType)
        self.dark = np.zeros(self.imageShape, dtype=self.cubeDType)

        self.affinity = conf["affinity"]
        self.darkCount = setFromConfig(conf, "darkCount", 1000)
        self.darkFile = setFromConfig(conf, "darkFile", "")

        self.loadDark()

        super().__init__(conf)

        return
    
    def setRoi(self, roi):

        self.roiWidth = roi[0]
        self.roiHeight = roi[1]
        self.roiLeft = roi[2]
        self.roiTop = roi[3]
        return

    def setExposure(self, exposure):
        self.exposure = exposure
        return
    
    def setBinning(self, binning):
        self.binning = binning
        return
    
    def setGain(self, gain):
        self.gain = gain
        return
    
    def setBitDepth(self, bitDepth):
        self.bitDepth = bitDepth
        return
    
    def expose(self):
        self.cubeRaw.write(self.data)
        self.cube.write(self.data.astype(self.cubeDType) - self.dark[np.newaxis, :,:])
        return

    def read(self):
        return self.cube.read()
    
    def takeDark(self):
        self.setDark(np.zeros_like(self.dark))
        dark = np.zeros(self.imageShape, dtype=np.float64)
        for i in range(self.darkCount):
            frames = self.read().astype(np.float64)
            dark += np.mean(frames, axis=0)
        dark /= self.darkCount
        self.setDark(dark)        
        return 

    def setDark(self, dark):
        self.dark = dark.astype(self.cubeDType)
        return
    
    def saveDark(self,filename=''):
        if filename == '':
            filename = self.darkFile
        np.save(filename, self.dark)
        return
    
    def loadDark(self,filename=''):
        #If no file given, first try dark file
        if filename == '':
            filename = self.darkFile
        #If we are still without a file, set zeros
        if filename == '':
            self.dark = np.zeros_like(self.dark)
        else: #If we have a filename
            self.dark = np.load(filename)
        return
    
    def plot(self):
        arr = np.sum(self.read(), axis=0)
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()
        return