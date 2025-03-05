"""
Loop Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sys import platform

from pyRTC.SlopesProcess import SlopesProcess

@jit(nopython=True)
def computeFullFramePYWFS(p1=np.array([],dtype=np.float32), 
                       p2=np.array([],dtype=np.float32),
                       p3=np.array([],dtype=np.float32), 
                       p4=np.array([],dtype=np.float32), 
                       flatNorm=True):
    signal = np.concatenate((p1,p2,p3,p4))
    #signal_normed = signal / np.sum(signal)

    return signal

class FullFrameProcess(SlopesProcess):

    def __init__(self, conf) -> None:

        super().__init__(conf)


        self.refSignal = np.zeros((self.signalSize))
        return

    def __del__(self):
        self.stop()
        self.alive=False
        return



    def takeRefFullFrame(self):
        iters= 10
        ff_signal = np.zeros((self.signalSize))
        for i in range(iters):
            image = self.readImage().astype(self.signalDType)
            if self.signalType == "slopes":
                if self.wfsType == "PYWFS":
                    p1,p2,p3,p4 = image[self.p1mask], image[self.p2mask], image[self.p3mask], image[self.p4mask]
                    ff_signal += computeFullFramePYWFS(p1=p1,
                                                        p2=p2,
                                                        p3=p3,
                                                        p4=p4,
                                                        flatNorm=self.flatNorm)
                    
        ff_signal /= iters
        self.refSignal = ff_signal
                



    def takeRefSlopes(self):    
        return 

    def setRefSlopes(self, refSlopes):
        return
    
    def saveRefSlopes(self,filename=''):
        return

    def loadRefSlopes(self,filename=''):
        return
    
    def computeSignal(self):
        image = self.readImage().astype(self.signalDType)
        if self.signalType == "slopes":
            if self.wfsType == "PYWFS":
                p1,p2,p3,p4 = image[self.p1mask], image[self.p2mask], image[self.p3mask], image[self.p4mask]
                ff_signal = computeFullFramePYWFS(p1=p1,
                                                    p2=p2,
                                                    p3=p3,
                                                    p4=p4,
                                                    flatNorm=self.flatNorm)
                
                
                    
            elif self.wfsType == "SHWFS":
                
                # threshold = np.std(image[image < np.mean(image)])*self.shwfsContrast
                threshold = self.imageNoise*self.shwfsContrast
                image[image < threshold] = 0
                slopes = computeSlopesSHWFS(image, 
                                                    self.refSlopes, 
                                                    self.subApSpacing,
                                                    self.offsetX,
                                                    self.offsetY)
                slope_signal = slopes[self.validSubAps]

            self.signal.write(ff_signal-self.refSignal)
            self.signal2D.write(self.computeSignal2D(ff_signal-self.refSignal))
        return
    
    def computeImageNoise(self):
        img = self.readImage()
        if img[img < 0].size > 0:
            self.imageNoise = compute_fwhm_dark_subtracted_image(img)/2
        else:
            print("Image is not dark subtracted")
        return

    def setPupils(self, pupilLocs, pupilRadius):
        self.pupilLocs = pupilLocs
        self.pupilRadius = pupilRadius
        self.computePupilsMask()
        if self.signalType == "slopes":
            self.signalSize = np.count_nonzero(self.pupilMask)
            slopemask =  self.pupilMask[self.pupilLocs[0][1]-self.pupilRadius+1:self.pupilLocs[0][1]+self.pupilRadius, 
                                        self.pupilLocs[0][0]-self.pupilRadius+1:self.pupilLocs[0][0]+self.pupilRadius] > 0
            self.setValidSubAps(np.concatenate([slopemask, slopemask, slopemask, slopemask], axis=1))
            self.signal = ImageSHM("signal", (self.signalSize,), self.signalDType)
            self.signal2D = ImageSHM("signal2D", (self.validSubAps.shape[0], self.validSubAps.shape[1]), self.signalDType)
            
        return

    def computePupilsMask(self):
        pupils = []
        self.pupilMask = np.zeros(self.imageShape)
        xx,yy = np.meshgrid(np.arange(self.pupilMask.shape[0]),np.arange(self.pupilMask.shape[1]))
        for i, pupil_loc in enumerate(self.pupilLocs):
            px, py = pupil_loc
            zz = np.sqrt((xx-px)**2 + (yy-py)**2)
            pupils.append(zz < self.pupilRadius)
            self.pupilMask += pupils[-1]*(i+1)
        self.p1mask = self.pupilMask == 1
        self.p2mask = self.pupilMask == 2
        self.p3mask = self.pupilMask == 3
        self.p4mask = self.pupilMask == 4
        return

    def plotPupils(self):
        # plt.figure(figsize=(10,8))
        plt.imshow(self.pupilMask, cmap = 'inferno',origin='lower',aspect ='auto')
        plt.colorbar()
        plt.title("Pupil Mask (Value is Pupil Number)")
        plt.show()

        plt.imshow(self.pupilMask*self.readImage(), cmap = 'inferno',origin='lower',aspect ='auto')
        colors = ['g','b','orange', 'r']
        for i in range(len(self.pupilLocs)):
            px, py = self.pupilLocs[i]
            plt.axvline(x = px, color = colors[i], alpha = 0.6)
            plt.axhline(y = py, color = colors[i], alpha = 0.6)
        plt.colorbar()
        plt.title("Pupil Mask * Image ")
        plt.show()
        return

    def computeSignal2D(self, signal, validSubAps=None):
        if validSubAps is None and isinstance(self.validSubAps, np.ndarray):
            validSubAps = self.validSubAps
        else:
            return -1
        curSignal2D = np.zeros(validSubAps.shape)
        if self.wfsType.lower() == "pywfs":
            slopemask = validSubAps[:,:validSubAps.shape[1]//4]
            curSignal2D[:,:validSubAps.shape[1]//4][slopemask] = signal[:signal.size//4]
            curSignal2D[:,validSubAps.shape[1]//4:validSubAps.shape[1]//2][slopemask] = signal[signal.size//4:signal.size//2]
            curSignal2D[:,validSubAps.shape[1]//2:(validSubAps.shape[1]//4)*3][slopemask] = signal[signal.size//2:(signal.size//4)*3]
            curSignal2D[:,(validSubAps.shape[1]//4)*3:][slopemask] = signal[(signal.size//4)*3:]

        else:
            curSignal2D[validSubAps] = signal
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
    set_affinity(conf["slopes"]["affinity"]%os.cpu_count())
    decrease_nice(pid)

    slopes = FullFrameProcess(conf=conf)
    slopes.start()

    l = Listener(slopes, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)