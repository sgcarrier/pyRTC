from pyRTC.utils import *
from pyRTC.Modulator import *
from pyRTC.Pipeline import *

import os
import argparse

from pipython import GCSDevice, pitools

class PIE517Modulator(Modulator):
    def __init__(self, conf) -> None:
        #Initialize the pyRTC super class
        super().__init__(conf)

        self.amplitudeX = conf["amplitude"]
        self.frequency = conf["frequency"]
        self.amplitudeY = conf["amplitude"]*conf["relativeAmplitude"]
        self.offsetX = conf["offsetX"]
        self.offsetY = conf["offsetY"]
        self.phaseOffset = conf["phaseOffset"]
        self.sampling = 1/conf["digitalFreq"]

        self.maxChannelValue = conf["maxValue"]
        self.minChannelValue = conf["minValue"]

        self.wavegens = (1, 2)
        self.wavetables = (1, 2)

        self.numOfTRFrames = conf["numOfTRFrames"]
        self.modLambdaFactor = conf["modLambdaFactor"]


        #originalDirectory = os.getcwd()
        #os.chdir(conf['libFolder'])
        self.mod = GCSDevice()
        devices = self.mod.EnumerateUSB()
        self.mod.ConnectUSB(devices[0])
        #os.chdir(originalDirectory)

        #self.servosOn = conf["servosOn"]
        #for axis in self.mod.axes:
        #    self.mod.SVO(axis, int(conf["servosOn"]))

        #if conf["autoZero"]:
        #    self.mod.ATZ()

        self.defineCircle()

        self.makeWavetables()

        return
    
    def __del__(self):
        super().__del__()
        self.mod.close()
        
        return  
    
    def makeWavetables(self):
        numPoints = int(1.0 / (self.frequency * self.sampling) )
        # #Define sine and cosine waveforms for wave tables
        self.mod.WAV_SIN_P(table=self.wavetables[0], 
                        firstpoint=0, 
                        numpoints=numPoints, 
                        append='X',
                        center=numPoints / 2, 
                        amplitude=self.amplitudeX, 
                        offset=self.offsetX- self.amplitudeX/2, 
                        seglength=numPoints)

        self.mod.WAV_SIN_P(table=self.wavetables[1], 
                        firstpoint=numPoints // 4 + self.phaseOffset, 
                        numpoints=numPoints, append='X',
                        center=numPoints / 2, 
                        amplitude=self.amplitudeY, 
                        offset=self.offsetY - self.amplitudeY/2, 
                        seglength=numPoints)
        pitools.waitonready(self.mod)

        if self.mod.HasWSL(): 
            self.mod.WSL(self.wavegens, self.wavetables)


    def start(self):
        if self.checkValues():
            super().start()
            #Move axes to their start positions
            startpos = (self.offsetX, self.offsetY + self.amplitudeY / 2)
            pos = {"A": startpos[0], "B": startpos[1]}
            self.goTo(startpos)
            
            #Start wave generators {}'.format(self.wavegens))
            self.mod.WGO(self.wavegens, mode=[1] * len(self.wavegens))


    def stop(self):
        super().stop()
        #Reset wave generators
        self.mod.WGO(self.wavegens, mode=[0] * len(self.wavegens))

        pitools.waitonready(self.mod)
        time.sleep(0.1)
        pos = {"A": self.offsetX, "B": self.offsetY}
        self.goTo(pos)
        pitools.waitonready(self.mod)
        return
    

    def checkValues(self):
        #Check max/min values
        max_pos = np.max([self.offsetX + (self.amplitudeX / 2), self.offsetY + (self.amplitudeY / 2) ])
        min_pos = np.min([self.offsetX - (self.amplitudeX / 2), self.offsetY - (self.amplitudeY / 2) ])

        if max_pos > self.maxChannelValue:
            print("WARNING!!! Tried to set a value in the wavetable that EXCEEDS max channel value setting!")
            return False
        if min_pos < self.minChannelValue:
            print("WARNING!!! Tried to set a value in the wavetable BELOW min channel value setting!")
            return False
        
        return True

    def defineCircle(self):
        if self.checkValues():
            angles = np.linspace(0, np.pi*2, self.numOfTRFrames, endpoint=False)

            XX = np.cos(angles)*(self.amplitudeX/2) + self.offsetX
            YY = np.sin(angles)*(self.amplitudeY/2) + self.offsetY

            self.points = []
            for i in range(self.numOfTRFrames):
                self.points.append({"A": XX[i], "B": YY[i]})


        self.currentPos = None

    def step(self):
        if self.running:
            print("Warning tried to move manually when modulating with wavetable! Ignoring move command")
            return 
        
        if self.currentPos == None:
            self.currentPos = 0
            self.goTo(self.points[self.currentPos])
        elif self.currentPos >= (self.numOfTRFrames-1):
            self.currentPos = 0
            self.goTo(self.points[self.currentPos])
        else:
            self.currentPos += 1
            self.goTo(self.points[self.currentPos])

    
    def resetPos(self):
        '''
        Reset the current position counter to None and sets the position to default.
        This means the next time step() is called, it will go to the first position
        in the defined circle.
        '''
        if self.running:
            print("Warning tried to move manually when modulating with wavetable! Ignoring move command")
            return 
        
        self.currentPos = None
        pos = {"A": self.offsetX, "B": self.offsetY}
        self.goTo(pos)
        pitools.waitonready(self.mod)
        return


    def goTo(self, pos):
        if self.running:
            print("Warning tried to move manually when modulating with wavetable! Ignoring move command")
            return 
        
        if isinstance(pos, dict):
            # Using a dictionary of the channels and values we want to assign to those channels
            for channel, value in pos.items():
                if self.minChannelValue <= value <= self.maxChannelValue:
                    self.mod.MOV(channel, value)
                else:
                    return -1
            # Wait for the channels to stabilize
            pitools.waitontarget(self.mod, list(pos.keys()))

    def getCurrentPos(self):
        return dict(self.mod.qPOS())


           
