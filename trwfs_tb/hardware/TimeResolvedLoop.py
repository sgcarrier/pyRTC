from pyRTC.Loop import *
from scripts.modulation_weights import *


@jit(nopython=True)
def updateCorrectionTR(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64)):
    signal_per_mode = slopes_TR @ weights
    #TODO Might be able to optimize this with einsum
    new_corr = np.diag(gCM.astype(np.float64) @ signal_per_mode)
    return correction - new_corr

class TimeResolvedLoop(Loop):


    def __init__(self, conf, fsm) -> None:
        #Initialize the pyRTC Loop super class
        super().__init__(conf)

        self.numFrames  = len(fsm.points)
        self.weightFile = setFromConfig(self.confLoop, "weightFile", "")
        self.loadWeights()

        self.fsm = fsm

        self.IM_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        self.IMCubeFile = setFromConfig(self.confLoop, "IMCubeFile", "")

        self.loadIMCube()

    def calcFrameWeights(self, maxNumModes=None):
        '''
        Calculate the weights to give to every frame for every mode
        '''
        self.fsm.stop()
        self.fsm.currentPos = None
        pos = {"A": 5.0, "B": 5.0}
        self.fsm.goTo(pos)
        time.sleep(0.1)
        im_cube = self.pushPullIM_cube(self.fsm, maxNumModes=maxNumModes)

        weighting_cube = modWeightsFromIMCube(im_cube=im_cube)
        self.frame_weights[:,:maxNumModes] = weighting_cube

    def loadWeights(self,filename=''):
        self.frame_weights = np.ones((self.numFrames,self.numModes)) / self.numFrames

        if filename == '':
            filename = self.weightFile
        if filename == '':
            self.frame_weights = np.ones((self.numFrames,self.numModes)) / self.numFrames
        else:
            self.frame_weights = np.ones((self.numFrames,self.numModes)) / self.numFrames
            frame_weights_from_file = np.load(filename)
            self.frame_weights[:, :frame_weights_from_file.shape[1]] = frame_weights_from_file

    def loadIMCube(self,filename=''):
        if filename == '':
            filename = self.IMCubeFile
        if filename == '':
            self.IM_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        else:
            self.IM_cube = np.load(filename)

        self.IM = np.sum(self.IM_cube * self.frame_weights[np.newaxis, :, :], axis=1)

        self.computeCM()

    def pushPullIM_cube(self, maxNumModes=None):

        numModFrames = len(self.fsm.points)

        if maxNumModes is None:
            maxNumModes = self.numModes
        if maxNumModes > self.numModes:
            maxNumModes = self.numModes

        ref_slopes = np.zeros((self.signalSize, self.numFrames))
        self.fsm.stop()
        self.fsm.resetPos()

        for s in range(self.numFrames):
            self.fsm.step()
            #Average out N new WFS frames
            ref_slopes[:,s] =  np.zeros((self.signalSize))
            for n in range(self.numItersIM):
                ref_slopes[:,s] += self.wfsShm.read()
            ref_slopes[:,s] /= self.numItersIM

        #For each mode
        for i in range(maxNumModes):
            #Reset the correction
            correction = self.flat.copy()
            #Plus amplitude
            correction[i] = self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.wfsShm.read()

            self.fsm.currentPos = None
            tmp_plus =  np.zeros((self.signalSize, self.numFrames))
            for s in range(self.numFrames):
                self.fsm.step()
                #Average out N new WFS frames
                for n in range(self.numItersIM):
                    tmp_plus[:,s] += self.wfsShm.read()
                tmp_plus[:,s] /= self.numItersIM


                tmp_plus[:,s] = tmp_plus[:,s] - ref_slopes[:,s]

            #minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.wfsShm.read()

            self.fsm.currentPos = None
            tmp_minus =  np.zeros((self.signalSize, self.numFrames))
            for s in range(self.numFrames):
                self.fsm.step()
                #Average out N new WFS frames
                for n in range(self.numItersIM):
                    tmp_minus[:,s] += self.wfsShm.read()
                tmp_minus[:,s] /= self.numItersIM

                tmp_minus[:,s] = tmp_minus[:,s] - ref_slopes[:,s]


            #Compute the normalized difference
            self.IM_cube[:,:,i] = (tmp_plus-tmp_minus)/(2*self.pokeAmp)

        return


    def getTRSlopes(self):
        '''
        Get the slopes for every frame position
        '''
        self.fsm.currentPos = None
        signal_TR =  np.zeros((self.signalSize, 48))
        for s in range(self.numFrames):
            self.fsm.step()
            signal_TR[:,s] = self.wfsShm.read()
        return signal_TR



    def timeResolvedIntegratorWithTurbulence(self):

        if self.turbulenceGenerator != None:
            self.turbModes = self.turbulenceGenerator.getNextTurbAsModes()

        else:
            self.turbModes = 0

        slopes_TR = self.getTRSlopes()

        currentCorrection = self.wfcShm.read()
        newCorrection = updateCorrectionTR(correction=currentCorrection, 
                                           gCM=self.gCM, 
                                           slopes_TR=slopes_TR,
                                           weights=self.frame_weights)
        newCorrection[self.numActiveModes:] = 0
        print(f"Current correction = {newCorrection}")
        if self.turbulenceGenerator != None:
            print(f"Turb : {self.turbModes}")

        self.wfcShm.write(newCorrection + self.turbModes)


    def computeIM(self):
        self.pushPullIM_cube()

        self.IM = np.sum(self.IM_cube * self.frame_weights[np.newaxis, :, :], axis=1)

        self.computeCM()
        return


    def plotWeights(self):
        plt.figure()
        im1 = plt.imshow(self.frame_weights)
        plt.colorbar(im1)
        plt.title("Measured weights for each modulation frame and KL mode")
        plt.ylabel("Modulation Frame")
        plt.xlabel("KL mode")
        plt.show()




