from pyRTC.Loop import *
#from scripts.modulation_weights import *
from pyRTC.LoopWithRemoteWFC import *
import pickle

@jit(nopython=True)
def updateCorrectionTR(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    signal_per_mode = slopes_TR @ weights 
    signal_per_mode_normed = signal_per_mode / np.sum(signal_per_mode, axis=0)
    #TODO Might be able to optimize this with einsum
    #new_corr = np.diag(gCM.astype(np.float64) @ (signal_per_mode_normed - ref_signal_per_mode_normed))
    nModes = weights.shape[1]
    new_corr = np.array([np.dot(gCM.astype(np.float64)[i,:],  (signal_per_mode_normed - ref_signal_per_mode_normed)[:,i]) for  i in range(nModes)])

    return correction - new_corr


@jit(nopython=True)
def calc_TR_Residual(CM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    signal_per_mode = slopes_TR @ weights 
    signal_per_mode_normed = signal_per_mode / np.sum(signal_per_mode, axis=0)
    #TODO Might be able to optimize this with einsum
    #new_corr = np.diag(gCM.astype(np.float64) @ (signal_per_mode_normed - ref_signal_per_mode_normed))
    nModes = weights.shape[1]
    new_corr = np.array([np.dot(CM.astype(np.float64)[i,:],  (signal_per_mode_normed - ref_signal_per_mode_normed)[:,i]) for  i in range(nModes)])

    return new_corr


@jit(nopython=True)
def updateCorrectionTRFF_no_zero(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       ref_signal_normed=np.array([[]], dtype=np.float64),):
    signal_normed = slopes_TR / np.sum(slopes_TR)
    #TODO Might be able to optimize this with einsum
    signal_tmp = signal_normed - ref_signal_normed
    for i in range(signal_tmp.shape[0]):
        if signal_normed[i] == 0:
            signal_tmp[i] = 0
    new_corr = gCM.astype(np.float64) @ (signal_tmp)
    return correction - new_corr



@jit(nopython=True)
def updateCorrectionTR_no_zero(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    signal_per_mode = slopes_TR @ weights 
    signal_per_mode_normed = signal_per_mode / np.sum(signal_per_mode, axis=0)
    #TODO Might be able to optimize this with einsum
    #new_corr = np.diag(gCM.astype(np.float64) @ (signal_per_mode_normed - ref_signal_per_mode_normed))
    nModes = weights.shape[1]
    signal_masks = signal_per_mode_normed > 0
    new_corr = np.zeros(nModes)
    for i in range(nModes):
        tmp = np.zeros(signal_per_mode_normed.shape[0])
        mask_for_mode = signal_masks[:,i]
        tmp[mask_for_mode] = (signal_per_mode_normed[:,i])[mask_for_mode] - (ref_signal_per_mode_normed[:,i])[mask_for_mode]
        new_corr[i] = np.dot(gCM.astype(np.float64)[i,:], tmp)
    return correction - new_corr

@jit(nopython=True)
def calc_TRFF_residual(CM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       ref_signal_normed=np.array([[]], dtype=np.float64),):
    signal_normed = slopes_TR / np.sum(slopes_TR)
    #TODO Might be able to optimize this with einsum
    new_corr = CM.astype(np.float64) @ (signal_normed - ref_signal_normed)
    return new_corr


@jit(nopython=True)
def updateCorrectionTRFF_weighted(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    nModes= weights.shape[1]
    new_corr = np.zeros(gCM.shape[0], dtype=np.float64)
    for mode in range(nModes):
        signal_for_mode = (slopes_TR[:,:] * weights[np.newaxis, :, mode]).flatten()
        if np.sum(signal_for_mode) != 0:
            signal_for_mode /= np.sum(signal_for_mode)

        signal_final = signal_for_mode - ref_signal_per_mode_normed[:,mode]
        #new_corr[mode] = np.dot(gCM[mode, :],signal_final[:])
        for k in range(gCM.shape[1]):
           if signal_for_mode[k] != 0:
               new_corr[mode] += gCM[mode, k] * signal_final[k]

    return correction - new_corr


@jit(nopython=True)
def calc_TRFF_residual_weighted(CM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       weights=np.array([[]], dtype=np.float64),
                       ref_signal_per_mode_normed=np.array([[]], dtype=np.float64),):
    signal_per_mode =(slopes_TR[:,:,np.newaxis] * weights[np.newaxis, :, :]).reshape(-1, weights.shape[-1]) 
    signal_per_mode_normed = signal_per_mode / np.sum(signal_per_mode, axis=0)
    #TODO Might be able to optimize this with einsum
    nModes = weights.shape[1]
    new_corr = np.array([np.dot(CM.astype(np.float64)[i,:],  (signal_per_mode_normed - ref_signal_per_mode_normed)[:,i]) for  i in range(nModes)])
    return new_corr

class TimeResolvedLoopWithRemoteWFC(pyRTCComponent):


    def __init__(self, conf,  remoteWFC, settings_name="loop" ) -> None:
        
        #Initialize the pyRTC Loop super class
        self.confWFS = conf["trwfs"]
        self.confWFC = conf["wfc"]
        self.confLoop = conf[settings_name]
        self.name = settings_name

        self.numFrames  = conf['trwfs']['nFrames']
        
        #Read wfs signal's metadata and open a stream to the shared memory
        self.signalMeta = ImageSHM("signal_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signalDType = float_to_dtype(self.signalMeta[3])
        self.signalSize = int(self.signalMeta[2]//self.signalDType.itemsize) //  self.numFrames 
        self.signalShm = ImageSHM("signal", (self.numFrames ,self.signalSize  ), self.signalDType)
        self.nullSignal = np.zeros(( self.numFrames, self.signalSize ), dtype=self.signalDType)

        #Read wfs SLOPES metadata and open a stream to the shared memory
        self.signal2DMeta = ImageSHM("signal2D_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signal2DDType = float_to_dtype(self.signal2DMeta[3])
        self.signal2DSize = int(self.signal2DMeta[2]//self.signal2DDType.itemsize)
        self.signal2D_width, self.signal2D_height = int(self.signal2DMeta[4]),  int(self.signal2DMeta[5])
        print(self.signal2DMeta[3], (self.signal2D_width, self.signal2D_height), self.signal2DDType)
        self.signal2DShm = ImageSHM("signal2D", (self.signal2D_width, self.signal2D_height), self.signal2DDType)

        self.remoteWFC = remoteWFC
        self.numModes = self.confWFC['numModes']

        self.numDroppedModes = setFromConfig(self.confLoop, "numDroppedModes", 0)
        self.numActiveModes = self.numModes - self.numDroppedModes
        self.flat = np.zeros(self.numModes, dtype=np.float32)

        self.gain = setFromConfig(self.confLoop, "gain", 0.1)
        self.leakyGain = setFromConfig(self.confLoop, "leakyGain", 0)
        self.perturbAmp = 0
        self.hardwareDelay = setFromConfig(self.confWFC, "hardwareDelay", 0)
        self.pokeAmp = setFromConfig(self.confLoop, "pokeAmp", 1e-2)
        self.numItersIM = setFromConfig(self.confLoop, "numItersIM", 100) 
        self.delay = setFromConfig(self.confLoop, "delay", 0)
        self.IMMethod = setFromConfig(self.confLoop, "IMMethod", "push-pull") 
        self.IMFile = setFromConfig(self.confLoop, "IMFile", "")
        
        self.IM_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        self.push_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        self.pull_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)

        self.push_pull_cube_file = setFromConfig(self.confLoop, "pushPullFile", "")
        self.leakyGain = 0.0
        
        self.currentCorrection = np.zeros((self.numModes))
        self.newCorrection_tmp_delay_1 =  np.zeros((self.numModes))
        self.signal_TR_ref =  np.zeros((self.signalSize, self.numFrames))

        self.first_loop = True

        self.delay = 0

        self.ref_signal_per_mode_normed = None
        self.ref_signal_normed = None
        self.frame_weights = np.ones((self.numFrames,self.numModes))

        self.FF_active= False
        self.FF_weighted_active= False


        self.FF_w_correction_function = updateCorrectionTRFF_weighted
        self.FF_correction_function =  updateCorrectionTRFF_no_zero
        self.TR_norm_correction_function = updateCorrectionTR_no_zero

        self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
        self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.signalDType)


        self.loadPushPullCube()

        super().__init__(self.confLoop)    


    def setGain(self, gain):
        self.gain = gain
        self.gCM = self.gain*self.CM
        return
    
    def flatten(self):
        #self.wfcShm.write(self.flat)
        self.remoteWFC.run("flatten")
        return
    
    def computeCM(self):
        self.numActiveModes = self.numModes-self.numDroppedModes
        if self.numActiveModes < 0:
            print("Invalid Number of Modes used in CM. Check numDroppedModes")
            return
        self.CM[:self.numActiveModes,:] = np.linalg.pinv(self.IM[:,:self.numActiveModes], rcond=0)
        self.CM[self.numActiveModes:,:] = 0
        self.gCM = self.gain*self.CM
        self.fIM = np.copy(self.IM)
        self.fIM[:,self.numActiveModes:] = 0
        return 

    def setPeturbAmp(self, amp):
        self.perturbAmp = amp
        return

    def convertForTransmission(self, data):
        return (data*1e9).astype(np.int32)

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

    def saveIMCube(self, filename):
         data = {"push": self.push_cube,
                 "pull": self.pull_cube,
                 "ref" : self.ref_slopes,
                 "pokeAmp": self.pokeAmp,
                 "weights": self.frame_weights}
         with open(filename, 'wb') as f: 
              pickle.dump(data, f)

    def loadPushPullCube(self,filename=''):
        if filename == '':
            filename = self.push_pull_cube_file
        if filename != '':
            with open(filename, 'rb') as f: 
                data = pickle.load(f)
                self.push_cube = data["push"]
                self.pull_cube = data["pull"]
                self.ref_slopes = data["ref"]
                self.pokeAmp = data["pokeAmp"]
                self.frame_weights = data["weights"]

            self.makeIM(self.push_cube,
                        self.pull_cube,  
                        self.ref_slopes,
                        self.pokeAmp,
                        self.frame_weights)

            self.computeCM()

    def findModeOrder(self, modeNumber):
        return 1
        # order = 1
        # rangeList = [0,1,2]
        # done = False
        # while not done:
        #     if modeNumber in rangeList:
        #         done = True
        #     else:
        #         rangeList = list(range(np.max(rangeList)+1,np.max(rangeList)+len(rangeList)+3))
        #         order +=1
        # return order


    def pushPullRef_cube(self, maxNumModes=None):
        
        if maxNumModes is None:
            maxNumModes = self.numModes
        if maxNumModes > self.numModes:
            maxNumModes = self.numModes

        self.ref_slopes = np.zeros((self.signalSize, self.numFrames))
        #Average out N new WFS frames
        self.ref_slopes=  np.zeros((self.signalSize, self.numFrames))
        for n in range(self.numItersIM):
            self.ref_slopes += self.signalShm.read().T
        self.ref_slopes /= self.numItersIM

        #For each mode
        for i in range(maxNumModes):

            currentModePokeAmp = self.pokeAmp /np.sqrt(self.findModeOrder(i))
            print(f"pushPullRef_cube - Mode {i}/{maxNumModes}, with pokeAmp={currentModePokeAmp}")
            #Reset the correction
            correction = self.flat.copy()
            #Plus amplitude
            correction[i] = currentModePokeAmp
            #Post a new shape to be made
            self.remoteWFC.run("write", self.convertForTransmission(correction))
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()

            self.tmp_plus =  np.zeros((self.signalSize, self.numFrames))
            #Average out N new WFS frames
            for n in range(self.numItersIM):
                self.tmp_plus += self.signalShm.read().T
            self.tmp_plus /= self.numItersIM

            #minus amplitude
            correction[i] = -currentModePokeAmp
            #Post a new shape to be made
            self.remoteWFC.run("write", self.convertForTransmission(correction))
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()


            self.tmp_minus =  np.zeros((self.signalSize, self.numFrames))
            #Average out N new WFS frames
            for n in range(self.numItersIM):
                self.tmp_minus += self.signalShm.read().T
            self.tmp_minus /= self.numItersIM


            #Compute the normalized difference
            #self.IM_cube[:,:,i] = (tmp_plus-tmp_minus)/(2*self.pokeAmp)

            self.push_cube[:,:,i] = self.tmp_plus
            self.pull_cube[:,:,i] = self.tmp_minus

        return


    def getTRSlopes(self):
        '''
        Get the slopes for every frame position
        '''
        signal_TR =  np.zeros((self.signalSize, self.numFrames))
        signal_TR = self.signalShm.read().T - self.signal_TR_ref
        return signal_TR


    def grabRefTRSlopes(self):
        '''
        Get the slopes for every frame position and use that as the ref slopes
        '''
        self.signal_TR_ref =  np.zeros((self.signalSize, self.numFrames))
        for i in range(10):
            self.signal_TR_ref += self.signalShm.read().T
        self.signal_TR_ref /= 10


    def setDelay(self, newDelay):
        if newDelay != 0:
            self.delayed_signal = np.zeros((newDelay, self.signalSize, self.numFrames))
            self.delay = newDelay
        else:
            self.delay = newDelay


    def timeResolvedIntegratorWithLeak(self):

        self.currentCorrection = (1-self.leakyGain)*np.array(self.remoteWFC.getProperty("currentCorrection"))

        slopes_TR = self.getTRSlopes()
        self.latest_slopes = slopes_TR
        # Remove this next line because it would grab the current correction AND turbulence applied to the DM 
        #currentCorrection = self.wfcShm.read()

        if self.FF_active:
            if self.ref_signal_normed is not None:
                newCorrection = updateCorrectionTRFF_no_zero(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes.flatten(),
                                                ref_signal_normed = self.ref_signal_normed)
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        elif self.FF_weighted_active:
            if self.ref_signal_per_mode_normed is not None:
                newCorrection = updateCorrectionTRFF_weighted(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes,
                                                weights=self.frame_weights,
                                                ref_signal_per_mode_normed = self.ref_signal_per_mode_normed)
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        else:
            if self.ref_signal_per_mode_normed is not None:
                newCorrection = self.TR_norm_correction_function(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes,
                                                weights=self.frame_weights,
                                                ref_signal_per_mode_normed = self.ref_signal_per_mode_normed)
            else:
                print("Error: weighted ref signal never defined, skipping loop")
                return
        newCorrection[self.numActiveModes:] = 0

        if np.isnan(newCorrection).any(): 
            self.currentCorrection = self.currentCorrection # dont change correction due to nan 
        else:
            self.currentCorrection = newCorrection # Safe to update
        self.remoteWFC.run("write", self.convertForTransmission(self.currentCorrection))



    def timeResolvedIntegratorWithTurbulence(self):

        if self.turbulenceGenerator != None:
            self.turbModes = self.turbulenceGenerator.getNextTurbAsModes()

        else:
            self.turbModes = 0

        if self.first_loop or (self.delay == 0):
            slopes_TR = self.getTRSlopes()
            self.latest_slopes = slopes_TR
            for i in range(self.delay):
                self.delayed_signal[i,:, :] = self.latest_slopes
            self.first_loop = False

        # Remove this next line because it would grab the current correction AND turbulence applied to the DM 
        #currentCorrection = self.wfcShm.read()

        if self.FF_active:
            if self.ref_signal_normed is not None:
                newCorrection = self.updateCorrectionTRFF_no_zero(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes.flatten(),
                                                ref_signal_normed = self.ref_signal_normed)
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        elif self.FF_weighted_active:
            if self.ref_signal_per_mode_normed is not None:
                newCorrection = self.updateCorrectionTRFF_weighted(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes,
                                                weights=self.frame_weights,
                                                ref_signal_per_mode_normed = self.ref_signal_per_mode_normed)
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        else:
            if self.ref_signal_per_mode_normed is not None:
                newCorrection = self.TR_norm_correction_function(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes,
                                                weights=self.frame_weights,
                                                ref_signal_per_mode_normed = self.ref_signal_per_mode_normed)
            else:
                print("Error: weighted ref signal never defined, skipping loop")
                return
        newCorrection[self.numActiveModes:] = 0
        self.latest_correction = newCorrection
        #print(f"Current correction = {newCorrection}")
        #if self.turbulenceGenerator != None:
        #    print(f"Turb : {self.turbModes}")
        if self.delay != 0:
            self.delayed_signal = np.roll(self.delayed_signal, 1, axis=0)
            self.delayed_signal[0,:, :] = self.getTRSlopes()
            self.latest_slopes = self.delayed_signal[-1,:, :]
        
        # Instead keep track of the currentCorrection manually instead of fetching from DM 
        #self.currentCorrection = self.newCorrection_tmp_delay_1
        #self.newCorrection_tmp_delay_1 = newCorrection

        if np.isnan(newCorrection).any(): 
            self.currentCorrection = self.currentCorrection # dont change correction due to nan 
        else:
            self.currentCorrection = newCorrection # Safe to update
        self.remoteWFC.run("write", self.convertForTransmission(self.currentCorrection + self.turbModes))

    def resetCurrentCorrection(self):
        self.currentCorrection = np.zeros((self.numModes))
        self.newCorrection_tmp_delay_1 = np.zeros((self.numModes))
        self.first_loop = True


    def makeIM(self, push, pull, ref, poke, weights):
        self.IM       = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
        push_weighted = np.sum(push * weights[np.newaxis, :, :], axis=1)
        pull_weighted = np.sum(pull * weights[np.newaxis, :, :], axis=1)
        ref_weighted  = ref @ weights
        for mode in range(self.numModes):
            push_signal      = ((push_weighted[:,mode]/np.sum(push_weighted[:,mode])) - (ref_weighted[:,mode]/np.sum(ref_weighted[:,mode])))
            pull_signal      = ((pull_weighted[:,mode]/np.sum(pull_weighted[:,mode])) - (ref_weighted[:,mode]/np.sum(ref_weighted[:,mode])))
            self.IM[:,mode]  = (push_signal - pull_signal) / (2*(poke/np.sqrt(self.findModeOrder(mode))))

        self.ref_signal_per_mode_normed = (ref_weighted ) / np.sum(ref_weighted, axis=0)


    def modWeightsFromPushPullRef(self, push, pull, ref, pokeAmp):

        numFrames =push.shape[1]
        maxNumModes = push.shape[2]
        weighting_cube = np.zeros((numFrames, maxNumModes))
        for i in range(maxNumModes):
            signal_push = (push[:,:,i]/np.sum(push[:,:,i], axis=0)) - (ref/np.sum(ref))
            signal_pull = (pull[:,:,i]/np.sum(pull[:,:,i], axis=0)) - (ref/np.sum(ref))
            total = (signal_push - signal_pull) / (2*(pokeAmp/np.sqrt(self.findModeOrder(i))))
            avg_val = np.mean(total, axis=0)
            weighting_cube[:,i] = np.sqrt(((np.mean((total-avg_val[np.newaxis,:])**2, axis=0))))
            weighting_cube[:,i] = (weighting_cube[:,i]  / np.sum(np.abs(weighting_cube[:,i])))*numFrames

        return weighting_cube

    def computeIM(self):
        self.pushPullRef_cube()
        
        weighting_cube = self.modWeightsFromPushPullRef(self.push_cube,
                                                   self.pull_cube,  
                                                   self.ref_slopes,
                                                   self.pokeAmp)

        #weighting_cube = modWeightsFromIMCube(im_cube=self.IM_cube)
        self.frame_weights[:,:self.numModes] = weighting_cube

        self.makeIM(self.push_cube,
                    self.pull_cube,  
                    self.ref_slopes,
                    self.pokeAmp,
                    self.frame_weights)

        self.computeCM()
        return
    

    def changeWeightsAndUpdate(self, newWeights):
        self.frame_weights[:,:self.numModes] = newWeights

        self.makeIM(self.push_cube,
                    self.pull_cube,  
                    self.ref_slopes,
                    self.pokeAmp,
                    self.frame_weights)

        self.computeCM()

    def switchToFF(self):
        self.FF_active= True
        self.FF_weighted_active= False

        self.CM = np.zeros((self.numModes, self.signalSize*self.numFrames),dtype=self.signalDType)
        self.IM       = np.zeros((self.signalSize*self.numFrames, self.numModes),dtype=self.signalDType)
        #push_flat = self.push_cube.reshape(-1, self.push_cube.shape[-1])
        #pull_flat = self.pull_cube.reshape(-1, self.pull_cube.shape[-1])
        push_flat = self.push_cube
        pull_flat = self.pull_cube
        ref_flat  = self.ref_slopes
        
        for mode in range(self.numModes):
            IM_tmp       = np.zeros((self.signalSize, self.numFrames),dtype=self.signalDType)
            for f in range(self.numFrames):
                push_signal      = push_flat[:,f,mode]/(np.sum(push_flat[:,f,mode])* self.numFrames)
                pull_signal      = pull_flat[:,f, mode]/(np.sum(pull_flat[:,f,mode])* self.numFrames)

                if isinstance(self.pokeAmp, float):
                    IM_tmp[:,f]  = (push_signal - pull_signal) / (2*(self.pokeAmp/np.sqrt(self.findModeOrder(mode))))
                else:
                    IM_tmp[:,f]  = (push_signal - pull_signal) / (2*self.pokeAmp[mode])
            self.IM[:, mode] = IM_tmp.flatten()

        ref_signal_normed_tmp = np.zeros((self.signalSize, self.numFrames))
        for f in range(self.numFrames):
            ref_signal_normed_tmp[:,f] = (ref_flat[:,f]/(np.sum(ref_flat[:,f])* self.numFrames))
                

        self.ref_signal_normed = ref_signal_normed_tmp.flatten()

        self.computeCM()




    def switchToFFwithWeights(self):
        self.FF_weighted_active= True
        self.FF_active= False

        self.CM = np.zeros((self.numModes, self.signalSize*self.numFrames),dtype=self.signalDType)

        self.IM  = np.zeros((self.signalSize*self.numFrames, self.numModes),dtype=self.signalDType)
        # push_flat = self.push_cube.reshape(-1, self.push_cube.shape[-1])
        # pull_flat = self.pull_cube.reshape(-1, self.pull_cube.shape[-1])
        ref_weighted  = (self.ref_slopes[:,:,np.newaxis] * self.frame_weights[np.newaxis, :, :]).reshape(-1, self.frame_weights.shape[-1]) 
        for mode in range(self.numModes):
            push_signal = ((self.push_cube[:,:,mode]*self.frame_weights[np.newaxis, :, mode]).flatten()/np.sum(self.push_cube[:,:,mode]*self.frame_weights[np.newaxis,:, mode]))
            pull_signal = ((self.pull_cube[:,:,mode]*self.frame_weights[np.newaxis, :, mode]).flatten()/np.sum(self.pull_cube[:,:,mode]*self.frame_weights[np.newaxis,:, mode]))
            if isinstance(self.pokeAmp, float):
                self.IM[:,mode]  = (push_signal - pull_signal) / (2*(self.pokeAmp/np.sqrt(self.findModeOrder(mode))))
            else:
                self.IM[:,mode]  = (push_signal - pull_signal) / (2*self.pokeAmp[mode])

        self.ref_signal_per_mode_normed = (ref_weighted ) / np.sum(ref_weighted, axis=0)

        self.computeCM()



    def plotWeights(self):
        plt.figure()
        im1 = plt.imshow(self.frame_weights)
        plt.colorbar(im1)
        plt.title("Measured weights \n for each modulation frame and KL mode")
        plt.ylabel("Modulation Frame")
        plt.xlabel("KL mode")
        plt.show()


