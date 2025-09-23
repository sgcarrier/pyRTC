from pyRTC.Loop import *
from scripts.modulation_weights import *
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
def updateCorrectionTRFF(correction=np.array([], dtype=np.float64), 
                       gCM=np.array([[]], dtype=np.float64),  
                       slopes_TR=np.array([[]], dtype=np.float64),
                       ref_signal_normed=np.array([[]], dtype=np.float64),):
    signal_normed = slopes_TR / np.sum(slopes_TR)
    #TODO Might be able to optimize this with einsum
    new_corr = gCM.astype(np.float64) @ (signal_normed - ref_signal_normed)
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

class TimeResolvedLoopWithRemoteWFC(LoopWithRemoteWFS):


    def __init__(self, conf,  settings_name="loop" ) -> None:
        #Initialize the pyRTC Loop super class
        super().__init__(conf, settings_name=settings_name)

        self.numFrames  = conf['trwfs']['nFrames']
        #self.weightFile = setFromConfig(self.confLoop, "weightFile", "")
        #self.loadWeights()


        self.IM_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        self.push_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)
        self.pull_cube = np.zeros((self.signalSize, self.numFrames, self.numModes),dtype=self.signalDType)

        self.push_pull_cube_file = setFromConfig(self.confLoop, "pushPullFile", "")

        

        self.currentCorrection = np.zeros((self.numModes))
        self.newCorrection_tmp_delay_1 =  np.zeros((self.numModes))
        self.signal_TR_ref =  np.zeros((self.signalSize, 48))

        self.first_loop = True


        self.delay = 0

        self.ref_signal_per_mode_normed = None
        self.ref_signal_normed = None
        self.frame_weights = np.ones((self.numFrames,self.numModes))

        self.FF_active= False
        self.FF_weighted_active= False


        self.FF_w_correction_function = updateCorrectionTRFF_weighted
        self.FF_correction_function =  updateCorrectionTRFF
        self.TR_norm_correction_function = updateCorrectionTR



        self.loadPushPullCube()



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
        order = 1
        rangeList = [0,1,2]
        done = False
        while not done:
            if modeNumber in rangeList:
                done = True
            else:
                rangeList = list(range(np.max(rangeList)+1,np.max(rangeList)+len(rangeList)+3))
                order +=1
        return order



    def pushPullIM_cube(self, maxNumModes=None):

        if maxNumModes is None:
            maxNumModes = self.numModes
        if maxNumModes > self.numModes:
            maxNumModes = self.numModes

        ref_slopes = np.zeros((self.signalSize, self.numFrames))


        #Average out N new WFS frames
        ref_slopes[:,s] =  np.zeros((self.signalSize))
        for n in range(self.numItersIM):
            ref_slopes[:,:] += self.signalShm.read().T
        ref_slopes[:,:] /= self.numItersIM

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
            self.signalShm.read()

            tmp_plus =  np.zeros((self.signalSize, self.numFrames))
   
            #Average out N new WFS frames
            for n in range(self.numItersIM):
                tmp_plus[:,:] += self.signalShm.read().T
            tmp_plus[:,:] /= self.numItersIM
            tmp_plus = tmp_plus - ref_slopes

            #minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()

            tmp_minus =  np.zeros((self.signalSize, self.numFrames))

            #Average out N new WFS frames
            for n in range(self.numItersIM):
                tmp_minus += self.signalShm.read().T
            tmp_minus /= self.numItersIM

            tmp_minus = tmp_minus - ref_slopes

            #Compute the normalized difference
            self.IM_cube[:,:,i] = (tmp_plus-tmp_minus)/(2*self.pokeAmp)

        return

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
            self.wfcShm.write(correction)
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
            self.wfcShm.write(correction)
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
                newCorrection = self.FF_correction_function(correction=self.currentCorrection,
                                                gCM=self.gCM, 
                                                slopes_TR=self.latest_slopes.flatten(),
                                                ref_signal_normed = self.ref_signal_normed)
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        elif self.FF_weighted_active:
            if self.ref_signal_per_mode_normed is not None:
                newCorrection = self.FF_w_correction_function(correction=self.currentCorrection,
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
        self.wfcShm.write(self.currentCorrection + self.turbModes)

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



    # def calcFrameWeights(self, maxNumModes=None):
    #     '''
    #     Calculate the weights to give to every frame for every mode
    #     '''
    #     self.pushPullIM_cube(maxNumModes=maxNumModes)

    #     weighting_cube = modWeightsFromIMCube(im_cube=self.IM_cube)

    #     if maxNumModes is None:
    #         maxModes = self.numModes
    #     elif maxNumModes > self.numModes:
    #         maxModes = self.numModes

    #     self.frame_weights[:,:maxModes] = weighting_cube
