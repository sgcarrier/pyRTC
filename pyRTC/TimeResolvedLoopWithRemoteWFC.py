from pyRTC.Loop import *
#from scripts.modulation_weights import *
from pyRTC.LoopWithRemoteWFC import *
import pickle
import time
import ctypes


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
        #self.IMMethod = setFromConfig(self.confLoop, "IMMethod", "push-pull") 
        #self.IMFile = setFromConfig(self.confLoop, "IMFile", "")
        self.REFFile = setFromConfig(self.confLoop, "REFFile", "")
        
        self.IM_cube = np.zeros(( self.numFrames, self.signalSize, self.numModes),dtype=self.signalDType)
        self.push_cube = np.zeros(( self.numFrames, self.signalSize, self.numModes),dtype=self.signalDType)
        self.pull_cube = np.zeros(( self.numFrames, self.signalSize, self.numModes),dtype=self.signalDType)

        self.push_pull_cube_file = setFromConfig(self.confLoop, "pushPullFile", "")
        self.leakyGain = 0.0
        
        self.currentCorrection = np.zeros((self.numModes))
        self.newCorrection_tmp_delay_1 =  np.zeros((self.numModes))
        self.signal_TR_ref =  np.zeros((self.numFrames, self.signalSize))

        self.first_loop = True

        self.delay = 0

        self.ref_signal_per_mode_normed = None
        self.ref_signal_normed = None
        self.frame_weights = np.ascontiguousarray(np.ones((self.numFrames,self.numModes), dtype=np.float32))

        self.FF_active= False
        self.FF_weighted_active= False

        self.IM = np.ascontiguousarray(np.zeros((self.signalSize, self.numModes),dtype=self.signalDType))
        self.CM = np.ascontiguousarray(np.zeros((self.numModes, self.signalSize),dtype=self.signalDType))

        self.loadCLibAndFunctions()

        self.loadPushPullCube()

        self.aimedLoopPeriod = 0.002    # Default 500 Hz
        self.currentLoopPeriod = 0

        self.loopCounterLimit= 1000
        self._loop_counter = 0
        self._loop_delay = 0

        self.manual_cam_obj = None
        self.left_limit = 0
        self.right_limit = 47

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



    def loadCLibAndFunctions(self):
        # Load the shared library
        self.lib = ctypes.CDLL('res/trpwfs_lib.dll')

        # Define the C function's argument types and return type
        self.lib.trpwfs_FF_W_calc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.trpwfs_FF_W_calc.restype = None  # No return value

        self.lib.trpwfs_FF_calc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.trpwfs_FF_W_calc.restype = None  # No return value


        self.lib.trpwfs_TR_calc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.trpwfs_TR_calc.restype = None  # No return value

        self.lib.matmul_sum_axis0.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.matmul_sum_axis0.restype = None

        self.lib.trpwfs_NORM_calc.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.trpwfs_NORM_calc.restype = None


        self.FF_w_correction_function = self.lib.trpwfs_FF_W_calc
        self.FF_correction_function =  self.lib.trpwfs_FF_calc
        self.TR_norm_correction_function = self.lib.trpwfs_TR_calc
        self.classic_norm_correction_function = self.lib.trpwfs_NORM_calc


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
                self.frame_weights = np.ascontiguousarray(data["weights"], dtype=np.float32)

            if self.REFFile != '':
                with open(filename, 'rb') as f: 
                    data = pickle.load(f)
                    self.ref_slopes = np.ascontiguousarray(data["ref"], dtype=np.float32)

            self.makeIM(self.push_cube,
                        self.pull_cube,  
                        self.ref_slopes,
                        self.pokeAmp,
                        self.frame_weights)

            self.computeCM()



    def findModeOrder(self, modeNumber):
        order = 1
        #rangeList = [0,1,2]
        #done = False
        #while not done:
        #    if modeNumber in rangeList:
        #        done = True
        #    else:
        #        rangeList = list(range(np.max(rangeList)+1,np.max(rangeList)+len(rangeList)+3))
        #        order +=1
        return order


    def pushPullRef_cube(self, maxNumModes=None):

        #TODO make bypass 
        
        if maxNumModes is None:
            maxNumModes = self.numModes
        if maxNumModes > self.numModes:
            maxNumModes = self.numModes

        #Average out N new WFS frames
        self.ref_slopes=  np.ascontiguousarray(np.zeros((self.numFrames, self.signalSize)), dtype=np.float32)
        for n in range(self.numItersIM):
            self.ref_slopes += self.getTRSlopes_bypass()
            time.sleep(1e-3)
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
            for j in range(20):
                self.getTRSlopes_bypass()
                time.sleep(1e-3)

            self.tmp_plus =  np.zeros((self.numFrames, self.signalSize))
            #Average out N new WFS frames
            for n in range(self.numItersIM):
                self.tmp_plus += self.getTRSlopes_bypass()
                time.sleep(1e-3)
            self.tmp_plus /= self.numItersIM

            #minus amplitude
            correction[i] = -currentModePokeAmp
            #Post a new shape to be made
            self.remoteWFC.run("write", self.convertForTransmission(correction))
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            for j in range(20):
                self.getTRSlopes_bypass()
                time.sleep(1e-3)


            self.tmp_minus =  np.zeros((self.numFrames, self.signalSize))
            #Average out N new WFS frames
            for n in range(self.numItersIM):
                self.tmp_minus += self.getTRSlopes_bypass()
                time.sleep(1e-3)
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
        #signal_TR = np.ascontiguousarray(np.zeros((self.signalSize, self.numFrames),order='C', dtype=np.float32))
        signal_TR = np.ascontiguousarray(self.signalShm.read() - self.signal_TR_ref, dtype=np.float32)
        return signal_TR
    
    def getTRSlopes_bypass(self):
        '''
        Get the slopes for every frame position
        '''
        if self.manual_cam_obj:
        #signal_TR = np.ascontiguousarray(np.zeros((self.signalSize, self.numFrames),order='C', dtype=np.float32))
            signal_TR = np.ascontiguousarray((self.manual_cam_obj.data[:,:,self.left_limit:self.right_limit]).reshape(self.numFrames, -1), dtype=np.float32)
        return signal_TR


    def grabRefTRSlopes(self):
        '''
        Get the slopes for every frame position and use that as the ref slopes
        '''
        self.signal_TR_ref = np.ascontiguousarray(np.zeros((self.numFrames, self.signalSize), dtype=np.float32))
        for i in range(10):
            self.signal_TR_ref += self.signalShm.read()
        self.signal_TR_ref /= 10


    def setDelay(self, newDelay):
        if newDelay != 0:
            self.delayed_signal = np.zeros((newDelay, self.signalSize, self.numFrames))
            self.delay = newDelay
        else:
            self.delay = newDelay


    def timeResolvedIntegratorWithLeak_C(self):

        if self._loop_counter == 0:
            self._start_timer = time.perf_counter()


        if self.first_loop:
            newCorrection = np.ascontiguousarray((1-self.leakyGain)*np.array(self.remoteWFC.getProperty("currentCorrection")), dtype=np.float32)
            self.first_loop = False
            self.latest_slopes = np.zeros_like(self.getTRSlopes_bypass())
        else:
            newCorrection = np.ascontiguousarray((1-self.leakyGain)* self._local_currentCorrection, dtype=np.float32)
        #self.currentCorrection = np.ascontiguousarray((1-self.leakyGain)*np.array(self.remoteWFC.getProperty("currentCorrection")))

        #newCorrection = self.currenCorrection.copy()
        # Remove this next line because it would grab the current correction AND turbulence applied to the DM

        #self.latest_slopes = self.getTRSlopes()
        #CORRECT = True
        if not np.all(self.getTRSlopes_bypass() == self.latest_slopes):
            self.latest_slopes = self.getTRSlopes_bypass()
            CORRECT = True
        else:
            CORRECT = False
    
        #self.latest_slopes = self.getTRSlopes()
        if CORRECT == False:
            tmp_newCorrection = newCorrection.copy()
            newCorrection *= 0

        if self.FF_active:
            if self.ref_signal_normed is not None:
                self.FF_correction_function(self.numModes, self.numFrames, self.signalSize,
                                            self.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            self.latest_slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            self.ref_signal_normed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            newCorrection.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

            else:
                print("Error: ref signal never defined, skipping loop")
                return
        elif self.FF_weighted_active:
            if self.ref_signal_per_mode_normed is not None:
                self.FF_w_correction_function(self.numModes, self.numFrames, self.signalSize,
                                              self.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                              self.latest_slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                              self.frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                              self.ref_signal_per_mode_normed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                              newCorrection.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            else:
                print("Error: ref signal never defined, skipping loop")
                return
        else:
            if self.ref_signal_per_mode_normed is not None:
                self.TR_norm_correction_function(self.numModes, self.numFrames, self.signalSize,
                                                 self.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 self.latest_slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 self.frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 self.ref_signal_per_mode_normed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 newCorrection.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            else:
                print("Error: weighted ref signal never defined, skipping loop")
                return
        newCorrection[self.numActiveModes:] = 0

        if CORRECT :
            if np.isnan(newCorrection).any():
                self.currentCorrection = self.currentCorrection # dont change correction due to nan
            else:
                self.currentCorrection = newCorrection # Safe to update
            self.remoteWFC.run("write", self.convertForTransmission(self.currentCorrection))

            self._local_currentCorrection = self.currentCorrection
        else:
            newCorrection = tmp_newCorrection.copy()

        self.accurate_delay(self._loop_delay)

        if self._loop_counter > self.loopCounterLimit:
            self._loop_counter = 0
            self._stop_timer = time.perf_counter()
            self.currentLoopPeriod = ((self._stop_timer - self._start_timer)/self.loopCounterLimit)
            self._loop_delay += np.max([0,self.aimedLoopPeriod -self.currentLoopPeriod])
            self._loop_delay = np.min([self._loop_delay, self.aimedLoopPeriod]) # Safety measure in case, the function is called manually
        else:
            self._loop_counter += 1

    def busy_wait(self):
        for i in range(1000):
            _ = i*i

    def accurate_delay(self, delay_s):
        """Function to provide an accurate time delay in seconds using a busy-wait."""
        target_time = time.perf_counter() + delay_s
        while time.perf_counter() < target_time:
            pass

    def resetCurrentCorrection(self):
        self.currentCorrection = np.zeros((self.numModes))
        self._local_currentCorrection = np.zeros((self.numModes))
        self.remoteWFC.run("flatten")
        self.first_loop = True

       
    def makeIM(self, push, pull, ref, poke, weights):
        self.IM       = np.ascontiguousarray(np.zeros((self.signalSize, self.numModes)), dtype=np.float32)
        push_weighted = np.sum(push * weights[:, np.newaxis, :], axis=0)
        pull_weighted = np.sum(pull * weights[:, np.newaxis, :], axis=0)
        ref_weighted  = (ref.T @ weights).T
        for mode in range(self.numModes):
            push_signal      = (push_weighted[:,mode] / np.sum(push_weighted[:,mode]))
            pull_signal      = (pull_weighted[:,mode] / np.sum(pull_weighted[:,mode]))
            self.IM[:,mode]  = (push_signal - pull_signal) / (2*(poke/np.sqrt(self.findModeOrder(mode))))

        self.ref_signal_per_mode_normed = np.ascontiguousarray( (ref_weighted/ np.sum(ref_weighted, axis=1)[:, np.newaxis]), dtype=np.float32)


    def modWeightsFromPushPullRef(self, push, pull, ref, pokeAmp):

        numFrames =push.shape[0]
        maxNumModes = push.shape[2]
        weighting_cube = np.zeros((numFrames, maxNumModes))
        for i in range(maxNumModes):
            signal_push = (push[:,:,i]/(np.sum(push[:,:,i], axis=1)[:, np.newaxis]))
            signal_pull = (pull[:,:,i]/(np.sum(pull[:,:,i], axis=1)[:, np.newaxis]))
            total = (signal_push - signal_pull) / (2*(pokeAmp/np.sqrt(self.findModeOrder(i))))
            avg_val = np.mean(total, axis=1)
            weighting_cube[:,i] = np.sqrt(((np.mean((total-avg_val[:, np.newaxis])**2, axis=1))))
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

    def computeCM_FF(self):
        self.numActiveModes = self.numModes-self.numDroppedModes
        if self.numActiveModes < 0:
            print("Invalid Number of Modes used in CM. Check numDroppedModes")
            return
        self.CM[:self.numActiveModes,:,:] = (np.linalg.pinv((self.IM[:,:,:self.numActiveModes]).reshape(-1, self.numActiveModes), rcond=0)).reshape(self.numActiveModes, self.numFrames, -1)
        self.CM[self.numActiveModes:,:,:] = 0
        self.gCM = np.ascontiguousarray(self.gain*self.CM, dtype=np.float32)
        self.fIM = np.copy(self.IM)
        self.fIM[:,:,self.numActiveModes:] = 0
        return


    def changeWeightsAndUpdate(self, newWeights):
        self.frame_weights[:,:self.numModes] = newWeights
        self.frame_weights[:,self.numModes:] = 0

        self.makeIM(self.push_cube,
                    self.pull_cube,
                    self.ref_slopes,
                    self.pokeAmp,
                    self.frame_weights)

        self.computeCM()

    def switchToFF(self):
        self.FF_active= True
        self.FF_weighted_active= False

        self.CM = np.ascontiguousarray(np.zeros((self.numModes, self.numFrames, self.signalSize)),  dtype=np.float32)
        self.IM = np.ascontiguousarray(np.zeros((self.numFrames, self.signalSize, self.numModes)),  dtype=np.float32)
        push_flat = self.push_cube
        pull_flat = self.pull_cube
        ref_flat  = self.ref_slopes
        
        for mode in range(self.numModes):
            IM_tmp       = np.zeros((self.numFrames, self.signalSize),dtype=self.signalDType)
            for f in range(self.numFrames):
                push_signal      = push_flat[f,:,mode]/(np.sum(push_flat[f,:,mode])* self.numFrames)
                pull_signal      = pull_flat[f,:, mode]/(np.sum(pull_flat[f,:,mode])* self.numFrames)

                if isinstance(self.pokeAmp, float):
                    IM_tmp[f,:]  = (push_signal - pull_signal) / (2*(self.pokeAmp/np.sqrt(self.findModeOrder(mode))))
                else:
                    IM_tmp[f,:]  = (push_signal - pull_signal) / (2*self.pokeAmp[mode])
            self.IM[:,:, mode] = IM_tmp.copy()

        ref_signal_normed_tmp = (ref_flat/(np.sum(ref_flat)))
        self.ref_signal_normed = np.ascontiguousarray(ref_signal_normed_tmp, dtype=np.float32)

        self.computeCM_FF()




    def switchToFFwithWeights(self):
        self.FF_weighted_active= True
        self.FF_active= False

        self.CM = np.ascontiguousarray(np.zeros((self.numModes, self.numFrames, self.signalSize)),  dtype=np.float32)
        self.IM = np.ascontiguousarray(np.zeros((self.numFrames, self.signalSize, self.numModes)),  dtype=np.float32)
        # push_flat = self.push_cube.reshape(-1, self.push_cube.shape[-1])
        # pull_flat = self.pull_cube.reshape(-1, self.pull_cube.shape[-1])
        ref_weighted  = (self.ref_slopes.T[:, :, np.newaxis] * self.frame_weights[np.newaxis, :, :])
        for mode in range(self.numModes):
            push_signal = ((self.push_cube[:,:,mode]*self.frame_weights[:, np.newaxis, mode])/np.sum(self.push_cube[:,:,mode]*self.frame_weights[:, np.newaxis, mode]))
            pull_signal = ((self.pull_cube[:,:,mode]*self.frame_weights[:, np.newaxis, mode])/np.sum(self.pull_cube[:,:,mode]*self.frame_weights[:, np.newaxis, mode]))
            if isinstance(self.pokeAmp, float):
                self.IM[:, :, mode] = (push_signal - pull_signal) / (2*(self.pokeAmp/np.sqrt(self.findModeOrder(mode))))
            else:
                self.IM[:, :, mode] = (push_signal - pull_signal) / (2*self.pokeAmp[mode])

        self.ref_signal_per_mode_normed = np.ascontiguousarray((ref_weighted.T) / np.sum(ref_weighted, axis=(0,1))[:, np.newaxis, np.newaxis], dtype=np.float32)

        self.computeCM_FF()



    def plotWeights(self):
        plt.figure()
        im1 = plt.imshow(self.frame_weights)
        plt.colorbar(im1)
        plt.title("Measured weights \n for each modulation frame and KL mode")
        plt.ylabel("Modulation Frame")
        plt.xlabel("KL mode")
        plt.show()

