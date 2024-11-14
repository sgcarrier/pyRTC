from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pylablib.devices import Andor
import argparse
import sys
import os 

class AndorIXon(WavefrontSensor):

    def __init__(self, conf) -> None:
        super().__init__(conf)

        # Open it with no temperature control, no gain and the fan on
        self.cam = Andor.AndorSDK2Camera(idx=conf["idx"], temperature="off", fan_mode="low", amp_mode=None)
        time.sleep(0.1)
        #Turnoff cooler
        if self.cam.is_cooler_on():
            print("WARNING: cooler is on, turning it off")
            self.cam.set_cooler(on=False)


        self.total_photon_flux = 0
        self.activateNoise = False

        self.random_state_photon_noise = np.random.default_rng(seed=int(time.time()))

        if "exposure" in conf:
            self.setExposure(conf["exposure"])

    def open_shutter(self):
        self.cam.setup_shutter("open")
    def close_shutter(self):
        self.cam.setup_shutter("closed")

    def close_camera(self):
        self.close_shutter()
        self.cam.close()
        


    def setExposure(self, exposure):
        super().setExposure(exposure)
        self.cam.set_exposure(self.exposure)
        return

    def sample_image_events(self, original_image, num_events):
        # Flatten the original image to create a 1D probability distribution
        flat_image = original_image.flatten()    # Normalize the flattened image to create a probability distribution (sum to 1)
        prob_dist = flat_image / flat_image.sum()    # Sample indices from the flattened image based on the probability distribution
        sampled_indices = np.random.choice(len(flat_image), size=num_events, p=prob_dist)    # Create a new image with the same shape as the original, initialized to zeros
        sampled_image = np.zeros_like(original_image, dtype=int)    # Increment counts in the sampled image based on sampled indices
        for index in sampled_indices:
            row, col = np.unravel_index(index, original_image.shape)
            sampled_image[row, col] += 1    
        return sampled_image

    def expose(self):
        
        ## Unsure if necessary, I think the pylablib sdk for andor already manages this
        #while self.cam.acquisition_in_progress():
        #    time.sleep(0.1)

        img_raw = self.cam.snap()
        self.img = img_raw
        
        self.data = np.ndarray((self.img.shape[0],self.img.shape[1]), 
                               buffer= self.img, 
                               dtype=np.uint16)
        #data_float = np.ndarray((self.img.shape[0],self.img.shape[1]), 
        #                       buffer= self.img, 
        #                       dtype=np.float32)
        
        data_float = self.data.astype(np.float32)
        
        data_no_dark = self.data.astype(self.imageDType) - self.dark 

        data_no_dark[data_no_dark<0] = 0


        if self.total_photon_flux > 0:
            data_no_dark = self.sample_image_events(data_no_dark, self.total_photon_flux)
            #data_no_dark = (((data_no_dark) / np.sum(data_no_dark) * self.total_photon_flux))

        if self.activateNoise:
            data_no_dark = (self.random_state_photon_noise.poisson(data_no_dark))


        #super().expose()
        self.imageRaw.write(self.data)
        #Check float here
        self.image.write(data_no_dark.astype(self.imageDType))

        return

    def __del__(self):
        super().__del__()
        time.sleep(1e-1)
        self.close_shutter()
        self.cam.close()
        
        return