from pyRTC.TimeResolvedWavefrontSensor import TimeResolvedWavefrontSensor
from pyRTC.pyRTCComponent import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
from astropy.io import fits
import h5py
import argparse
import sys
import os 
from pyRTC.hardware.Hermes.Hermes import *

# Class for the high-speed MPD SPAD camera
# Since we process cubes of images, we dont inherit WavefrontSensor
class MPDSPADCam(TimeResolvedWavefrontSensor):

    def __init__(self, conf) -> None:
        # Default settings for Hermes cam
        self.setExposure(setFromConfig(conf, "exposure", 100))
        self.setNFrames(setFromConfig(conf, "nFrames", 100))
        self.setNIntegFrames(setFromConfig(conf, "nIntegFrames", 300))
        self.setNCounters(setFromConfig(conf, "nCounters", 1))

        self.cam = Hermes(Hermes.CameraMode.NORMAL) # Start in normal
        self.cam.SetCameraPar(Exposure = self.exposure,  # if in normal mode, this is ignored and forced to 10.40 us, else is in 10ns increments
                                NFrames = self.nFrames, 
                                NIntegFrames = self.nIntegFrames , 
                                NCounters = self.nCounters , 
                                Force8bit = Hermes.State.DISABLED, 
                                Half_array = Hermes.State.DISABLED, 
                                Signed_data = Hermes.State.DISABLED)
        self.cam.ApplySettings()
        
        super().__init__(conf)
        return 

    def __del__(self):
        super().__del__()
        time.sleep(1e-1)
        del self.cam  # Hermes SDK managed the deallocation of memory
        return

    def applySettingsToCamera(self):
        if self.running:
            print("Stop device first")
        else:
            self.cam.SetCameraPar(Exposure = self.exposure,  # if in normal mode, this is ignored and forced to 10.40 us, else is in 10ns increments
                                NFrames = self.nFrames, 
                                NIntegFrames = self.nIntegFrames , 
                                NCounters = self.nCounters , 
                                Force8bit = Hermes.State.DISABLED, 
                                Half_array = Hermes.State.DISABLED, 
                                Signed_data = Hermes.State.DISABLED)
            self.cam.ApplySettings()

    def setNCounters(self, value):
        if 1 <= value <= 3:
            self.nCounters = value
        else:
            raise ValueError("Invalid value for nCounters")  

    def setNIntegFrames(self, value):
        if 1 <= value <= 65534:
            self.nIntegFrames = value
        else:
            raise ValueError("Invalid value for nIntegFrames")

    def setNFrames(self, value):
        if 1 <= value <= 65534:
            self.nFrames = value
        else:
            raise ValueError("Invalid value for nFrames")

    def setExposure(self, value):
        if 1 <= value <= 65534:
            self.exposure = value
        else:
            raise ValueError("Invalid value for exposure")

    def enable_sync_mode(self):
        if (int(self.nFrames) > 100) or (int(self.nFrames) < 0):
            print("Change the nFrames parameter first, limit is 100 frames for sync mode")
        self.cam.SetSyncInState(Hermes.State.ENABLED, int(self.nFrames))

    def disable_sync_mode(self):
        self.cam.SetSyncInState(Hermes.State.DISABLED, 0)

    def advancedMode(self, adv):
        if self.running:
            print("Stop device before switching modes")
        else:
            if adv:
                self.cam.SetAdvancedMode(Hermes.State.ENABLED)
            else:
                self.cam.SetAdvancedMode(Hermes.State.DISABLED)

    def expose(self):
        
        self.cam.SnapPrepare() # TODO I think this becomes blocking when in sync mode. Otherwise, might have to use IsTriggered()
        self.cam.SnapAcquire()

        # TODO do we want to use other counters?
        self.frames = self.cam.SnapGetImageBuffer()[0]  # frames of counter 1 
        self.data = np.ndarray((self.frames.shape[0],self.frames.shape[1], self.frames.shape[2]), 
                            buffer= np.ascontiguousarray(self.frames), 
                            dtype=np.uint16)
        super().expose()
        return
    
    
    def record_data(self, number_of_acquisitions, filename_prefix, to_fits=True, to_hdf5=True):

        data_cube = np.zeros((number_of_acquistions, self.frames.shape[0], self.frames.shape[1], self.frames.shape[2]))

        for acq in range(number_of_acquisitions):
            data_cube[acq,:,:,:] = self.read()

        if to_fits:
            self.save_images_to_fits(data_cube, f"{filename}.hdf5")
        if to_hdf5:
            self.save_images_to_hdf5(data_cube, f"{filename}.hdf5")



    def save_images_to_fits(self, data_cube, filename, headers=None, overwrite=True):
        """
        Save AxNxWxH image array to a FITS file.

        Parameters:
        -----------
        data_cube : numpy.ndarray
            Array with shape (A, N, W, H) where:
            A = number of acquisitions
            N = number of image frames per acquisition
            W, H = image dimensions
        filename : str
            Output FITS filename
        headers : dict, optional
            Header dictionary to add to FITS file
        overwrite : bool
            Whether to overwrite existing file
        """

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU(data_cube)

        # Add headers if provided
        if headers:
            for key, value in headers.items():
                primary_hdu.header[key] = value

        # Create HDU list and write
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()



    def save_images_to_hdf5(self, data_cube, filename, dataset_name='images', attrs=None, compression='gzip'):
        """
        Save AxNxWxH image array to an HDF5 file.

        Parameters:
        -----------
        data_cube : numpy.ndarray
            Array with shape (A, N, W, H) where:
            A = number of acquisitions
            N = number of image frames per acquisition
            W, H = image dimensions
        filename : str
            Output HDF5 filename
        dataset_name : str
            Name of the dataset in HDF5 file
        attrs : dict, optional
            Attributes dictionary to add to dataset
        compression : str
            Compression algorithm ('gzip', 'lzf', 'szip', or None)
        """
        with h5py.File(filename, 'w') as f:
            # Create dataset with compression
            dset = f.create_dataset(dataset_name, data=data_cube, compression=compression)

            # Add dimension labels as attributes
            dset.attrs['dimensions'] = ['acquisitions', 'frames', 'width', 'height']
            dset.attrs['shape_description'] = f'({data_cube.shape[0]} acq, {data_cube.shape[1]} frames, {data_cube.shape[2]}x{data_cube.shape[3]} pixels)'

            # Add custom attributes if provided
            if attrs:
                for key, value in attrs.items():
                    dset.attrs[key] = value
