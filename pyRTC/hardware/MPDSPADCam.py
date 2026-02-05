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

        self.screamers = [(20,12), (4, 40), (7, 2), (7, 33), (7, 48), (9, 7), (10, 12), (11, 14), (11, 36), (11, 50), (12, 5), (12, 51), (13, 9), (13, 45), (14, 53), (19, 33), (21, 30), (21, 37), (28, 7), (28, 34), (28, 44), (28, 50), (29, 17), (29, 42), (30, 60), (30, 61), (31, 43), (31, 44)]

        self.cam = Hermes(Hermes.CameraMode.NORMAL) # Start in normal
        self.cam.SetCameraPar(Exposure = self.exposure,  # if in normal mode, this is ignored and forced to 10.40 us, else is in 10ns increments
                                NFrames = self.nFrames, 
                                NIntegFrames = self.nIntegFrames , 
                                NCounters = self.nCounters , 
                                Force8bit = Hermes.State.DISABLED, 
                                Half_array = Hermes.State.DISABLED, 
                                Signed_data = Hermes.State.DISABLED)
        self.cam.ApplySettings()
        self.inSyncMode = False

        #self._data_cube = np.zeros((100, self.frames.shape[0], self.frames.shape[1], self.frames.shape[2]))
        #self._number_of_acquisitions = 100
        self._done_recording = False
        self._start_recording = False 
        self.offset_read = 0
        self.rolling_data = None
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
        self.inSyncMode = True 

    def disable_sync_mode(self):
        self.cam.SetSyncInState(Hermes.State.DISABLED, 0)
        self.inSyncMode = False

    def advancedMode(self, adv):
        if self.running:
            print("Stop device before switching modes")
        else:
            if adv:
                self.cam.SetAdvancedMode(Hermes.State.ENABLED)
            else:
                self.cam.SetAdvancedMode(Hermes.State.DISABLED)


    def start(self):
        self.cam.ContAcqToMemoryStart()
        self.offset_read = 0
        self.rolling_data = None
        super().start()

    def stop(self):
        super().stop()
        time.sleep(0.01)
        self.cam.ContAcqToMemoryStop()
        self.offset_read = 0
        self.rolling_data = None


    def expose_snap(self):
        
        if self.cam.IsTriggered() or (not self.inSyncMode) :
            self.cam.SnapPrepare()
            self.cam.SnapAcquire()

            # TODO do we want to use other counters?
            #data = np.empty((0,), dtype=np.uint8)
            #while data.size < (self.nFrames+self.offset_read)*32*64:
            #    data = np.concatenate((data, self.cam.ContAcqToMemoryGetBuffer()))
            self.frames = self.cam.SnapGetImageBuffer()[0]  # frames of counter 1 

            for idx in self.screamers:
                self.frames[:,idx[0], idx[1]] = 0
            
            self.data = np.ndarray((self.frames.shape[0],self.frames.shape[1], self.frames.shape[2]), 
                                buffer= np.ascontiguousarray(self.frames), 
                                dtype=self.frames.dtype)
            super().expose()

            if self._start_recording:
                self._data_cube[self._acq_number,:,:,:] = self.data
                self._acq_number += 1
                if self._acq_number >= self._number_of_acquisitions:
                    self._done_recording = True
                    self._start_recording = False
        return

    def expose(self):
        
        if self.cam.IsTriggered() or (not self.inSyncMode) :
            #self.cam.SnapPrepare()
            #self.cam.SnapAcquire()

            # TODO do we want to use other counters?
            #self.rolling_data = np.empty((0,), dtype=np.uint8)
            if self.rolling_data is not None:
                current_size = self.rolling_data.size
            else:
                current_size = 0
            while current_size < (self.nFrames)*32*64:
                data = self.cam.ContAcqToMemoryGetBuffer()
                all_frames = (self.cam.BufferToFrames(data, self.cam.num_pixels, self.cam.num_counters)[0])
                if self.rolling_data is not None:
                    self.rolling_data = np.concatenate((self.rolling_data, all_frames))
                else:
                    self.rolling_data = all_frames
                
                current_size = self.rolling_data.size

            self.frames = self.rolling_data[:self.nFrames]
            self.rolling_data = self.rolling_data[self.nFrames:]
            #self.offset_read = self.nFrames - ((all_frames.shape[0]-self.offset_read) % self.nFrames)
            
            #self.data = (self.cam.BufferToFrames(buf, self.cam.num_pixels, self.cam.num_counters)[0])[:48,:,:]
            
            for idx in self.screamers:
                self.frames[:,idx[0], idx[1]] = 0
            
            # self.data = np.ndarray((frames.shape[0],frames.shape[1], frames.shape[2]), 
            #                     buffer= np.ascontiguousarray(frames), 
            #                     dtype=frames.dtype)
            self.data = self.frames
            super().expose()

            if self._start_recording:
                self._data_cube[self._acq_number,:,:,:] = self.data
                self._acq_number += 1
                if self._acq_number >= self._number_of_acquisitions:
                    self._done_recording = True
                    self._start_recording = False
        return
    
    
    def record_data_direct(self, number_of_acquisitions, filename_prefix, to_fits=True, to_hdf5=True):

        self._acq_number = 0
        self._number_of_acquisitions = number_of_acquisitions
        self._data_cube = np.zeros((number_of_acquisitions, self.frames.shape[0], self.frames.shape[1], self.frames.shape[2]))
        self._done_recording = False
        self._start_recording = True
        while (self._done_recording == False):
            time.sleep(0.01)

        if to_fits:
            self.save_images_to_fits(self._data_cube, f"{filename_prefix}.fits")
        if to_hdf5:
            self.save_images_to_hdf5(self._data_cube, f"{filename_prefix}.hdf5")


    def record_data_bypass(self, number_of_acquisitions, filename_prefix, to_fits=True, to_hdf5=True):

        if self.running:
            print("Stop running before calling this function")
            return 

        data_cube = np.zeros((number_of_acquisitions, self.data.shape[0], self.data.shape[1], self.data.shape[2]))
        acq = 0
        while (acq < number_of_acquisitions) :
            if self.cam.IsTriggered() or (not self.inSyncMode) :
                self.cam.SnapPrepare()
                self.cam.SnapAcquire()
                new_acq = self.cam.SnapGetImageBuffer()[0]

                if new_acq.shape[0] != self.nFrames:  # Sometimes the snap returns nothing
                    continue 

                #print(f"{acq}/{number_of_acquisitions}")
                if acq == 0:
                    data_cube[acq,:,:,:] = new_acq
                    acq += 1
                else:
                    #if not np.array_equal(data_cube[acq-1,:,:,:], new_acq):  #Avoid recording the same acquisition back to back
                    data_cube[acq,:,:,:] = new_acq
                    acq += 1

        if to_fits:
            self.save_images_to_fits(data_cube, f"{filename_prefix}.fits")
        if to_hdf5:
            self.save_images_to_hdf5(data_cube, f"{filename_prefix}.hdf5")


    def record_data(self, number_of_acquisitions, filename_prefix, to_fits=True, to_hdf5=True):

        data_cube = np.zeros((number_of_acquisitions, self.frames.shape[0], self.frames.shape[1], self.frames.shape[2]))
        acq = 0
        while (acq < number_of_acquisitions) :
            new_acq =  self.read()
            print(f"{acq}/{number_of_acquisitions}")
            if acq == 0:
                data_cube[acq,:,:,:] = new_acq
                acq += 1
            else:
                if not np.array_equal(data_cube[acq-1,:,:,:], new_acq):  #Avoid recording the same acquisition back to back
                    data_cube[acq,:,:,:] = new_acq
                    acq += 1

        if to_fits:
            self.save_images_to_fits(self._data_cube, f"{filename_prefix}.fits")
        if to_hdf5:
            self.save_images_to_hdf5(self._data_cube, f"{filename_prefix}.hdf5")


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
        primary_hdu = fits.PrimaryHDU(data_cube.astype('<f8'))

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
    set_affinity((conf["trwfs"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFC = conf["trwfs"]
    trwfs = MPDSPADCam(conf=confWFC)
    trwfs.start()

    l = Listener(trwfs, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)