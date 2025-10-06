import numpy as np
from numba import jit, prange
import time


import ctypes
import numpy as np







# Ultra-optimized CPU version for high-frequency processing
@jit(nopython=True, parallel=True, nogil=True)
def ultra_fast_function(gCM=np.array([[]], dtype=np.float32),
                                    slopes_TR=np.array([[]], dtype=np.float32),
                                   weights=np.array([[]], dtype=np.float32),
                                   ref_signal_per_mode_normed=np.array([[]], dtype=np.float32),
                                   new_corr=np.array([], dtype=np.float32)):
    #nModes= weights.shape[1]
    signal_sum = np.sum(slopes_TR @ weights, axis=0)
    for i in prange(weights.shape[0]):
        for j in range(slopes_TR.shape[0]):
            if slopes_TR[j,i] !=0:
                new_corr +=  gCM[:, i, j]* (((slopes_TR[j,i] * weights[i, :])/signal_sum) - ref_signal_per_mode_normed[j,i,:])


@jit(nopython=True, parallel=True, nogil=True)
def ultra_fast_function_mult(gCM=np.array([[]], dtype=np.float32),
                                    slopes_TR=np.array([], dtype=np.float32),
                                   weights=np.array([[]], dtype=np.float32),
                                   ref_signal_per_mode_normed=np.array([[]], dtype=np.float32),
                                   new_corr=np.array([], dtype=np.float32)):
    signal_sum = slopes_TR @ weights
    #signal_sum = np.sum(slopes_TR)

    for mode in prange(gCM.shape[0]):
        new_corr[mode] = np.dot(gCM[mode, :],  (slopes_TR * weights[:, mode] / signal_sum[mode])  - ref_signal_per_mode_normed[:, mode])

@jit(nopython=True, parallel=True, error_model='numpy', nogil=True)
def ultra_fast_function_const(slopes_TR=np.array([[]], dtype=np.float32),
                                   new_corr=np.array([], dtype=np.float32)):
    #nModes= weights.shape[1]
    nFrames = gweights.shape[0]
    signal_size = slopes_TR.shape[0]
    signal_sum = np.sum(slopes_TR @ weights, axis=0)
    for i in prange(nFrames):
        for j in range(signal_size):
            if slopes_TR[j,i] !=0:
                new_corr +=  ggCM[:, i, j]* (((slopes_TR[j,i] * gweights[i, :])/signal_sum) - gref_signal[j,i,:])

# Ultra-optimized CPU version for high-frequency processing
@jit(nopython=True, parallel=True, error_model='numpy',fastmath=True)
def ultra_fast_function_no_weights(gCM=np.array([[]], dtype=np.float32),
                                    slopes_TR=np.array([[]], dtype=np.float32),
                                   ref_signal_per_mode_normed=np.array([[]], dtype=np.float32),
                                   new_corr=np.array([], dtype=np.float32)):
    #nModes= weights.shape[1]
    nFrames = slopes_TR.shape[1]
    signal_size = slopes_TR.shape[0]
    signal_sum = np.sum(slopes_TR)
    for i in prange(nFrames):
        for j in range(signal_size):
            if slopes_TR[j,i] !=0:
                new_corr +=  gCM[:, i, j]* ((slopes_TR[j,i]/signal_sum) - ref_signal_per_mode_normed[j,i])

# Ultra-optimized CPU version for high-frequency processing
@jit(nopython=True, parallel=True)
def ultra_fast_function_ori(gCM=np.array([[]], dtype=np.float32),
                                    slopes_TR=np.array([[]], dtype=np.float32),
                                   weights=np.array([[]], dtype=np.float32),
                                   ref_signal_per_mode_normed=np.array([[]], dtype=np.float32)):
    nModes= weights.shape[1]
    nFrames = weights.shape[0]
    new_corr = np.zeros(gCM.shape[0], dtype=np.float32)
    for mode in prange(nModes):
        signal_for_mode = np.zeros_like(slopes_TR)
        for i in range(nFrames):
            signal_for_mode[:,i] = slopes_TR[:,i] * weights[i, mode]
        #signal_for_mode_t = signal_for_mode.flatten() / np.sum(signal_for_mode)
        #if np.sum(signal_for_mode) != 0:
        #    signal_for_mode /= np.sum(signal_for_mode)
        signal_final = (signal_for_mode.flatten() / np.sum(signal_for_mode)) - ref_signal_per_mode_normed[:,mode]
        new_corr[mode] = np.dot(gCM[mode, :],signal_final[:])
        #for k in range(gCM.shape[1]):
        #   if signal_for_mode[k] != 0:
        #       new_corr[mode] += gCM[mode, k] * (signal_for_mode[k] - ref_signal_per_mode_normed[k,mode])

    return new_corr


# Version with pre-allocated work arrays (even faster for repeated calls)
@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def ultra_fast_function_preallocated(gCM, slopes_TR, weights, ref_signal_per_mode_normed,
                                     work_array1, work_array2, new_corr):
    """
    Version with pre-allocated work arrays - zero allocation during execution
    """
    nModes = weights.shape[1]
    nFrames = weights.shape[0]
    for mode in prange(nModes):

        for i in range(nFrames):
            slopes_TR[:, i] *= weights[i, mode]
        work_array1 = slopes_TR.flatten() / np.sum(slopes_TR)

        work_array2 =  work_array1 - ref_signal_per_mode_normed[:,mode]

        new_corr[mode] = 0
        for k in range(gCM.shape[1]):
            if work_array2[k] != 0:
                new_corr[mode] += gCM[mode, k] * work_array2[k]



class HighFrequencyProcessor:
    """
    Wrapper class optimized for high-frequency processing
    """

    def __init__(self, nModes, nFrames, nSignalPoints, gCM, weights, ref_signal_per_mode_normed):
        # Store references (no copying)
        self.gCM = np.ascontiguousarray(gCM, dtype=np.float32)
        self.weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.ref_signal = np.ascontiguousarray(ref_signal_per_mode_normed, dtype=np.float32)
        self.nModes = nModes 
        self.nFrames = nFrames 
        self.nSignalPoints = nSignalPoints
        # Pre-allocate work arrays
        self.signal_size = gCM.shape[1]
        self.work_array1 = np.zeros(self.signal_size, dtype=np.float32)
        self.work_array2 = np.zeros(self.signal_size, dtype=np.float32)
        self.result = np.zeros(self.nModes, dtype=np.float32)

        # Warm up JIT compilation
        test_slopes = np.ones((self.nSignalPoints, self.nFrames), dtype=np.float32)
        new_corr = np.zeros(self.nModes, dtype=np.float32)
        self._process_internal(test_slopes, new_corr)

    def process(self, slopes_TR):
        """
        Ultra-fast processing with pre-allocated arrays
        """
        slopes_contiguous = np.ascontiguousarray(slopes_TR, dtype=np.float32)
        new_corr = np.ascontiguousarray(np.zeros(self.nModes), dtype=np.float32)
        return self._process_internal(slopes_contiguous, new_corr)

    def _process_internal(self, slopes_TR, new_corr):
        """Internal processing method"""

        lib.trpwfs_FF_W_calc(self.nModes, 
                        self.nFrames, 
                        self.nSignalPoints,
                        self.gCM.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        slopes_TR.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self.ref_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        new_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


        #ultra_fast_function_mult(self.gCM, slopes_TR, self.weights, self.ref_signal, new_corr)
        #ultra_fast_function_no_weights(self.gCM, slopes_TR, self.ref_signal[:,:,0], new_corr)
        #ultra_fast_function_const(slopes_TR, new_corr)
        return new_corr.copy()

    def benchmark(self, slopes_TR, num_iterations=1000):
        """Benchmark the processing speed"""
        # Warm up
        for _ in range(10):
            self.process(slopes_TR)

        # Time the processing
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            result = self.process(slopes_TR)
        end_time = time.perf_counter()

        avg_time_us = (end_time - start_time) * 1_000_000 / num_iterations
        max_frequency = 1_000_000 / avg_time_us

        return {
            'avg_time_us': avg_time_us,
            'max_frequency_hz': max_frequency,
            'can_do_1khz': avg_time_us < 1000,
            'can_do_500hz': avg_time_us < 2000
        }


# Example usage and benchmarking
if __name__ == "__main__":

    import os 
    #os.environ["OMP_NUM_THREADS"] = '4'

    # Load the shared library
    lib = ctypes.CDLL("REVOLT/res/trpwfs_lib.dll")

    # Define the C function's argument types and return type
    lib.trpwfs_FF_W_calc.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.trpwfs_FF_W_calc.restype = None  # No return value
    # Create test data
    nModes = 100
    nFrames = 48
    nSignalPoints = 1000

    gCM = np.random.random((nModes, nFrames,nSignalPoints)).astype(np.float32)
    weights = np.random.random((nFrames, nModes)).astype(np.float32)
    ref_signal = np.random.random((nSignalPoints,nFrames, nModes)).astype(np.float32)
    slopes_TR = np.random.random((nSignalPoints,nFrames)).astype(np.float32)

    # Create processor
    processor = HighFrequencyProcessor(nModes, nFrames, nSignalPoints, gCM, weights, ref_signal)

    # Benchmark
    results = processor.benchmark(slopes_TR,2000)

    print(f"Average processing time: {results['avg_time_us']:.1f} µs")
    print(f"Maximum frequency: {results['max_frequency_hz']:.0f} Hz")
    print(f"Can handle 1kHz: {results['can_do_1khz']}")
    print(f"Can handle 500Hz: {results['can_do_500hz']}")

    # Process data
    result = processor.process(slopes_TR)
    print(f"Output shape: {result.shape}")
    print(f"Sample values: {result[:5]}")