import numpy as np
from numba import cuda
import math
import time


@cuda.jit
def optimized_cuda_kernel_50k(gCM, slopes_TR, weights, ref_signal_per_mode_normed, new_corr,
                              signal_workspace, signal_final_workspace, nModes, signal_size):
    """
    Highly optimized CUDA kernel for 50k signal points
    Uses shared memory and cooperative groups for maximum performance
    """
    # Thread and block indices
    mode = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    if mode >= nModes:
        return

    # Shared memory for reduction operations
    shared_sums = cuda.shared.array(256, cuda.float32)
    shared_signals = cuda.shared.array(256, cuda.float32)

    # Each thread processes multiple elements
    elements_per_thread = (signal_size + block_size - 1) // block_size

    # Phase 1: Calculate weighted signal and sum in parallel
    local_sum = 0.0
    for elem_idx in range(elements_per_thread):
        global_idx = tid + elem_idx * block_size
        if global_idx < signal_size:
            # Convert flat index to 2D coordinates for slopes_TR
            row = global_idx // weights.shape[0]
            col = global_idx % weights.shape[0]

            if row < slopes_TR.shape[0] and col < slopes_TR.shape[1]:
                signal_val = slopes_TR[row, col] * weights[col, mode]
                signal_workspace[mode, global_idx] = signal_val
                local_sum += signal_val
            else:
                signal_workspace[mode, global_idx] = 0.0

    # Store local sum in shared memory
    shared_sums[tid] = local_sum
    cuda.syncthreads()

    # Parallel reduction to get total sum
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared_sums[tid] += shared_sums[tid + stride]
        cuda.syncthreads()
        stride //= 2

    total_sum = shared_sums[0]
    cuda.syncthreads()

    # Phase 2: Normalize and calculate final correlation
    local_correlation = 0.0
    if total_sum != 0.0:
        inv_sum = 1.0 / total_sum

        for elem_idx in range(elements_per_thread):
            global_idx = tid + elem_idx * block_size
            if global_idx < signal_size:
                # Normalize signal
                normalized_signal = signal_workspace[mode, global_idx] * inv_sum
                # Calculate final signal
                final_val = normalized_signal - ref_signal_per_mode_normed[global_idx, mode]
                # Accumulate correlation
                if normalized_signal != 0.0:
                    local_correlation += gCM[mode, global_idx] * final_val

    # Store correlation in shared memory for reduction
    shared_signals[tid] = local_correlation
    cuda.syncthreads()

    # Final reduction for correlation
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared_signals[tid] += shared_signals[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Thread 0 writes the final result
    if tid == 0:
        new_corr[mode] = shared_signals[0]


@cuda.jit
def simple_cuda_kernel_50k(gCM, slopes_TR, weights, ref_signal_per_mode_normed, new_corr):
    """
    Simpler kernel - one block per mode, threads cooperate within block
    """
    mode = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    if mode >= weights.shape[1]:
        return

    # Shared memory for reductions
    shared_data = cuda.shared.array(256, np.float32)

    # Calculate how many elements each thread processes
    nRows, nCols = slopes_TR.shape
    total_elements = nRows * nCols
    elements_per_thread = (total_elements + block_size - 1) // block_size

    # Phase 1: Calculate sum of weighted signal


    local_sum = 0.0
    for i in range(elements_per_thread):
        elem_idx = tid + i * block_size
        if elem_idx < total_elements:
            row = elem_idx // nCols
            col = elem_idx % nCols
            if col < weights.shape[0]:  # Bounds check
                local_sum += slopes_TR[row, col] * weights[col, mode]

    shared_data[tid] = local_sum
    cuda.syncthreads()

    # Reduction for sum
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared_data[tid] += shared_data[tid + stride]
        cuda.syncthreads()
        stride //= 2

    total_sum = shared_data[0]
    cuda.syncthreads()

    # Phase 2: Calculate correlation
    local_corr = 0.0
    if total_sum != 0.0:
        inv_sum = 1.0 / total_sum

        for i in range(elements_per_thread):
            elem_idx = tid + i * block_size
            if elem_idx < total_elements:
                row = elem_idx // nCols
                col = elem_idx % nCols
                if col < weights.shape[0] and elem_idx < ref_signal_per_mode_normed.shape[0]:
                    signal_val = slopes_TR[row, col] * weights[col, mode] * inv_sum
                    final_val = signal_val - ref_signal_per_mode_normed[elem_idx, mode]
                    if signal_val != 0.0:
                        local_corr += gCM[mode, elem_idx] * final_val

    shared_data[tid] = local_corr
    cuda.syncthreads()

    # Final reduction
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared_data[tid] += shared_data[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        new_corr[mode] = shared_data[0]


class HighFrequencyGPUProcessor:
    """
    GPU processor optimized for 50k signal points at high frequency
    """

    def __init__(self, gCM, weights, ref_signal_per_mode_normed):
        # Validate dimensions
        self.nModes = weights.shape[1]
        self.signal_size = 50000  # Fixed at 50k

        print(f"Initializing GPU processor:")
        print(f"  Modes: {self.nModes}")
        print(f"  Signal size: {self.signal_size}")
        print(f"  gCM shape: {gCM.shape}")
        print(f"  weights shape: {weights.shape}")
        print(f"  ref_signal shape: {ref_signal_per_mode_normed.shape}")

        # Copy static matrices to GPU (done once)
        self.gCM_gpu = cuda.to_device(np.ascontiguousarray(gCM, dtype=np.float32))
        self.weights_gpu = cuda.to_device(np.ascontiguousarray(weights, dtype=np.float32))
        self.ref_signal_gpu = cuda.to_device(np.ascontiguousarray(ref_signal_per_mode_normed, dtype=np.float32))

        # Pre-allocate output and workspace arrays
        self.new_corr_gpu = cuda.device_array(self.nModes, dtype=np.float32)

        # Workspace for intermediate calculations (if using optimized kernel)
        self.signal_workspace_gpu = cuda.device_array((self.nModes, self.signal_size), dtype=np.float32)
        self.signal_final_workspace_gpu = cuda.device_array((self.nModes, self.signal_size), dtype=np.float32)

        # Optimize block and grid configuration
        self.threads_per_block = 256  # Good for shared memory usage
        self.blocks_per_grid = self.nModes  # One block per mode

        # Pre-allocate pinned memory for faster transfers
        self.slopes_pinned = cuda.pinned_array((200, 250), dtype=np.float32)  # Adjust size as needed
        self.result_pinned = cuda.pinned_array(self.nModes, dtype=np.float32)

        print(f"GPU configuration: {self.blocks_per_grid} blocks x {self.threads_per_block} threads")

        # Warm up
        self._warmup()

    def _warmup(self):
        """Warm up the GPU kernel"""
        dummy_slopes = np.random.random((200, 250)).astype(np.float32)
        for _ in range(5):
            self.process(dummy_slopes)
        print("GPU warmup complete")

    def process(self, slopes_TR):
        """
        High-speed processing - only slopes_TR is transferred
        """
        # Copy slopes_TR to pinned memory first (faster transfer)
        np.copyto(self.slopes_pinned[:slopes_TR.shape[0], :slopes_TR.shape[1]], slopes_TR)

        # Transfer to GPU
        slopes_TR_gpu = cuda.to_device(self.slopes_pinned[:slopes_TR.shape[0], :slopes_TR.shape[1]])

        # Launch optimized kernel
        simple_cuda_kernel_50k[self.blocks_per_grid, self.threads_per_block](
            self.gCM_gpu,
            slopes_TR_gpu,
            self.weights_gpu,
            self.ref_signal_gpu,
            self.new_corr_gpu
        )

        # Copy result back using pinned memory
        self.new_corr_gpu.copy_to_host(self.result_pinned)
        return self.result_pinned.copy()

    def process_stream(self, slopes_TR, stream=None):
        """
        Streaming version for even better performance
        """
        if stream is None:
            stream = cuda.stream()

        # Async transfer
        slopes_TR_gpu = cuda.to_device(slopes_TR, stream=stream)

        # Async kernel launch
        simple_cuda_kernel_50k[self.blocks_per_grid, self.threads_per_block, stream](
            self.gCM_gpu, slopes_TR_gpu, self.weights_gpu, self.ref_signal_gpu, self.new_corr_gpu
        )

        # Async copy back
        result = self.new_corr_gpu.copy_to_host(stream=stream)
        stream.synchronize()

        return result

    def benchmark(self, slopes_TR, num_iterations=1000):
        """Comprehensive benchmarking"""
        # Warm up
        for _ in range(20):
            self.process(slopes_TR)

        # Benchmark different methods
        methods = {
            'standard': self.process,
            'streaming': self.process_stream
        }

        results = {}
        for method_name, method_func in methods.items():
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                result = method_func(slopes_TR)
                end = time.perf_counter()
                times.append((end - start) * 1_000_000)  # Convert to microseconds

            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)

            results[method_name] = {
                'avg_time_us': avg_time,
                'min_time_us': min_time,
                'max_time_us': max_time,
                'std_time_us': std_time,
                'max_frequency_hz': 1_000_000 / avg_time,
                'can_do_1khz': avg_time < 1000,
                'can_do_500hz': avg_time < 2000,
                'success_rate_1khz': np.sum(np.array(times) < 1000) / len(times) * 100
            }

        return results


# Example usage
if __name__ == "__main__":
    # Create realistic test data for 50k signal points
    nModes = 100
    nSensors = 250
    signal_size = 50000

    gCM = np.random.random((nModes, signal_size)).astype(np.float32)
    weights = np.random.random((nSensors, nModes)).astype(np.float32)
    ref_signal = np.random.random((signal_size, nModes)).astype(np.float32)
    slopes_TR = np.random.random((200, nSensors)).astype(np.float32)  # 200*250 = 50k

    print(f"Test data shapes:")
    print(f"  gCM: {gCM.shape}")
    print(f"  weights: {weights.shape}")
    print(f"  ref_signal: {ref_signal.shape}")
    print(f"  slopes_TR: {slopes_TR.shape}")
    print(f"  Total signal elements: {slopes_TR.size}")

    # Create and benchmark GPU processor
    gpu_processor = HighFrequencyGPUProcessor(gCM, weights, ref_signal)

    # Run benchmark
    results = gpu_processor.benchmark(slopes_TR, num_iterations=100)

    for method, stats in results.items():
        print(f"\n{method.upper()} METHOD:")
        print(f"  Average time: {stats['avg_time_us']:.1f} ± {stats['std_time_us']:.1f} µs")
        print(f"  Range: {stats['min_time_us']:.1f} - {stats['max_time_us']:.1f} µs")
        print(f"  Max frequency: {stats['max_frequency_hz']:.0f} Hz")
        print(f"  1kHz success rate: {stats['success_rate_1khz']:.1f}%")
        print(f"  Can handle 1kHz: {stats['can_do_1khz']}")
        print(f"  Can handle 500Hz: {stats['can_do_500hz']}")

    # Test actual processing
    result = gpu_processor.process(slopes_TR)
    print(f"\nProcessing result:")
    print(f"  Output shape: {result.shape}")
    print(f"  Sample values: {result[:5]}")