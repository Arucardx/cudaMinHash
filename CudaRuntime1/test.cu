/*
#pragma once 
#include "cudaargs.h"
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include "cuda.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <cccl/cub/cub.cuh>




#define NUM_HASH_FUNCTIONS 64
#define MINHASH_THREADS 64

const uint32_t k = 4;

void test_minhash();
int test_murmur3_32_3();
int test_pipeline();

// this implementation of the murmur3 hash requires little endian architecture (standard for nvidia)
__global__ void murmur3_32(uint8_t* __restrict__ buffer, uint32_t* __restrict__ offsets, uint32_t* __restrict__ hashes, uint32_t length, uint32_t seed) {

	uint32_t text_idx = blockIdx.x;

	uint32_t data_end = offsets[text_idx];
	uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]) + threadIdx.x;

	for (; data_start < data_end - length + 1; data_start += blockDim.x) {
		uint32_t h = seed;
		uint32_t k;

		for (size_t i = 0; i < length / 4; i++) {

			//eventuell unaligned, deswegen byteweise lesen
#pragma unroll
			for (size_t j = 0; j < 4; j++) {
				k = k << 8 | buffer[data_start + i * 4 + 3 - j];
			}
;
			k *= 0xcc9e2d51;
			k = (k << 15) | (k >> 17);
			k *= 0x1b873593;

			h ^= k;
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}

		k = 0;

#pragma unroll
		for (size_t i = length & 3; i; i--) {
			k <<= 8;
			k |= buffer[data_start + (length & ~3) + i - 1];
		}

		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;

		h ^= k;
		h ^= length;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;

		hashes[data_start + text_idx * (1 - length)] = h;
	}

}

__inline__ __device__ void murmur3_32_f(uint8_t* __restrict__ buffer, uint32_t length, uint32_t seed, uint32_t* hash) {

	uint32_t h = seed;
	uint32_t k;

	for (size_t i = 0; i < length / 4; i++) {

		//eventuell unaligned, deswegen byteweise lesen
#pragma unroll
		for (size_t j = 0; j < 4; j++) {
			k = k << 8 | buffer[i * 4 + 3 - j];
		}

		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;

		h ^= k;
		h = (h << 13) | (h >> 19);
		h = h * 5 + 0xe6546b64;
	}

	k = 0;

#pragma unroll
	for (size_t i = length & 3; i; i--) {
		k <<= 8;
		k |= buffer[(length & ~3) + i - 1];
	}

	k *= 0xcc9e2d51;
	k = (k << 15) | (k >> 17);
	k *= 0x1b873593;

	h ^= k;
	h ^= length;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	*hash = h;
}

//num_docs * num_buckets
#define NUM_BUCKET_LANES 64
__device__ void murmur3_32_lsh(uint8_t* __restrict__ buffer, uint32_t seed) {

	uint32_t text_idx = blockIdx.x;

	uint32_t lane_size = sizeof(uint32_t) * (NUM_HASH_FUNCTIONS / NUM_BUCKET_LANES);
	uint32_t data_start = text_idx * NUM_BUCKET_LANES + threadIdx.x * lane_size;

	murmur3_32_f(&buffer[data_start], lane_size, seed, (uint32_t*)&buffer[data_start]);
}


//reuse for lsh_hashing?
//output-array = input array, no race condition?
__device__ void murmur3_32_fn(uint8_t* __restrict__ buffer, uint32_t* __restrict__ offsets, uint32_t* __restrict__ hashes, uint32_t length, uint32_t seed) {
	uint32_t text_idx = blockIdx.x;

	uint32_t data_end = offsets[text_idx];
	uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]) + threadIdx.x;

	for (; data_start < data_end - length + 1; data_start += blockDim.x) {
		uint32_t h = seed;
		uint32_t k;

		for (size_t i = 0; i < length / 4; i++) {

			//eventuell unaligned, deswegen byteweise lesen
#pragma unroll
			for (size_t j = 0; j < 4; j++) {
				k = k << 8 | buffer[data_start + i * 4 + 3 - j];
			}

			k *= 0xcc9e2d51;
			k = (k << 15) | (k >> 17);
			k *= 0x1b873593;

			h ^= k;
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}

		k = 0;

#pragma unroll
		for (size_t i = length & 3; i; i--) {
			k <<= 8;
			k |= buffer[data_start + (length & ~3) + i - 1];
		}

		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;

		h ^= k;
		h ^= length;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;

		hashes[data_start + text_idx * (1 - length)] = h;
	}
}

__device__ void minhash_fn(uint32_t* __restrict__ hashes, uint32_t* __restrict__ offsets, uint32_t* __restrict__ signature, uint32_t* __restrict__ a, uint32_t* __restrict__ b) {

	uint32_t text_idx = blockIdx.x;

	//uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]);
	uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]) + text_idx * (1 - k);
	//uint32_t data_end = offsets[text_idx];
	uint32_t data_end = offsets[text_idx] + (text_idx + 1) * (1 - k);

	uint32_t a_h = a[threadIdx.x];
	uint32_t b_h = b[threadIdx.x];
	uint32_t c_h = a[threadIdx.x + 64];
	uint32_t d_h = b[threadIdx.x + 64];
	uint32_t tmp;

	uint32_t hash;
	uint32_t minimum1 = UINT32_MAX;
	uint32_t minimum2 = UINT32_MAX;

	//int read_off = threadIdx.x & 0x7;


	for (uint32_t i = data_start; i < data_end; i++) {

		tmp = hashes[i];

		hash = a_h * tmp + b_h;
		minimum1 = min(hash, minimum1);

		hash = c_h * tmp + d_h;
		minimum2 = min(hash, minimum2);
		//hash ^= hash >> 16;
		//hash *= 0x7feb352d;
		//hash ^= hash >> 15;

	}
	signature[text_idx * NUM_HASH_FUNCTIONS + threadIdx.x] = minimum1;
	signature[text_idx * NUM_HASH_FUNCTIONS + threadIdx.x + 64] = minimum2;

}

__global__ void pipeline(uint8_t* __restrict__ buffer, uint32_t* __restrict__ offsets, uint32_t* __restrict__ hashes, uint32_t* __restrict__ a, uint32_t* __restrict__ b, uint32_t length, uint32_t seed) {

	murmur3_32_fn(buffer, offsets, hashes, length, seed);

	__syncthreads();

	minhash_fn(hashes, offsets, (uint32_t*)buffer, a, b);

}

//experimental
__global__ void minhash2(uint32_t* __restrict__ shingle_hashes, uint32_t* __restrict__ offsets, uint32_t* __restrict__ signatures, uint32_t* __restrict__ a, uint32_t* __restrict__ b, uint32_t length) {

	//shared memory == 48kb, so a maximum about 48000 / 8 / 3 == 2000 hash functions should be enough
	__shared__ uint32_t shm[NUM_HASH_FUNCTIONS];

	__syncwarp();

	uint32_t text_idx = blockIdx.x;
	//uint32_t hashes_end = offsets[text_idx] + (text_idx + 1) * (1 - length);
	//uint32_t hashes_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]) + text_idx * (1 - length);
	uint32_t hashes_end = offsets[text_idx];
	uint32_t hashes_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]);

	for (uint32_t cycle = hashes_start; cycle < hashes_end; cycle += blockDim.x) {
		uint32_t lane = cycle + threadIdx.x;
		uint32_t hash;

		for (uint32_t i = 0; i < NUM_HASH_FUNCTIONS; i++) {
			// we use this kind of hacky approach to always have all 32 threads in the warp reduction (__shfl_down_sync)
			// masking out some of them in the last loop-cycle is too much of a mess 
			if (lane < hashes_end) {
				//hash = ((uint64_t)a[i] * shingle_hashes[lane] + b[i]) % 4294967311ULL;
				hash = a[i] * shingle_hashes[lane] + b[i];
			}
			else {
				hash = UINT32_MAX;
			}
			for (int offset = 16; offset > 0; offset /= 2) {
				hash = min(hash, __shfl_down_sync(0xffffffff, hash, offset));
			}
			if (threadIdx.x == 0) {
				shm[i] = min(shm[i], hash);
			}
		}
	}

	__syncwarp();

	for (size_t i = threadIdx.x; i < NUM_HASH_FUNCTIONS; i += blockDim.x) {
		signatures[text_idx * NUM_HASH_FUNCTIONS + i] = shm[i];
		//printf("thread: %d, shm: %d\n", threadIdx.x, shm[i]);
	}

}


//interesting: if we start with NUM_HASH_FUNCTIONS = 128, runtime varies between 2 and 75 seconds (????)
//if we use 64 and compute two positions per loop, we and up at 2 seconds consistent (lmao wtf)
__global__ void minhash(uint32_t* __restrict__ hashes, uint32_t* __restrict__ offsets, uint32_t* __restrict__ signature, uint32_t* __restrict__ a, uint32_t* __restrict__ b) {

	uint32_t text_idx = blockIdx.x;

	//uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]);
	uint32_t data_start = (text_idx == 0 ? 0 : offsets[text_idx - 1]) + text_idx * (1 - k);
	//uint32_t data_end = offsets[text_idx];
	uint32_t data_end = offsets[text_idx] + (text_idx + 1) * (1 - k);

	uint32_t a_h = a[threadIdx.x];
	uint32_t b_h = b[threadIdx.x];
	uint32_t c_h = a[threadIdx.x + 64];
	uint32_t d_h = b[threadIdx.x + 64];
	uint32_t tmp;

	uint32_t hash;
	uint32_t minimum1 = UINT32_MAX;
	uint32_t minimum2 = UINT32_MAX;

	//int read_off = threadIdx.x & 0x7;


	for (uint32_t i = data_start; i < data_end; i++) {

		tmp = hashes[i];

		hash = a_h * tmp + b_h;
		minimum1 = min(hash, minimum1);

		hash = c_h * tmp + d_h;
		minimum2 = min(hash, minimum2);
		//hash ^= hash >> 16;
		//hash *= 0x7feb352d;
		//hash ^= hash >> 15;

	}
	signature[text_idx * NUM_HASH_FUNCTIONS + threadIdx.x] = minimum1;
	signature[text_idx * NUM_HASH_FUNCTIONS + threadIdx.x + 64] = minimum2;
}





__global__ void fill(uint32_t* dest, uint32_t length) {

	int lane = threadIdx.x;

	for (int i = lane; i < length; i += blockDim.x) {
		dest[i] = blockIdx.x * 100 + threadIdx.x;
	}
}

__global__ void fill2(uint32_t* dest, uint32_t length, uint32_t max_val) {

	int lane = threadIdx.x;

	for (int i = lane; i < length; i += blockDim.x) {
		dest[i] = min((i + 1) * 200, max_val);
	}

}

__global__ void fill3(uint8_t* buffer, uint32_t length) {

	for (int i = threadIdx.x; i < length; i += blockDim.x) {
		buffer[i] = (i * i + i);
	}
}

/*
int main()
{

	//test_pipeline();


}

int test_pipeline() {
	uint32_t* a;
	uint32_t* b;

	cudaFree(0);
	cudaDeviceSynchronize();
	size_t free_mem, total_mem;


	static_assert(k > 0, "shingle-length has to be > 0");
	const uint64_t text_size = 1024 * 1024 * 1024;
	const uint64_t num_texts = text_size / 200;
	const uint64_t offset_size = num_texts * sizeof(uint32_t);
	const uint64_t hashes_size = (text_size - num_texts * (k - 1)) * sizeof(uint32_t);
	const uint64_t signature_size = num_texts * NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t);

	const uint64_t hash_functions_size = NUM_HASH_FUNCTIONS * 2 * 2 * sizeof(uint32_t);

	const uint64_t required_memory = max(text_size, signature_size) + offset_size + hashes_size + hash_functions_size;

	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "free GPU Memory: " << free_mem << ", required GPU Memory: " << required_memory << std::endl;

	if (free_mem < required_memory) {
		std::cerr << "out of memory" << std::endl;
		return 1;
	}

	uint32_t x[] = { 2461073977, 4068265167, 1963639998, 4188804150, 1597603359, 685210601, 2220169462, 1898454865, 367060194, 3008642886, 1589030268, 1616821384, 3627590951, 1669470000, 2909847533, 4034144575, 2718905707, 3446680356, 3421991419, 2621724595, 2758513502, 2668464211, 2183142999, 3427830900, 1458577010, 83251161, 3799688785, 3419784521, 1640714226, 2936566898, 1948489329, 259256096, 2715610451, 3569507854, 2683418592, 537015816, 167984908, 3112576337, 2847269432, 1409804995, 1282483913, 2210268224, 2106705091, 3038709818, 551908191, 4055931993, 3932006546, 868676788, 1224531371, 2761743448, 4044535515, 1632880471, 2831905232, 579290406, 2467764078, 1258232538, 1514729904, 2084218199, 2971137878, 1892304382, 1251504018, 166958372, 3750459229, 2705136087, 1980218234, 1655285799, 1411888306, 274801197, 3530907986, 2687838608, 4164046837, 3338013560, 1123877287, 4058456770, 2401659054, 2330456067, 2229554264, 4091340450, 724655294, 2855898253, 2536541683, 614327932, 3975927644, 3686331917, 2856101398, 3451404009, 3945427099, 1070379979, 198527685, 1729907791, 3015599603, 3582146635, 774144649, 2823731578, 2521978147, 3813505280, 2412448396, 3994089931, 1693971116, 185452569, 539341057, 1870045304, 1206121220, 519076618, 1323465208, 4138483493, 307954332, 2337439688, 2308710946, 3744977555, 2368707081, 1716898113, 940543714, 3343254572, 960393533, 2723891258, 1041544634, 3925195877, 3819006552, 4086973939, 2260290730, 3954146092, 3210229665, 3050620709, 1197419521, 812692070, 943682205, 961395201 };
	uint32_t y[] = { 1570419827, 2951223800, 2827355939, 2044102183, 4057428307, 2501719356, 3283176262, 507623698, 1339505574, 3653461140, 3375066364, 380055397, 1015153906, 2580738893, 1117532264, 3925283057, 3445071761, 2283374130, 2032022830, 4221739735, 855508289, 2570227961, 2786453719, 957193324, 854515079, 2915303401, 1765661319, 1233653554, 1073987701, 2677611593, 706067201, 142690687, 1937914976, 3486252055, 2505048854, 392547294, 3665222936, 3411224521, 940726855, 695079259, 3025718235, 4290176028, 786745122, 612886497, 4175279516, 2112934219, 4240180129, 150792891, 1399579579, 4233915487, 3049722581, 1752369884, 3329151360, 3933055975, 785109753, 4044942470, 3663567947, 191090222, 2962989855, 4114663115, 2820197471, 2657135885, 1931482324, 4202328315, 2627181833, 1711876737, 587922347, 411447709, 992781311, 2189879108, 283895015, 3632149727, 1838354200, 3522342102, 1252390762, 513104615, 1271367373, 3877628913, 530593740, 1235237300, 3196309764, 2432553243, 2027404774, 437964364, 2480347037, 1155688848, 857969502, 2959479920, 149314159, 10261167, 3800455136, 3381359135, 3209115848, 1043937696, 980440598, 2313495643, 1121275014, 433096834, 1686340418, 4030932736, 3021657628, 1598334742, 4197833321, 3099140232, 1935505811, 870742622, 132325799, 1260723603, 353860042, 2882637216, 2397180549, 1036096199, 1361981717, 3216170423, 2090664817, 1427339081, 1608234190, 2577297807, 2846206486, 2467163145, 1948467913, 1809780980, 2478479022, 1203199685, 1274100715, 2791776856, 2201133065, 2810274986 };

	cudaError_t error;

	if ((error = cudaMalloc((void**)&a, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t))) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		return 1;
	}
	if ((error = cudaMalloc((void**)&b, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t))) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		return 1;
	}

	cudaMemcpy(a, x, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(b, y, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	uint8_t* buffer = nullptr;
	uint32_t buffer_size = max(text_size, signature_size);
	if ((error = cudaMalloc((void**)&buffer, buffer_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		return 1;
	}

	fill3 << <1, 1024 >> > (buffer, text_size);

	uint32_t* offsets = nullptr;
	if ((error = cudaMalloc((void**)&offsets, offset_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		return 1;
	}

	fill2 << <1, 1024 >> > (offsets, num_texts, text_size);

	uint32_t* hashes = nullptr;
	if ((error = cudaMalloc((void**)&hashes, hashes_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		return 1;
	}


	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return 1;
	}

	GpuTimer timer;
	std::cout << "start murmur3" << std::endl;
	timer.Start();

	/*
	murmur3_32<<<num_texts, 32>>>(buffer, offsets, hashes, k, 0x01);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return 1;
	}

	//reuse text-buffer for signatures
	uint32_t* signatures = (uint32_t*)buffer;
	std::cout << "start minhash" << std::endl;

	minhash << <num_texts, NUM_HASH_FUNCTIONS >> > (hashes, offsets, signatures, a, b);


	pipeline << <num_texts, 64 >> > (buffer, offsets, hashes, a, b, k, 0x01);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return 1;
	}


	timer.Stop();
	std::cout << "time: " << timer.Elapsed() << "ms" << std::endl;

	return 0;

}

void test_minhash() {
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<uint32_t> distribution(1, UINT32_MAX);
	uint32_t* a;
	uint32_t* b;

	cudaFree(0);
	cudaDeviceSynchronize();
	size_t free_mem, total_mem;


	uint32_t x[] = { 2461073977, 4068265167, 1963639998, 4188804150, 1597603359, 685210601, 2220169462, 1898454865, 367060194, 3008642886, 1589030268, 1616821384, 3627590951, 1669470000, 2909847533, 4034144575, 2718905707, 3446680356, 3421991419, 2621724595, 2758513502, 2668464211, 2183142999, 3427830900, 1458577010, 83251161, 3799688785, 3419784521, 1640714226, 2936566898, 1948489329, 259256096, 2715610451, 3569507854, 2683418592, 537015816, 167984908, 3112576337, 2847269432, 1409804995, 1282483913, 2210268224, 2106705091, 3038709818, 551908191, 4055931993, 3932006546, 868676788, 1224531371, 2761743448, 4044535515, 1632880471, 2831905232, 579290406, 2467764078, 1258232538, 1514729904, 2084218199, 2971137878, 1892304382, 1251504018, 166958372, 3750459229, 2705136087, 1980218234, 1655285799, 1411888306, 274801197, 3530907986, 2687838608, 4164046837, 3338013560, 1123877287, 4058456770, 2401659054, 2330456067, 2229554264, 4091340450, 724655294, 2855898253, 2536541683, 614327932, 3975927644, 3686331917, 2856101398, 3451404009, 3945427099, 1070379979, 198527685, 1729907791, 3015599603, 3582146635, 774144649, 2823731578, 2521978147, 3813505280, 2412448396, 3994089931, 1693971116, 185452569, 539341057, 1870045304, 1206121220, 519076618, 1323465208, 4138483493, 307954332, 2337439688, 2308710946, 3744977555, 2368707081, 1716898113, 940543714, 3343254572, 960393533, 2723891258, 1041544634, 3925195877, 3819006552, 4086973939, 2260290730, 3954146092, 3210229665, 3050620709, 1197419521, 812692070, 943682205, 961395201 };
	uint32_t y[] = { 1570419827, 2951223800, 2827355939, 2044102183, 4057428307, 2501719356, 3283176262, 507623698, 1339505574, 3653461140, 3375066364, 380055397, 1015153906, 2580738893, 1117532264, 3925283057, 3445071761, 2283374130, 2032022830, 4221739735, 855508289, 2570227961, 2786453719, 957193324, 854515079, 2915303401, 1765661319, 1233653554, 1073987701, 2677611593, 706067201, 142690687, 1937914976, 3486252055, 2505048854, 392547294, 3665222936, 3411224521, 940726855, 695079259, 3025718235, 4290176028, 786745122, 612886497, 4175279516, 2112934219, 4240180129, 150792891, 1399579579, 4233915487, 3049722581, 1752369884, 3329151360, 3933055975, 785109753, 4044942470, 3663567947, 191090222, 2962989855, 4114663115, 2820197471, 2657135885, 1931482324, 4202328315, 2627181833, 1711876737, 587922347, 411447709, 992781311, 2189879108, 283895015, 3632149727, 1838354200, 3522342102, 1252390762, 513104615, 1271367373, 3877628913, 530593740, 1235237300, 3196309764, 2432553243, 2027404774, 437964364, 2480347037, 1155688848, 857969502, 2959479920, 149314159, 10261167, 3800455136, 3381359135, 3209115848, 1043937696, 980440598, 2313495643, 1121275014, 433096834, 1686340418, 4030932736, 3021657628, 1598334742, 4197833321, 3099140232, 1935505811, 870742622, 132325799, 1260723603, 353860042, 2882637216, 2397180549, 1036096199, 1361981717, 3216170423, 2090664817, 1427339081, 1608234190, 2577297807, 2846206486, 2467163145, 1948467913, 1809780980, 2478479022, 1203199685, 1274100715, 2791776856, 2201133065, 2810274986 };

	cudaMalloc((void**)&a, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t));
	cudaMalloc((void**)&b, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t));

	cudaMemcpy(a, x, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(b, y, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);


	// 1 GiB an fiktiven hashes
	const uint32_t num_hashes = 1024 * 1024 * 1024;
	// avg. 200 hashes pro artikel
	const uint32_t num_texts = num_hashes / 200;

	uint32_t* hashes;
	cudaMalloc((void**)&hashes, sizeof(uint32_t) * num_hashes);
	fill << <1, 1024 >> > (hashes, num_hashes);


	uint32_t* offsets;
	cudaMalloc((void**)&offsets, sizeof(uint32_t) * num_texts);
	fill2 << <1, 1024 >> > (offsets, num_texts, num_hashes);


	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return;
	}


	uint32_t* signatures;
	cudaMalloc((void**)&signatures, NUM_HASH_FUNCTIONS * 2 * sizeof(uint32_t) * num_texts);

	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "free GPU Memory: " << free_mem << ", total GPU Memory: " << total_mem << std::endl;

	GpuTimer timer;
	std::cout << "start" << std::endl;
	timer.Start();


	minhash << <num_texts, NUM_HASH_FUNCTIONS >> > (hashes, offsets, signatures, a, b);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return;
	}

	timer.Stop();
	//avg. ca 2034 ms.
	std::cout << "time: " << timer.Elapsed() << "ms" << std::endl;

}

int test_murmur3_32_3() {
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<uint32_t> distribution(1, 0xFF);

	cudaFree(0);
	cudaDeviceSynchronize();
	size_t free_mem, total_mem;
	cudaError_t error;

	const uint32_t length = 1024 * 1024 * 1024;
	const uint32_t num_texts = length / 200;
	uint32_t k = 4;

	const uint32_t offset_size = num_texts * sizeof(uint32_t);
	const uint32_t text_size = length;
	const uint64_t hashes_size = (text_size + num_texts * (1 - k)) * sizeof(uint32_t);

	std::cout << "required memory: " << text_size + offset_size + hashes_size << std::endl;
	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "free GPU Memory: " << free_mem << ", total GPU Memory: " << total_mem << std::endl;

	uint8_t* buffer;
	if ((error = cudaMalloc((void**)&buffer, text_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
	}
	fill3 << <1, 1024 >> > (buffer, text_size);


	uint32_t* offsets;
	if ((error = cudaMalloc((void**)&offsets, offset_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
	}
	fill2 << <1, 1024 >> > (offsets, num_texts, text_size);


	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return 1;
	}

	uint32_t* hashes;
	if ((error = cudaMalloc((void**)&hashes, hashes_size)) != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
	}



	GpuTimer timer;
	std::cout << "start" << std::endl;
	timer.Start();

	murmur3_32 << <num_texts, 128 >> > (buffer, offsets, hashes, k, 0x01);


	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return 1;
	}

	timer.Stop();
	//avg. ca 2034 ms.
	std::cout << "time: " << timer.Elapsed() << "ms" << std::endl;
	return 0;

}

bool test_murmur3_32() {

	const uint32_t correct_hash = 0x24884CBA;
	const uint32_t seed = 0x9747b28c;
	const uint32_t threads_per_block = 32;

	//std::string text = "testHello, world!The quick brown fox jumps over the lazy dog";
	//std::string text = "Hello, world!Hello, world!Hello, world!Hello, world!";
	std::string text = "Hello, world!";
	int num_texts = 1;
	int k = 13;

	uint32_t* offsets, * hashes;
	cudaMallocManaged((void**)&offsets, num_texts * sizeof(uint32_t), cudaMemAttachGlobal);

	uint32_t num_hashes = text.size() - num_texts * (k - 1);
	cudaMallocManaged((void**)&hashes, num_hashes * sizeof(uint32_t), cudaMemAttachGlobal);

	offsets[0] = text.size();


	uint8_t* buffer;
	cudaMallocManaged((void**)&buffer, 256, cudaMemAttachGlobal);
	cudaMemcpy(buffer, text.c_str(), text.size(), cudaMemcpyHostToDevice);


	murmur3_32 << <num_texts, threads_per_block >> > (buffer, offsets, hashes, k, seed);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return false;
	}

	/*
	for(int i = 0; i < num_hashes; i++) {
		std::cout << "hashes[" << i << "] = " << hashes[i] << std::endl;
	}
	uint32_t positions[] = { 0, 13, 26, 39 };

	for (auto& position : positions) {
		if (hashes[position] != correct_hash) {
			std::cerr << "Hash mismatch at position " << position << ": expected " << correct_hash << ", got " << hashes[position] << std::endl;
			return 0;
		}
	}
	return false;
}

bool test_murmur3_32_2() {
	const uint32_t correct_hash = 0x24884CBA;
	const uint32_t seed = 0x9747b28c;
	const uint32_t threads_per_block = 32;

	//std::string text = "testHello, world!The quick brown fox jumps over the lazy dog";
	//std::string text = "Hello, world!Hello, world!Hello, world!Hello, world!";
	std::string text = "Hello, world!";
	int num_texts = 1;
	int k = 4;

	uint32_t* offsets, * hashes;
	cudaMallocManaged((void**)&offsets, num_texts * sizeof(uint32_t), cudaMemAttachGlobal);

	uint32_t num_hashes = text.size() - num_texts * (k - 1);
	cudaMallocManaged((void**)&hashes, num_hashes * sizeof(uint32_t), cudaMemAttachGlobal);

	offsets[0] = text.size();


	uint8_t* buffer;
	cudaMallocManaged((void**)&buffer, 256, cudaMemAttachGlobal);
	cudaMemcpy(buffer, text.c_str(), text.size(), cudaMemcpyHostToDevice);


	murmur3_32 << <num_texts, threads_per_block >> > (buffer, offsets, hashes, k, seed);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cerr << "Kernel execution failed:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
		return false;
	}

	for (int i = 0; i < num_hashes; i++) {
		std::cout << "hashes[" << i << "] = " << hashes[i] << std::endl;
	}
	return false;

}


static int hash_parameters(std::string file_name, int length, uint32_t* a, uint32_t* b)
*/