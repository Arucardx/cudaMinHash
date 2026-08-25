#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <type_traits>
#include "gputimer.h"
#include "cudaargs.h"
#include "cuda.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#define NUM_HASH_FUNCTIONS 256
#define BUCKET_LANES (NUM_HASH_FUNCTIONS / 2)
#define LANE_SIZE (NUM_HASH_FUNCTIONS / BUCKET_LANES)


__device__ __forceinline__ void murmur3_32(
	uint8_t* __restrict__ data,
	uint32_t length,
	uint32_t seed,
	uint32_t* hash
);

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r);

__device__ __forceinline__ uint64_t read64(const uint8_t* p);

__device__ __forceinline__ uint32_t read32(const uint8_t* p);

__device__ __forceinline__ void xxh_64(
	uint8_t* __restrict__ data,
	uint32_t length,
	uint64_t seed,
	uint64_t* hash
);

__global__ void shingle_hash32_kernel(
	uint8_t* __restrict__ buffer,
	uint32_t* __restrict__ offsets,
	uint32_t* __restrict__ hashes,
	int k,
	int seed
);

__global__ void shingle_hash64_kernel(
	uint8_t* __restrict__ buffer,
	uint32_t* __restrict__ offsets,
	uint64_t* __restrict__ hashes,
	int k,
	int seed
);

__device__ __inline__ void minhash_signature32(
	uint32_t* __restrict__ shingle_hashes,
	uint32_t* __restrict__ offsets,
	uint32_t* __restrict__ signature,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k
);

__device__ __inline__ void minhash_signature64(
	uint64_t* __restrict__ shingle_hashes,
	uint32_t* __restrict__ offsets,
	uint64_t* __restrict__ signature,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k
);

__device__ __inline__ void bucketing32(
	uint32_t* __restrict__ signatures,
	uint32_t* __restrict__ destination,
	int num_texts,
	int seed
);

__device__ __inline__ void bucketing64(
	uint64_t* __restrict__ signatures,
	uint64_t* __restrict__ destination,
	int num_texts,
	int seed
);


__global__ void minhash32_kernel(
	uint8_t* buffer,
	uint32_t* shingle_hashes,
	uint32_t* offsets,
	uint32_t num_texts,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k,
	int seed
);

__global__ void minhash64_kernel(
	uint8_t* buffer,
	uint64_t* shingle_hashes,
	uint32_t* offsets,
	uint32_t num_texts,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k,
	int seed
);


namespace gpu {

	uint64_t memory_required(
		int num_texts,
		int text_size,
		int shingle_length,
		int signature_length,
		int num_buckets,
		int hashes_size
	);

	void cuda_debug(cudaError_t cuda_status, std::string message);

	void load_hash_parameters_to_gpu(std::vector<uint64_t>& A, std::vector<uint64_t>& B, uint64_t* gpu_A, uint64_t* gpu_B);

	template <typename T>
	void minhash(std::vector<uint8_t>& texts, std::vector<uint32_t>& offsets, std::vector<T>& buckets, int k, int seed);
}
