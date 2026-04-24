#pragma once 
#include "cudaargs.h"
#include "cuda.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <iostream>
#include <random>
#include <chrono>
#include <string>

#define NUM_HASH_FUNCTIONS 128
#define BUCKET_LANES 64
#define LANE_SIZE (NUM_HASH_FUNCTIONS / BUCKET_LANES)
#define MINHASH_THREADS 64
#define MAX_MEMORY (1024 * 1024 * 2)

__constant__ uint32_t* offset_ptr;
__constant__ uint32_t* hashes_ptr;
__constant__ uint32_t* offsets_off;

__device__ __inline__ void murmur3_32(uint8_t* __restrict__ buffer, uint32_t length, uint32_t seed, uint32_t* hash) {

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

__device__ __inline__ void shingle_hash(uint8_t* __restrict__ buffer, uint32_t* __restrict__ offsets, uint32_t* __restrict__ hashes, int k, int seed) {

	uint32_t doc_num = blockIdx.x;

	uint32_t data_start = (doc_num == 0 ? 0 : offsets[doc_num - 1]);
	uint32_t data_length = offsets[doc_num];

	for (size_t i = threadIdx.x; i < data_length - k + 1; i += blockDim.x) {
		murmur3_32(&buffer[data_start + i], k, seed, &hashes[data_start + doc_num * (1 - k) + i]);
	}

}

__device__ void minhash_signature(uint32_t* __restrict__ shingle_hashes, uint32_t* __restrict__ offsets, uint32_t* __restrict__ signature, uint32_t* __restrict__ a, uint32_t* __restrict__ b, int k) {

	uint32_t hash_params[2 * (NUM_HASH_FUNCTIONS / MINHASH_THREADS)];
	uint32_t minima[NUM_HASH_FUNCTIONS / MINHASH_THREADS];

	uint32_t doc_num = blockIdx.x;
	uint32_t data_start = (doc_num == 0 ? 0 : (offsets[doc_num - 1] + doc_num * (1 - k)));
	uint32_t data_end = offsets[doc_num] + (doc_num + 1) * (1 - k);

#pragma unroll
	for (size_t i = 0; i < NUM_HASH_FUNCTIONS / MINHASH_THREADS; i++) {
		hash_params[2 * i] = a[threadIdx.x + i * MINHASH_THREADS];
		hash_params[2 * i + 1] = b[threadIdx.x + i * MINHASH_THREADS];
		minima[i] = UINT32_MAX;
	}

	uint32_t tmp, hash;
	for (size_t i = data_start; i < data_end; i++) {
		
		tmp = shingle_hashes[i];
#pragma unroll
		for (size_t j = 0; j < NUM_HASH_FUNCTIONS / MINHASH_THREADS; j++) {
			hash = hash_params[2 * j] * tmp + hash_params[2 * j + 1];
			minima[j] = min(hash, minima[j]);
		}
	}

#pragma unroll
	for (size_t i = 0; i < NUM_HASH_FUNCTIONS / MINHASH_THREADS; i++) {
		signature[doc_num * NUM_HASH_FUNCTIONS + threadIdx.x + i * MINHASH_THREADS] = minima[i];
	}

}

__device__ __inline__ void bucketing(uint32_t* __restrict__ signatures, uint32_t* __restrict__ destination, int seed) {
	
	uint32_t doc_num = blockIdx.x;

	uint32_t *read_offset, *write_offset;
	for (size_t i = threadIdx.x; i < BUCKET_LANES; i+= blockDim.x) {

		read_offset = &signatures[doc_num * NUM_HASH_FUNCTIONS + i * LANE_SIZE];
		write_offset = &destination[i * BUCKET_LANES + doc_num];

		murmur3_32((uint8_t*)read_offset, LANE_SIZE * 4, seed, write_offset);
	}
}

__global__ void minhash_pipeline(uint8_t* __restrict__ buffer, uint32_t* __restrict__ a, uint32_t* __restrict__ b, int k, int seed) {

	shingle_hash(buffer, offset_ptr, hashes_ptr, k, seed);

	minhash_signature(hashes_ptr, offset_ptr, (uint32_t*)buffer, a, b, k);

	//num_docs!
	bucketing((uint32_t*)buffer, (uint32_t*)&buffer[NUM_HASH_FUNCTIONS * 4 * 4], seed);

}



int main() {


	const char* text = "hellohellohellohello";
	int text_length = strlen(text);
	uint32_t offsets[] = { 5, 10, 15, 20};
	int num_docs = 4;

	uint8_t* host_buffer = new uint8_t[MAX_MEMORY];

	auto cudaDebug = [](cudaError_t error) {
		if (error != cudaSuccess) {
			std::cerr << cudaGetErrorString(error) << std::endl;
		}
	};

	uint8_t* buffer;
	cudaDebug(cudaMalloc(&buffer, MAX_MEMORY));
	cudaDebug(cudaMemcpy(buffer, text, text_length, cudaMemcpyHostToDevice));


	//4 byte alignment
	uint32_t offset = (text_length + 3) / 4 * 4;
	cudaDebug(cudaMemcpyToSymbol(offsets_off, &offset, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	//cudaDebug(cudaMemcpyToSymbol(offset_ptr, &buffer[offset], sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	cudaDebug(cudaMemcpy(&buffer[offset], offsets, num_docs * sizeof(uint32_t), cudaMemcpyHostToDevice));

	offset += num_docs * sizeof(uint32_t);
	//cudaDebug(cudaMemcpyToSymbol(&hashes_ptr, &buffer[offset], sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

	//pipeline

	cudaDebug(cudaMemcpy(host_buffer, &buffer[NUM_HASH_FUNCTIONS * num_docs * 4], num_docs * BUCKET_LANES * 4, cudaMemcpyDeviceToHost));

	uint32_t* b = (uint32_t*)host_buffer;

	for (int i = 0; i < BUCKET_LANES; i++) {
		for (int j = 0; j < num_docs; j++) {
			std::cout << b[i * num_docs + j];
		}
		std::cout << std::endl;
	}

	delete[] host_buffer;


}
