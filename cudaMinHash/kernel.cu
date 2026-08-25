#include "kernel.cuh"


//murmur3_32 hashfunction. the input data is not 32-byte aligned, so we read it bytewise.  
__device__ __forceinline__ void murmur3_32(uint8_t* __restrict__ data, uint32_t length, uint32_t seed, uint32_t* hash) {

	uint32_t h = seed;
	uint32_t k;

	for (size_t i = 0; i < length / 4; i++) {

		//eventuell unaligned, deswegen byteweise lesen
#pragma unroll
		for (size_t j = 0; j < 4; j++) {
			k = k << 8 | data[i * 4 + 3 - j];
		}

		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;

		h ^= k;
		h = (h << 13) | (h >> 19);
		h = h * 5 + 0xe6546b64;
	}

	k = 0;

	for (size_t i = length & 3; i; i--) {
		k <<= 8;
		k |= data[(length & ~3) + i - 1];
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

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r)
{
	return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t read64(const uint8_t* p)
{
	return
		((uint64_t)p[0]) |
		((uint64_t)p[1] << 8) |
		((uint64_t)p[2] << 16) |
		((uint64_t)p[3] << 24) |
		((uint64_t)p[4] << 32) |
		((uint64_t)p[5] << 40) |
		((uint64_t)p[6] << 48) |
		((uint64_t)p[7] << 56);
}

__device__ __forceinline__ uint32_t read32(const uint8_t* p)
{
	return
		((uint32_t)p[0]) |
		((uint32_t)p[1] << 8) |
		((uint32_t)p[2] << 16) |
		((uint32_t)p[3] << 24);
}

//xxh64 hashfunction. processing of 32-byte blocks is removed.
__device__ __forceinline__ void xxh_64(
	uint8_t* __restrict__ data,
	uint32_t length,
	uint64_t seed,
	uint64_t* hash) {

	constexpr uint64_t XXH_PRIME1 = 11400714785074694791ULL;
	constexpr uint64_t XXH_PRIME2 = 14029467366897019727ULL;
	constexpr uint64_t XXH_PRIME3 = 1609587929392839161ULL;
	constexpr uint64_t XXH_PRIME4 = 9650029242287828579ULL;
	constexpr uint64_t XXH_PRIME5 = 2870177450012600261ULL;

	const uint8_t* p = data;
	const uint8_t* end = p + length;

	uint64_t h64;

	//shouldnt be necessary, shingle-length is usually smaller than 32
	/*
	if (length >= 32)
	{
		const uint8_t* limit = end - 32;

		uint64_t v1 = seed + XXH_PRIME1 + XXH_PRIME2;
		uint64_t v2 = seed + XXH_PRIME2;
		uint64_t v3 = seed;
		uint64_t v4 = seed - XXH_PRIME1;

		do
		{
			v1 += read64(p) * XXH_PRIME2;
			v1 = rotl64(v1, 31);
			v1 *= XXH_PRIME1;
			p += 8;

			v2 += read64(p) * XXH_PRIME2;
			v2 = rotl64(v2, 31);
			v2 *= XXH_PRIME1;
			p += 8;

			v3 += read64(p) * XXH_PRIME2;
			v3 = rotl64(v3, 31);
			v3 *= XXH_PRIME1;
			p += 8;

			v4 += read64(p) * XXH_PRIME2;
			v4 = rotl64(v4, 31);
			v4 *= XXH_PRIME1;
			p += 8;

		} while (p <= limit);

		h64 =
			rotl64(v1, 1) +
			rotl64(v2, 7) +
			rotl64(v3, 12) +
			rotl64(v4, 18);

		v1 *= XXH_PRIME2;
		v1 = rotl64(v1, 31);
		v1 *= XXH_PRIME1;
		h64 ^= v1;
		h64 = h64 * XXH_PRIME1 + XXH_PRIME4;

		v2 *= XXH_PRIME2;
		v2 = rotl64(v2, 31);
		v2 *= XXH_PRIME1;
		h64 ^= v2;
		h64 = h64 * XXH_PRIME1 + XXH_PRIME4;

		v3 *= XXH_PRIME2;
		v3 = rotl64(v3, 31);
		v3 *= XXH_PRIME1;
		h64 ^= v3;
		h64 = h64 * XXH_PRIME1 + XXH_PRIME4;

		v4 *= XXH_PRIME2;
		v4 = rotl64(v4, 31);
		v4 *= XXH_PRIME1;
		h64 ^= v4;
		h64 = h64 * XXH_PRIME1 + XXH_PRIME4;
	}
	else
	{
		h64 = seed + XXH_PRIME5;
	}
	*/

	h64 = seed + XXH_PRIME5;
	h64 += length;

	while (p + 8 <= end)
	{
		uint64_t k1 = read64(p);

		k1 *= XXH_PRIME2;
		k1 = rotl64(k1, 31);
		k1 *= XXH_PRIME1;

		h64 ^= k1;

		h64 = rotl64(h64, 27);
		h64 = h64 * XXH_PRIME1 + XXH_PRIME4;

		p += 8;
	}

	if (p + 4 <= end)
	{
		h64 ^= (uint64_t)read32(p) * XXH_PRIME1;

		h64 = rotl64(h64, 23);
		h64 = h64 * XXH_PRIME2 + XXH_PRIME3;

		p += 4;
	}

	while (p < end)
	{
		h64 ^= (*p) * XXH_PRIME5;

		h64 = rotl64(h64, 11);
		h64 *= XXH_PRIME1;

		++p;
	}

	h64 ^= h64 >> 33;
	h64 *= XXH_PRIME2;

	h64 ^= h64 >> 29;
	h64 *= XXH_PRIME3;

	h64 ^= h64 >> 32;

	*hash = h64;

}


__global__ void shingle_hash32_kernel(
	uint8_t* __restrict__ buffer,
	uint32_t* __restrict__ offsets,
	uint32_t* __restrict__ hashes,
	int k,
	int seed) {

	uint32_t text_num = blockIdx.x;

	uint32_t data_start = (text_num == 0 ? 0 : offsets[text_num - 1]) + threadIdx.x;
	uint32_t data_end = offsets[text_num] - k + 1;


	for (; data_start < data_end; data_start += blockDim.x) {
		murmur3_32(&buffer[data_start], k, seed, &hashes[data_start - text_num * (k - 1)]);
	}

}

__global__ void shingle_hash64_kernel(
	uint8_t* __restrict__ buffer,
	uint32_t* __restrict__ offsets,
	uint64_t* __restrict__ hashes,
	int k,
	int seed) {

	uint32_t text_num = blockIdx.x;

	uint32_t data_start = (text_num == 0 ? 0 : offsets[text_num - 1]) + threadIdx.x;
	uint32_t data_end = offsets[text_num] - k + 1;


	for (; data_start < data_end; data_start += blockDim.x) {
		xxh_64(&buffer[data_start], k, seed, &hashes[data_start - text_num * (k - 1)]);
	}

}

__device__ __inline__ void minhash_signature32(
	uint32_t* __restrict__ shingle_hashes,
	uint32_t* __restrict__ offsets,
	uint32_t* __restrict__ signature,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k) {

	uint32_t text_num = blockIdx.x;
	//offsets basieren auf den texten, muessen also auf shingles angepasst werden
	uint32_t data_start = (text_num == 0 ? 0 : (offsets[text_num - 1] - text_num * (k - 1)));
	uint32_t data_end = offsets[text_num] - (text_num + 1) * (k - 1);
	//uint32_t data_start = (doc_num == 0 ? 0 : offsets[doc_num - 1]);
	//uint32_t data_end = offsets[doc_num];

	uint64_t a = A[threadIdx.x];
	uint64_t b = B[threadIdx.x];
	uint32_t minimum = UINT32_MAX;


	uint32_t hash;
	for (; data_start < data_end; data_start++) {
		//((a * x + b) mod 2^64) div 2^32
		hash = (uint32_t)((a * shingle_hashes[data_start] + b) >> 32);
		minimum = min(hash, minimum);
	}
	signature[text_num * NUM_HASH_FUNCTIONS + threadIdx.x] = minimum;
}

__device__ __inline__ void minhash_signature64(
	uint64_t* __restrict__ shingle_hashes,
	uint32_t* __restrict__ offsets,
	uint64_t* __restrict__ signature,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k) {

	uint32_t text_num = blockIdx.x;
	uint32_t data_start = (text_num == 0 ? 0 : (offsets[text_num - 1] - text_num * (k - 1)));
	uint32_t data_end = offsets[text_num] - (text_num + 1) * (k - 1);

	uint64_t a_h = A[2 * threadIdx.x];
	uint64_t a_l = A[2 * threadIdx.x + 1];
	uint64_t b_h = B[2 * threadIdx.x];
	uint64_t b_l = B[2 * threadIdx.x + 1];
	uint64_t minimum = UINT64_MAX;


	uint64_t hash;
	for (; data_start < data_end; data_start++) {

		//((a * x + b) mod 2^128) div 2^64
		uint64_t x = shingle_hashes[data_start];
		uint64_t ahx_l = a_h * x;
		uint64_t alx_h = __umul64hi(a_l, x);
		uint64_t alx_l = a_l * x;
		int carry = (alx_l + b_l < alx_l) ? 1 : 0;

		hash = ahx_l + alx_h + b_h + carry;

		minimum = min(hash, minimum);
	}
	signature[text_num * NUM_HASH_FUNCTIONS + threadIdx.x] = minimum;
}

__device__ __inline__ void bucketing32(
	uint32_t* __restrict__ signatures,
	uint32_t* __restrict__ destination,
	int num_texts,
	int seed) {

	uint32_t text_num = blockIdx.x;
	uint32_t* read_ptr, * write_ptr;

	for (size_t i = threadIdx.x; i < BUCKET_LANES; i += blockDim.x) {

		read_ptr = &signatures[text_num * NUM_HASH_FUNCTIONS + i * LANE_SIZE];
		write_ptr = &destination[text_num * BUCKET_LANES + i];
		//reihenfolge gleich aendern fuer spaetere datenstruktur
		//write_ptr = &destination[i * num_texts + text_num];

		murmur3_32((uint8_t*)read_ptr, LANE_SIZE * sizeof(uint32_t), seed, write_ptr);
	}
}

__device__ __inline__ void bucketing64(
	uint64_t* __restrict__ signatures,
	uint64_t* __restrict__ destination,
	int num_texts,
	int seed) {

	uint32_t text_num = blockIdx.x;
	uint64_t* read_ptr, * write_ptr;

	for (size_t i = threadIdx.x; i < BUCKET_LANES; i += blockDim.x) {

		read_ptr = &signatures[text_num * NUM_HASH_FUNCTIONS + i * LANE_SIZE];
		write_ptr = &destination[text_num * BUCKET_LANES + i];
		//reihenfolge gleich aendern fuer spaetere datenstruktur
		//write_ptr = &destination[i * num_texts + text_num];

		xxh_64((uint8_t*)read_ptr, LANE_SIZE * sizeof(uint64_t), seed, write_ptr);
	}
}

__global__ void minhash32_kernel(
	uint8_t* buffer,
	uint32_t* shingle_hashes,
	uint32_t* offsets,
	uint32_t num_texts,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k,
	int seed) {
	minhash_signature32(shingle_hashes, offsets, (uint32_t*)buffer, A, B, k);
	__syncthreads();
	bucketing32((uint32_t*)buffer, (uint32_t*)&buffer[NUM_HASH_FUNCTIONS * num_texts * sizeof(uint32_t)], num_texts, seed);
}

__global__ void minhash64_kernel(
	uint8_t* buffer,
	uint64_t* shingle_hashes,
	uint32_t* offsets,
	uint32_t num_texts,
	uint64_t* __restrict__ A,
	uint64_t* __restrict__ B,
	int k,
	int seed) {
	minhash_signature64(shingle_hashes, offsets, (uint64_t*)buffer, A, B, k);
	__syncthreads();
	bucketing64((uint64_t*)buffer, (uint64_t*)&buffer[NUM_HASH_FUNCTIONS * num_texts * sizeof(uint64_t)], num_texts, seed);
}



namespace gpu {

	uint64_t memory_required(int num_texts, int text_size, int shingle_length, int signature_length, int num_buckets, int hashes_size) {

		uint64_t shingle_hashes_size = ((uint64_t)text_size + (uint64_t)num_texts * (1 - shingle_length)) * hashes_size;
		uint64_t signature_size = (uint64_t)num_texts * signature_length * hashes_size;
		uint64_t buckets_size = (uint64_t)num_texts * num_buckets * hashes_size;

		uint64_t buffer1_size = signature_size > text_size ? signature_size : text_size;
		uint64_t buffer2_size = shingle_hashes_size > buckets_size ? shingle_hashes_size : buckets_size;

		uint64_t step1_size = buffer1_size + num_texts * sizeof(uint32_t) + shingle_hashes_size;
		uint64_t step2_size = signature_size + buckets_size;

		return step1_size > step2_size ? step1_size : step2_size;

	}

	void cuda_debug(cudaError_t cuda_status, std::string message) {
		if (cuda_status != cudaSuccess) {
			std::cerr << "error: " << message << ": " << cudaGetErrorString(cuda_status) << std::endl;
			exit(1);
		}
	}

	void load_hash_parameters_to_gpu(std::vector<uint64_t>& A, std::vector<uint64_t>& B, uint64_t* gpu_A, uint64_t* gpu_B) {
		uint64_t size = NUM_HASH_FUNCTIONS * sizeof(uint64_t);
		cudaMalloc(&gpu_A, size);
		cudaMemcpy(gpu_A, A.data(), size, cudaMemcpyHostToDevice);
		cudaMalloc(&gpu_B, size);
		cudaMemcpy(gpu_B, B.data(), size, cudaMemcpyHostToDevice);
	}


	template <typename T>
	void minhash(std::vector<uint8_t>& texts, std::vector<uint32_t>& offsets, std::vector<T>& buckets, int k, int seed) {

		size_t free_bytes, total_bytes;
		cudaMemGetInfo(&free_bytes, &total_bytes);
		std::cout << "gpu-memory: total = " << total_bytes << " bytes, free = " << free_bytes << " bytes" << std::endl;

		uint32_t text_size = texts.size();
		uint32_t num_texts = offsets.size();
		std::cout << "gpu-memory required: " << memory_required(num_texts, text_size, k, NUM_HASH_FUNCTIONS, BUCKET_LANES, sizeof(T)) << std::endl;

		//texts
		uint8_t* gpu_buffer1;
		uint64_t size1 = (uint64_t)num_texts * NUM_HASH_FUNCTIONS * sizeof(T);
		uint64_t size2 = (uint64_t)num_texts * text_size * sizeof(uint8_t);
		cudaMalloc(&gpu_buffer1, size1 > size2 ? size1 : size2);
		cudaMemcpy(gpu_buffer1, texts.data(), texts.size(), cudaMemcpyHostToDevice);


		//shingle-hashes
		T* gpu_buffer2;
		size1 = ((uint64_t)num_texts * text_size + num_texts * (1 - k)) * sizeof(T);
		size2 = (uint64_t)num_texts * BUCKET_LANES * sizeof(T);
		cudaMalloc(&gpu_buffer, size1 > size2 ? size1 : size2);


		//offsets
		uint32_t* gpu_offsets;
		size1 = num_texts * sizeof(uint32_t);
		cudaMalloc(&gpu_offsets, num_texts * sizeof(uint32_t));
		cudaMemcpy(gpu_offsets, offsets.data(), num_texts * sizeof(uint32_t), cudaMemcpyHostToDevice);

		cuda_debug(cudaDeviceSynchronize(), "fill gpu memory");

		GpuTimer timer;

		timer.Start();

		if (std::is_same_v<T, uint32_t>) {
			shingle_hash32_kernel << <num_texts, 128 >> > (gpu_buffer1, gpu_offsets, buffer2, k, seed);
		}
		else if (std::is_same_v<T, uint64_t>) {
			shingle_hash64_kernel << <num_texts, 128 >> > (gpu_buffer1, gpu_offsets, buffer2, k, seed);
		}
		else {
			std::cout << "invalid datatype" << std::endl;
			exit(1);
		}

		if ((error = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cout << "error shingle-hashing: " << cudaGetErrorString(error) << std::endl;
			exit(1);
		}

		if (std::is_same_v<T, uint32_t>) {
			minhash32_kernel << <num_texts, NUM_HASH_FUNCTIONS >> > (buffer1, buffer2, gpu_offsets, num_texts, gpu_A, gpu_B, k, 13);
		}
		else if (std::is_same_v<T, uint64_t>) {
			minhash64_kernel << <num_texts, NUM_HASH_FUNCTIONS >> > (buffer1, buffer2, gpu_offsets, num_texts, gpu_A, gpu_B, k, 13);
		}

		if ((error = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cout << "error minhashing: " << cudaGetErrorString(error) << std::endl;
			exit(1);
		}

		timer.Stop();

		std::cout << "elapsed: " << timer.Elapsed() << " ms" << std::endl;

		cudaFree(gpu_buffer1);
		cudaFree(gpu_buffer2);
		cudaFree(gpu_offsets);

	}
}
