from sklearn.utils import murmurhash3_32
from xxhash import xxh64

def shingle(text, k):
    return set([text[i:i+k] for i in range(len(text) - k + 1)])


def jaccard(shingle1 : set, shingle2 : set):
    intersect = shingle1.intersection(shingle2)
    return len(intersect) / (len(shingle1) + len(shingle2) - len(intersect))

def jaccard(text1, text2, k):
    return jaccard(shingle(text1, k), shingle(text2, k), k)

def hash_params(path = './params.txt', num_params = 128):
    params = open(path).readlines()
    assert(num_params <= len(params))

    A, B = [], []
    for i in range(num_params):
        a, b = params[i].strip().split(' ') 
        A.append(int(a))
        B.append(int(b))
    return A, B 


def shingle_hash32(data: bytes, k, hashf, seed = 13):

    if hashf == None:
        hashf = lambda x: murmurhash3_32(x, seed=seed, positive=True)
    return shingle_hash(data, hashf)

def shingle_hash64(data: bytes, k, hashf, seed = 13):

    if hashf == None:
        hashf = lambda x: xxh64(x, seed=seed).intdigest()
    return shingle_hash(data, hashf)


def shingle_hash(data, hashf):
    return [hashf(sh) for sh in shingle(data)]
    

def minhash_signature32(shingle_hashes,  A, B, hashf = None):

    if hashf == None:
        hashf = lambda x, a, b: ((a*x + b) % 2**64) // 2**32

    return minhash_signature(shingle_hashes, A, B, hashf, affine=32)


def minhash_signature64(shingle_hashes, A, B, hashf = None):

    if hashf == None:
        hashf = lambda x, a, b: ((a*x + b) % 2**128) // 2**64

    return minhash_signature(shingle_hashes, A, B, hashf, affine=64)


def minhash_signature(shingle_hashes, A, B, hashf, affine):

    assert(affine in [32, 64])
    if affine == 64:
        max_val = 2**64 - 1
    elif affine == 32:
        max_val = 2**32 - 1

    assert(len(A) == len(B))
    signature = [0] * len(A)
    for i, (a, b) in enumerate(zip(A, B)):
        minimum = max_val
        for x in shingle_hashes:
            minimum = min(minimum, hashf(x, a, b))
        signature[i] = minimum
    return signature


def bucketing32(minhash_signature, beta, gamma, hashf=None, seed=19):
    if hashf == None:
        hashf = lambda x: murmurhash3_32(x, seed, positive=True)

    return bucketing(minhash_signature, beta, gamma, hashf)


def bucketing64(minhash_signature, beta, gamma, hashf=None, seed=19):
    if hashf == None:
        hashf = lambda x: xxh64(x, seed).intdigest()

    return bucketing(minhash_signature, beta, gamma, hashf)


def bucketing(minhash_signature, beta, gamma, hashf):

    assert(beta * gamma == len(minhash_signature))
    buckets = [0] * beta

    for i in range(0, beta*gamma, gamma):
        data = bytes(minhash_signature[i:i+gamma])
        buckets[i] = hashf(data)

    return buckets

    