import numpy as np
import blosc
import warnings

class ZeusEncodedObject:
    def __init__(self,
                 data = None,
                 lookup_length = 255
                ):
        
        assert lookup_length < 256, "Maximum 255 element lookup is only supported now."
        
        self.encoded = None ## Array of type np.uint8 of same size of data
        self.raw = None ## Raw values that could not be encoded
        self.lookup = None ## Full precision arraw containing the raw values of the encoded array
        
        self.lookup_length = lookup_length
        
        self.SENTINEL = 255
        
        if data is not None:
            self.encode(data)
            
    def encode(self,data):
        
        values, counts = np.unique(data,return_counts=True)
        top_counts = np.argsort(counts)[-self.lookup_length:]
        
        self.encoded = np.empty(shape=data.shape,
                                dtype=np.uint8
                                )
        
        self.encoded[:] = self.SENTINEL
        self.lookup = values[top_counts]
        
        for i in range(self.lookup_length):
            self.encoded[data == self.lookup[i]] = i
        
        self.raw = data[self.encoded == self.SENTINEL]
        
    def decode(self):
        
        decoded = np.empty(shape=self.encoded.shape, dtype=self.raw.dtype)
        mask = self.encoded == self.SENTINEL
        decoded[np.where(mask)] = self.raw
        decoded[np.where(~mask)] = self.lookup[self.encoded[~mask]]
        
        return decoded
        
    def __array__(self, dtype=None, copy=None):
        return self.decode()
    
    @property
    def nbytes(self):
        return self.encoded.nbytes + self.raw.nbytes + self.lookup.nbytes


class ZeusCompressedObject:
    def __init__(self,
                 data = None,
                 clevel = 5,
                 compression_type = "zstd",
                 shuffle=blosc.BITSHUFFLE
                ):
        
        self.clevel = clevel
        self.compress_type = compression_type
        self.shuffle = shuffle
        
        self.compressed = None
        
        self.dtype = None
        self.shape = None
        
        if data is not None:
            self.compress(data)
        
    def compress(self,data):
        self.dtype = data.dtype
        self.shape = data.shape 
        
        self.compressed = blosc.compress(
            data.tobytes(),
            typesize=self.dtype.itemsize,
            clevel=self.clevel,
            shuffle=self.shuffle,
            cname=self.compress_type,
        )
    
        
    def decompress(self):
        
        
        return np.frombuffer(
            blosc.decompress(self.compressed),
            dtype=self.dtype
        ).reshape(self.shape)
        
    def __array__(self,dtype=None, copy=None):
        return self.decompress()
    
    @property
    def nbytes(self):
        return len(self.compressed)





def compress_int(x: np.array,
                unsigned:bool = False):
    """
    The purpose of this function is to estimate the best dtype to reduce the precision
    keeping the data lossless.
    """
    
    u_dtypes = [np.uint8,
        np.uint16,
        np.uint32,
        np.uint64
        ]

    dtypes = [
        np.int8,
        np.int16,
        np.int32,
        np.int64
        ]

    assert np.issubdtype(x.dtype,np.integer), "Only interger compression is supported"
    
    if len(x) == 0:
        return x   ## When recieving a completly empty sample
    
    if unsigned:
        max_value = x.max()
        
        for type in u_dtypes:
            if max_value < np.iinfo(type).max:
                return np.astype(x, type)
    
    else:
        max_value = x.max()
        min_value = x.min()
        
        for type in dtypes:
            if max_value < np.iinfo(type).max and min_value > np.iinfo(type).min:
                return np.astype(x,type)
    warnings.warn("[zeus-core] No suitable datatypes found for compression, returning original array")
    return x


def compress_float(x: np.array):
    
    """
    Similar to the compress_int method but for floats
    """
    
    if np.issubdtype(x.dtype,np.integer):
        return compress_int(x)
    
    
    if not np.issubdtype(x.dtype,np.floating):
        return x
    
    dtypes = [np.float16,
                np.float32,
                np.float64,
                np.float128]
    
    for type in dtypes:
        if np.array_equal(x, np.astype(x,type)):
            return np.astype(x,type)
    
    warnings.warn("[zeus-core] No suitable datatypes found for compression, returning original array")
    return x
