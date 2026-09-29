import numpy as np
from .enums import Layout
import warnings
import blosc
from .compression import ZeusCompressedObject, compress_int, compress_float
from .type_conversion import _dense_to_sparse, _sparse_to_dense
# from ._dense_to_sparse

class ZeusArray:
    """The main class for the Zeus array. This class is used for storage of sparse data"""
    
    def __init__(
        self,
        data: list|np.ndarray = None,
        coordinates: list|np.ndarray = None,
        type: Layout = Layout.Sparse,
        shape = None
        ):
        

        self.data = compress_float(data)
        self.coordinates = compress_int(coordinates, unsigned=True)        
        self.type = type
        
        if type == Layout.Sparse:
            if shape is None:
                raise ValueError("For Sparse array shape cannot be none")
            self.shape = shape
        else:
            self.shape = data.shape
    
    # def numpy(self):
        
    #     assert self.type == Layout.Dense, "Can only return numpy format for "
    
    @property
    def nbytes(self) -> int:
        bytes = 0
        if self.type == Layout.Sparse:
            bytes = self.data.nbytes + self.coordinates.nbytes
        else:
            bytes = self.data.nbytes
        return bytes
    
    
    @staticmethod
    def compress_array(zeus_array,
                       clevel = 5,
                       shuffle = blosc.BITSHUFFLE,
                       cname = "zstd"
                       ):
        
        zeus_array.data = ZeusCompressedObject(data = zeus_array.data,
                                               clevel = clevel,
                                               compression_type = cname,
                                               shuffle=shuffle) 
        
        zeus_array.coordinates = ZeusCompressedObject(data = zeus_array.coordinates,
                                                      clevel = clevel,
                                                      compression_type = cname,
                                                      shuffle=shuffle)
        
        
        return zeus_array
    
    @staticmethod
    def decompress_array(zeus_array):
        zeus_array.data = np.array(zeus_array.data)
        zeus_array.coordinates = np.array(zeus_array.coordinates)
        return zeus_array
    
    def __array__(self, dtype=None, copy=None):
        return _sparse_to_dense(ZeusArray(
            data =  np.array(self.data,
                             dtype=dtype,
                             copy=copy),
            coordinates= np.array(self.coordinates,
                                  dtype=dtype,
                                  copy=copy),
            type = self.type,
            shape = self.shape
        ))
        
    @staticmethod
    def create_zeus_array(data):
       
       sparse_array,non_zero_coord,array_shape = _dense_to_sparse(data) 
       
       return ZeusArray(
           data=sparse_array,
           coordinates=non_zero_coord,
           shape=array_shape,
           type=Layout.Sparse
           )
       
    