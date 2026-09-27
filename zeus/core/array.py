import numpy as np
from .enums import Layout
from collections.abc import Sequence
import warnings

# from ._dense_to_sparse

class ZeusArray:
    """The main class for the Zeus array. This class is used for storage of sparse data"""
    
    def __init__(
        self,
        data: list|np.ndarray = None,
        coordinates: list|np.ndarray = None,
        type: Layout = Layout.Sparse,
        shape = None,
        ):

        self.data = self.compress_float(data)
        self.coordinates = self.compress_int(coordinates, unsigned=True)        
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
        
        if unsigned:
            max_value = x.max()
            
            for type in u_dtypes:
                if max_value < np.iinfo(type).max:
                    return np.astype(x, type)
        
        else:
            max_value = x.max()
            min_value = x.min()
            
            for type in dtypes:
                if max_value < np.iinfo(type).max() and min_value > np.iinfo(type).min():
                    return np.astype(x,type)
        warnings.warn("[zeus-core] No suitable datatypes found for compression, returning original array")
        return x
    
    @staticmethod
    def compress_float(x: np.array):
        
        """
        Similar to the compress_int method but for floats
        """
        
        if np.issubdtype(x.dtype,np.integer):
            return self.compress_int(x)
        
        
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
        
        
        
        



class ZeusDataset:
    """
    This class holds a collection of  Zeus Arrays into a single continuous memory block.
    """
    
    def __init__(
        self,
        element_list: Sequence[ZeusArray]
    ):
        self.data_block = None
        self.coordinate_block = None
        self.shape_block = None
        self.offset_block = None
        
        self.create_ZeusDataset()
    