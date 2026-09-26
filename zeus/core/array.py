import numpy as np
from .enums import Layout
from collections.abc import Sequence
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

        self.data = data
        self.coordinates = coordinates        
        self.type = type
        
        if type == Layout.Sparse:
            if shape is None:
                raise ValueError("For Sparse array shape cannot be none")
            self.shape = shape
        else:
            self.shape = data.shape
    
    # def numpy(self):
        
    #     assert self.type == Layout.Dense, "Can only return numpy format for "
    
    
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
    