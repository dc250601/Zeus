import numpy as np
from .enums import Backend, Layout 
from .array import ZeusArray
from .index import index

def dense_to_sparse(dense_array: np.ndarray | ZeusArray,
                    backend: Backend = Backend.CPU
                    ) -> ZeusArray:
    
    if isinstance(dense_array, ZeusArray):
        assert dense_array.type == Layout.Dense, "Input ZeusArray must be of type Dense"
        data = dense_array.data
        
    assert backend == Backend.CPU, "Currently only CPU backend is supported"
    
 
 
def _single_element_dense_to_sparse(dense_array: np.ndarray) -> ZeusArray:
    
    """
    Convert a single element from dense layout to sparse layout
    """
    
    array_shape = dense_array.shape
    dense_array = dense_array.flatten()
    non_zero_coord = np.where(dense_array != 0)[0]
    sparse_array = dense_array[non_zero_coord]
    
    return ZeusArray(
        data = sparse_array,
        coordinates=non_zero_coord,
        shape=array_shape,
        type=Layout.Dense       
    )
    
def _single_element_sparse_to_dense(zeus_array: ZeusArray) -> ZeusArray:
    
    """
    Convert a single element from sparse layout to dense layout
    """
    assert zeus_array.type == Layout.Sparse, "Cannot convert non sparse layout to dense"
    dense_array = np.zeros(zeus_array.shape)
    dense_array[zeus_array.coordinates] = zeus_array.data
    
    return ZeusArray(
        data = dense_array,
        type=Layout.Dense
    )
        
        
         
    
    
    
        
    
    
