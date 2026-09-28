import numpy as np
from .array import ZeusArray
from collections.abc import Sequence
from .compression import ZeusCompressedObject
import warnings
import blosc
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
        
        self.create_zeus_dataset(element_list)
        self.is_compressed = False
    
       
    def create_zeus_dataset(self,element_list):
        
        data_list = []
        coordinate_list = []
        shape_list = []
         
        for elem in element_list:
            data_list.append(elem.data)
            coordinate_list.append(elem.coordinates)
            shape_list.append(elem.shape)
        
        self.data_block = ContiguousBlocks.create_block_from_list(data_list)
        self.coordinate_block = ContiguousBlocks.create_block_from_list(coordinate_list)
        self.shape_block = ContiguousBlocks.create_block_from_list(shape_list)
        
    @property
    def nbytes(self):
        return self.data_block.nbytes + self.coordinate_block.nbytes + self.shape_block.nbytes 

    @staticmethod
    def compress_dataset(
        dataset,
        clevel = 5,
        shuffle = blosc.BITSHUFFLE,
        compression_type = "zstd"):
        
        dataset.data_block = ContiguousBlocks.compress_contiguous_array(
            block = dataset.data_block,
            clevel = clevel,
            compression_type = compression_type,
            shuffle=shuffle
        )
        
        dataset.coordinate_block = ContiguousBlocks.compress_contiguous_array(
            block = dataset.coordinate_block,
            clevel = clevel,
            compression_type = compression_type,
            shuffle=shuffle
        )
        
        dataset.shape_bloc = ContiguousBlocks.compress_contiguous_array(
            block = dataset.shape_block,
            clevel = clevel,
            compression_type = compression_type,
            shuffle=shuffle
            )
        dataset.is_compressed = True
        return dataset
    
    
    @staticmethod
    def decompress_dataset(
        dataset):
        
        dataset.data_block = ContiguousBlocks.decompress_contiguous_array(
            block = dataset.data_block)
        
        dataset.coordinate_block = ContiguousBlocks.decompress_contiguous_array(
            block = dataset.coordinate_block
        )
        
        dataset.shape_block = ContiguousBlocks.decompress_contiguous_array(
            block = dataset.shape_block
            )
        dataset.is_compressed = False
        return dataset
        

    def __len__(self):
        return(self.data_block.offsets.shape[0])
    
    def __getitem__(self, key):
        
        if self.is_compressed:
            warnings.warn("[zeus-core] Cannot iterate over compressed state, decompressing")
            _ = ZeusDataset.decompress_dataset(self)
            
        if key == 0:
            data = self.data_block.data[0:self.data_block.offsets[key]]
            coordinate = self.coordinate_block.data[0:self.coordinate_block.offsets[key]]
            shape = self.shape_block.data[0:self.shape_block.offsets[key]]
        
        else:
            data = self.data_block.data[self.data_block.offsets[key-1]:self.data_block.offsets[key]]
            coordinate = self.coordinate_block.data[self.coordinate_block.offsets[key-1]:self.coordinate_block.offsets[key]]
            shape = self.shape_block.data[self.shape_block.offsets[key-1]:self.shape_block.offsets[key]]
        
        return ZeusArray(data=data,
                         coordinates=coordinate,
                         shape=shape
                         )

class ContiguousBlocks:
    
    def __init__(self,data,offsets):
        self.data = data
        self.offsets = offsets
        self.is_compressed = None
        self.dtype = None
    @staticmethod
    def create_block_from_list(data_list):
        data, offsets = ContiguousBlocks.create_contiguous_array(data_list)
        
        return ContiguousBlocks(data=data,
                                offsets=offsets
                                )
    
    @staticmethod
    def create_contiguous_array(data: Sequence[np.array]):
        offsets = np.cumulative_sum(np.array(list([len(x) for x in data])))
        cont = np.concatenate(data).flatten()
        return cont, offsets
    
    @staticmethod
    def compress_contiguous_array(block,
                                  clevel = 5,
                                  shuffle = blosc.BITSHUFFLE,
                                  compression_type = "zstd"):
        
       
        ## To be made mutithreaded in the future
        
        compressed_seq = []
        
        last_offset = 0
        last_compress_offset = 0
        
        for i in range(block.offsets.shape[0]):
            data = block.data[last_offset:block.offsets[i]]
            
            
            compresed_unit = blosc.compress(
                        data.tobytes(),
                        typesize=data.dtype.itemsize,
                        clevel=clevel,
                        shuffle=shuffle,
                        cname=compression_type,
                    )
            
            
            compressed_seq.append(compresed_unit)
            
            last_offset = block.offsets[i]
            last_compress_offset += len(compresed_unit)
            block.offsets[i] = last_compress_offset
        
        block.data = b"".join(compressed_seq) 
        block.is_compressed = True
        block.dtype = data.dtype
        
        return block
        
    @staticmethod
    def decompress_contiguous_array(block):
        if block.is_compressed == False:
            warnings.warn("[zeus-core] uncompressed block found returning the original")
            return block
        
        decompressed_seq = []
        last_offset = 0
        last_decompressed_offset = 0
        
        for i in range(block.offsets.shape[0]):
            data = np.frombuffer(blosc.decompress(block.data[last_offset:block.offsets[i]]),
                                 dtype=block.dtype
                                 )
            decompressed_seq.append(data)
            
            last_offset = block.offsets[i]
            last_decompressed_offset += len(data)
            block.offsets[i] = last_decompressed_offset
            
            
        block.data = np.concatenate(decompressed_seq)
        block.is_compressed = False
        
        return block
    @property
    def nbytes(self):
        return self.data.nbytes + self.offsets.nbytes    
    
    