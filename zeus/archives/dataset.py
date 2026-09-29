from collections.abc import Sequence
from zeus.core.dataset import ZeusArray, ZeusShard
from zeus.core.compression import ZeusEncodedObject
import numpy as np

import blosc

class ArchiveDataset:
    
    def __init__(self,data_shard=None):
        
        #######
        # flatened points and their offsets
        self.data_points = data_shard.data_block.data
        
        self.data_offsets = data_shard.data_block.offsets
        self.data_offsets_dtype = self.data_offsets.dtype
        
        #flatened coordinates of the sparse array
        self.coordinates = data_shard.coordinate_block.data
        self.coordinates_dtype = self.coordinates.dtype
        
        self.coordinate_offsets = data_shard.coordinate_block.offsets
        self.coordinate_offsets_dtype = self.coordinate_offsets.dtype
        
        # Shapes
        self.shapes = data_shard.shape_block.data
        self.shape_dtype = self.shapes.dtype
        
        self.shapes_offsets = data_shard.shape_block.offsets
        self.shapes_offsets_dtype = self.shapes_offsets.dtype
        #######
        self.is_compressed = False
        
        self.RunCompressionSequence()
    
    
    def RunCompressionSequence(self):
        
        self.is_compressed = True
        ## using a lookup map to compress similar samples
        self.data_points = ZeusEncodedObject(data=self.data_points)
        
        self.data_points_raw_dtype = self.data_points.raw.dtype
        self.data_points_encoded_dtype = self.data_points.encoded.dtype
        
        ## The won't fit in the compression window hence sending to iterative module
        self.data_points.raw = ArchiveDataset.__iterative_compressor(self.data_points.raw)
        self.data_points.encoded = ArchiveDataset.__iterative_compressor(self.data_points.encoded)
        
        self.data_offsets = ArchiveDataset.__compression_utility(self.data_offsets)
        self.coordinates = ArchiveDataset.__compression_utility(self.coordinates)
        self.coordinate_offsets = ArchiveDataset.__compression_utility(self.coordinate_offsets)
        self.shapes = ArchiveDataset.__compression_utility(self.shapes)
        self.shapes_offsets = ArchiveDataset.__compression_utility(self.shapes_offsets)

    def RunDecompressionSequence(self):
        self.is_compressed = False
        
        self.data_offsets = ArchiveDataset.__decompression_utility(self.data_offsets,
                                                                   self.data_offsets_dtype)
        self.coordinates = ArchiveDataset.__decompression_utility(self.coordinates,
                                                                  self.coordinates_dtype)
        self.coordinate_offsets = ArchiveDataset.__decompression_utility(self.coordinate_offsets,
                                                                         self.coordinate_offsets_dtype)
        self.shapes = ArchiveDataset.__decompression_utility(self.shapes,
                                                             self.shape_dtype)
        self.shapes_offsets = ArchiveDataset.__decompression_utility(self.shapes_offsets,
                                                                     self.shapes_offsets_dtype)
        
        self.data_points.raw = ArchiveDataset.__iterative_decompressor(self.data_points.raw,
                                                                       self.data_points_raw_dtype)
        self.data_points.encoded = ArchiveDataset.__iterative_decompressor(self.data_points.encoded,
                                                                           self.data_points_encoded_dtype)
        
        self.data_points = np.array(self.data_points)
        
    
    @staticmethod
    def CreateArchiveDatasetZeusArray(z_list: Sequence[ZeusArray]):
        
        z_data = ZeusShard.CreateZeusShardFromList(z_list)
        
        return ArchiveDataset(z_data)

    @staticmethod
    def __compression_utility(data):
        return blosc.compress(data.tobytes(),
                              typesize=data.dtype.itemsize,
                              clevel=9,
                              shuffle=blosc.SHUFFLE
                              ,cname="zstd")
    
    @staticmethod
    def __decompression_utility(data,data_dtype):
        return np.frombuffer(
            blosc.decompress(data),
            dtype=data_dtype
        )
    
    @staticmethod
    def __iterative_compressor(data):
        ITERATION_LENGTH = 1_000_000_000
        
        byte_stream = data.tobytes()
        compressed_chunks = []
        
        for start in range(0,len(byte_stream),ITERATION_LENGTH):
            chunk = byte_stream[start:min(len(byte_stream),(start+ITERATION_LENGTH))]
            
            compressed_chunks.append(
                blosc.compress(chunk,
                typesize=data.dtype.itemsize,
                clevel=9,
                shuffle=blosc.SHUFFLE
                ,cname="zstd")
            )
            
        return compressed_chunks
        
    @staticmethod
    def __iterative_decompressor(compressed_chunks,data_dtype):
        
        decompressed_array = []
        
        for chunks in compressed_chunks:
            decompressed_array.append(ArchiveDataset.__decompression_utility(chunks,data_dtype))
            
        return np.concatenate(decompressed_array,dtype=data_dtype)
    
    @property
    def nbytes(self):
        if self.is_compressed:
            b = 0
            b += len(self.data_offsets) + len(self.coordinates) + len(self.coordinate_offsets)
            b += len(self.shapes) + len(self.shapes_offsets)
            
            for c in self.data_points.raw:
                b += len(c)
            
            for c in self.data_points.encoded:
                b += len(c)
            
            return b
        
        else:
            b = 0
            b += self.data_offsets.nbytes + self.coordinates.nbytes + self.coordinate_offsets.nbytes
            b += self.shapes.nbytes + self.shapes_offsets.nbytes + self.data_points.nbytes
                        
            return b