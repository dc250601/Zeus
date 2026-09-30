from collections.abc import Sequence
from zeus.core.dataset import ZeusArray, ZeusShard
from zeus.core.compression import ZeusEncodedObject
import numpy as np

import blosc

class ArchiveDataset:
    
    def __init__(self):
        
        self.data_points = None
        self.data_offsets = None
        self.data_offsets_dtype = None
        self.coordinates = None
        self.coordinates_dtype = None
        self.coordinate_offsets = None
        self.coordinate_offsets_dtype = None
        self.shapes = None
        self.shape_dtype = None
        self.shapes_offsets = None
        self.shapes_offsets_dtype = None
        self.is_compressed = None
        
        self.data_points_raw_offsets = None ## Placeholders
        self.data_points_encoded_offsets = None ## Placeholders
    
    @staticmethod
    def CreateArchiveObjectFromShard(shard):
        
        archive_dataset = ArchiveDataset()
        
        archive_dataset.data_points = shard.data_block.data
        
        #######
        # flatened points and their offsets
        archive_dataset.data_points = shard.data_block.data
        

        
        archive_dataset.data_offsets = shard.data_block.offsets
        archive_dataset.data_offsets_dtype = archive_dataset.data_offsets.dtype
        
        #flatened coordinates of the sparse array
        archive_dataset.coordinates = shard.coordinate_block.data
        archive_dataset.coordinates_dtype = archive_dataset.coordinates.dtype
        
        archive_dataset.coordinate_offsets = shard.coordinate_block.offsets
        archive_dataset.coordinate_offsets_dtype = archive_dataset.coordinate_offsets.dtype
        
        # Shapes
        archive_dataset.shapes = shard.shape_block.data
        archive_dataset.shape_dtype = archive_dataset.shapes.dtype
        
        archive_dataset.shapes_offsets = shard.shape_block.offsets
        archive_dataset.shapes_offsets_dtype = archive_dataset.shapes_offsets.dtype
        #######
        archive_dataset.is_compressed = False
        
        archive_dataset.RunCompressionSequence()    
        
        return archive_dataset
    
    def RunCompressionSequence(self):
        
        self.is_compressed = True
        ## using a lookup map to compress similar samples
        self.data_points = ZeusEncodedObject(data=self.data_points)
        
        self.data_points_raw_dtype = self.data_points.raw.dtype
        self.data_points_encoded_dtype = self.data_points.encoded.dtype
        
        ## The won't fit in the compression window hence sending to iterative module
        self.data_points.raw, self.data_points_raw_offsets = ArchiveDataset.__iterative_compressor(self.data_points.raw)
        self.data_points.encoded, self.data_points_encoded_offsets = ArchiveDataset.__iterative_compressor(self.data_points.encoded)
        
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
                                                                       self.data_points_raw_offsets,
                                                                       self.data_points_raw_dtype)
        
        self.data_points.encoded = ArchiveDataset.__iterative_decompressor(self.data_points.encoded,
                                                                           self.data_points_encoded_offsets,
                                                                           self.data_points_encoded_dtype)
        
        self.data_points = np.array(self.data_points)
        
    
    @staticmethod
    def CreateArchiveDatasetZeusArray(z_list: Sequence[ZeusArray]):
        
        z_data = ZeusShard.CreateZeusShardFromList(z_list)
        
        return ArchiveDataset.CreateArchiveObjectFromShard(z_data)

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
        
        compressed_stream = bytearray()
        offsets = []     
        for start in range(0,len(byte_stream),ITERATION_LENGTH):
            chunk = byte_stream[start:min(len(byte_stream),(start+ITERATION_LENGTH))]
            
            compressed_chunks = blosc.compress(
                chunk,
                typesize=data.dtype.itemsize,
                clevel=9,
                shuffle=blosc.SHUFFLE
                ,cname="zstd")
            
            compressed_stream.extend(compressed_chunks)
            offsets.append(len(compressed_chunks))
        
        offsets = np.cumulative_sum(offsets, dtype=np.dtype("<u8"))
        return compressed_stream, offsets
        
    @staticmethod
    def __iterative_decompressor(compressed_stream,
                                 offsets,
                                 data_dtype):
        
        
        last_index = 0
        
        decompressed_stream = bytearray()
        
        for chunk_idx in range(len(offsets)):
            chunk = compressed_stream[last_index:offsets[chunk_idx]]
            last_index = offsets[chunk_idx]
            
            chunk = blosc.decompress(chunk)
            
            decompressed_stream.extend(chunk)
            
        return np.frombuffer(decompressed_stream,dtype=data_dtype)
    
    @property
    def nbytes(self):
        if self.is_compressed:
            b = 0
            b += len(self.data_offsets) + len(self.coordinates) + len(self.coordinate_offsets)
            b += len(self.shapes) + len(self.shapes_offsets)
            b += len(self.data_points.raw) + len(self.data_points.encoded)

            return b
        
        else:
            b = 0
            b += self.data_offsets.nbytes + self.coordinates.nbytes + self.coordinate_offsets.nbytes
            b += self.shapes.nbytes + self.shapes_offsets.nbytes + self.data_points.nbytes
                        
            return b