import struct
import os
import warnings

from .helper import *
from .magic import ZEUS_SHARD_MAGIC
from .magic import ZEUS_SHARD_HEADER

from .magic import ZEUS_ARCHIVE_MAGIC
from .magic import ZEUS_ARCHIVE_HEADER

from zeus.archives import ArchiveDataset


def writeshard(shard_data,path):

    nsamples = shard_data.data_block.offsets.shape[0]

    header = ZEUS_SHARD_HEADER.pack(
        ZEUS_SHARD_MAGIC,
        nsamples,

        nbytes(shard_data.data_block.data),
        nbytes(shard_data.data_block.offsets),
        encode_bool(shard_data.data_block.is_compressed),
        encode_dtype(shard_data.data_block.dtype),
        

        nbytes(shard_data.coordinate_block.data),
        nbytes(shard_data.coordinate_block.offsets),
        encode_bool(shard_data.coordinate_block.is_compressed),
        encode_dtype(shard_data.coordinate_block.dtype),
        
        nbytes(shard_data.shape_block.data),
        nbytes(shard_data.shape_block.offsets),
        encode_bool(shard_data.shape_block.is_compressed),
        encode_dtype(shard_data.shape_block.dtype),
        
    )

    if os.path.exists(path): os.remove(path)


    with open(path, "wb") as f:

        f.write(header)

        f.write(shard_data.data_block.data)
        f.write(shard_data.data_block.offsets)

        f.write(shard_data.coordinate_block.data)
        f.write(shard_data.coordinate_block.offsets)

        f.write(shard_data.shape_block.data)
        f.write(shard_data.shape_block.offsets)
        
        
        
        
def writearchive(archive_data,path):
    
    if archive_data.is_compressed == False:
        warnings.warn(
            "[zeus-io] Uncompressed archive data found.\n"
            "Compressing it before writing.\n"
            "This might take some time.")
        archive_data.RunCompressionSequence()

    header = ZEUS_ARCHIVE_HEADER.pack(
        ZEUS_ARCHIVE_MAGIC,

        encode_dtype(archive_data.data_points_raw_dtype),
        encode_dtype(archive_data.data_points_encoded_dtype),
        encode_dtype(archive_data.data_points.lookup.dtype),
        encode_dtype(archive_data.data_offsets_dtype),
        
        encode_dtype(archive_data.coordinates_dtype),
        encode_dtype(archive_data.coordinate_offsets_dtype),
        
        encode_dtype(archive_data.shape_dtype),
        encode_dtype(archive_data.shapes_offsets_dtype),

        nbytes(archive_data.data_points.raw),
        nbytes(archive_data.data_points.encoded),
        nbytes(archive_data.data_points.lookup),
        nbytes(archive_data.data_offsets),
        
        nbytes(archive_data.coordinates),
        nbytes(archive_data.coordinate_offsets),
        nbytes(archive_data.shapes),
        nbytes(archive_data.shapes_offsets),
        
    )

    if os.path.exists(path): os.remove(path)


    with open(path, "wb") as f:

        f.write(header)
        
        f.write(archive_data.data_points.raw)
        f.write(archive_data.data_points.encoded)
        f.write(archive_data.data_points.lookup)
        f.write(archive_data.data_offsets)
        
        f.write(archive_data.coordinates)
        f.write(archive_data.coordinate_offsets)
        
        f.write(archive_data.shapes)
        f.write(archive_data.shapes_offsets)