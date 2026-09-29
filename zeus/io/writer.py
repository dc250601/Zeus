import struct
import os
from .helper import *
from .magic import ZEUS_SHARD_MAGIC as MAGIC
from .magic import ZEUS_SHARD_HEADER as HEADER

def writeshard(shard_data,path):

    nsamples = shard_data.data_block.offsets.shape[0]

    header = HEADER.pack(
        MAGIC,
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