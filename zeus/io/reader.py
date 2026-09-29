import struct
from .helper import *
from zeus.core.dataset import ContiguousBlocks, ZeusShard
from .magic import ZEUS_SHARD_HEADER as HEADER

def readshard(path):

    validate_zeus_shard_file(path=path)

    with open(path, "rb") as f:
        header_bytes = f.read(HEADER.size)

        (
            magic,
            nsamples,

            data_size,
            data_offsets_size,
            data_is_compressed,
            data_dtype,

            coordinate_size,
            coordinate_offsets_size,
            coordinate_is_compressed,
            coordinate_dtype,

            shape_size,
            shape_offsets_size,
            shape_is_compressed,
            shape_dtype

        ) = HEADER.unpack(header_bytes)
        
        
        data_dtype = decode_dtype(data_dtype)
        coordinate_dtype = decode_dtype(coordinate_dtype)
        shape_dtype = decode_dtype(shape_dtype)
        
        data_is_compressed = decode_bool(data_is_compressed)
        coordinate_is_compressed = decode_bool(coordinate_is_compressed)
        shape_is_compressed = decode_bool(shape_is_compressed)
        
        
        ############################################################
        data = f.read(data_size)

        data_offsets = np.frombuffer(
            f.read(data_offsets_size),
            dtype="<u8"
        ).copy()    


        data_block = ContiguousBlocks(
            data=data,
            offsets=data_offsets,
            is_compressed=data_is_compressed,
            dtype=data_dtype,
        )
        ############################################################
        
        coordinates = f.read(coordinate_size)

        coordinate_offsets = np.frombuffer(
            f.read(coordinate_offsets_size),
            dtype="<u8"
        ).copy()
        
        coordinate_block = ContiguousBlocks(
            data=coordinates,
            offsets=coordinate_offsets,
            is_compressed=coordinate_is_compressed,
            dtype=coordinate_dtype,
        )
        ############################################################
        shape = f.read(shape_size)

        shape_offsets = np.frombuffer(
            f.read(shape_offsets_size),
            dtype="<u8"
            ).copy()
        
        shape_block = ContiguousBlocks(
            data=shape,
            offsets=shape_offsets,
            is_compressed=shape_is_compressed,
            dtype=shape_dtype,
        )
        ############################################################
        
        shard = ZeusShard(
            data_block = data_block,
            coordinate_block=coordinate_block,
            shape_block=shape_block,
            is_compressed=data_is_compressed
            
        )
    return shard