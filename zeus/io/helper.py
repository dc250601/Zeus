import numpy as np
from .magic import ZEUS_SHARD_MAGIC
from .magic import ZEUS_SHARD_HEADER

from .magic import ZEUS_ARCHIVE_HEADER
from .magic import ZEUS_ARCHIVE_MAGIC

def nbytes(x):
    return x.nbytes if hasattr(x, "nbytes") else len(x)

def encode_bool(value):

    if value is None:
        return 255

    return 1 if value else 0


def decode_bool(value):

    if value == 255:
        return None

    if value == 0:
        return False

    if value == 1:
        return True

    raise ValueError(f"Invalid encoded boolean value: {value}")


def encode_dtype(dtype):


    if dtype is None:
        return b""

    return np.dtype(dtype).str.encode("ascii")


def decode_dtype(value):
    
    value = value.rstrip(b"\x00")

    if value == b"":
        return None

    return np.dtype(value.decode("ascii"))


def validate_zeus_shard_file(path):
    
    with open(path, "rb") as f:
        header_bytes = f.read(ZEUS_SHARD_HEADER.size)
        
        if len(header_bytes) != ZEUS_SHARD_HEADER.size:
            raise ValueError("Invalid or truncated Zeus Shard file")
        
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

        ) = ZEUS_SHARD_HEADER.unpack(header_bytes)

        
        if magic != ZEUS_SHARD_MAGIC:
            raise ValueError(
                f"Invalid Zeus shard magic: {magic}"
            )
            

def validate_zeus_archive_file(path):
    
    with open(path, "rb") as f:
        header_bytes = f.read(ZEUS_ARCHIVE_HEADER.size)
        
        if len(header_bytes) != ZEUS_ARCHIVE_HEADER.size:
            raise ValueError("Invalid or truncated Zeus Archive file")
        
        (
            magic,
            
            data_points_raw_dtype,
            data_points_encoded_dtype,
            data_points_lookup_dtype,
            data_offsets_dtype,

            coordinates_dtype,
            coordinate_offsets_dtype,

            shape_dtype,
            shapes_offsets_dtype,

            data_points_raw_size,
            data_points_raw_offsets_size,
            data_points_encoded_size,
            data_points_encoded_offsets_size,
            data_points_lookup_size,
            data_offsets_size,

            coordinates_size,
            coordinate_offsets_size,

            shapes_size,
            shapes_offsets_size

        ) = ZEUS_ARCHIVE_HEADER.unpack(header_bytes)

        
        if magic != ZEUS_ARCHIVE_MAGIC:
            raise ValueError(
                f"Invalid Zeus Archive magic: {magic}"
            )
            


def read_data_chunk(data, is_compressed,dtype=None):
    
    is_compressed is not None, "Cannot accept None values for compression flag" 
    if is_compressed:
        return data
    else:
        return np.frombuffer(data,dtype=dtype).copy()
    
    