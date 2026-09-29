import numpy as np
from .magic import ZEUS_SHARD_MAGIC
from .magic import ZEUS_SHARD_HEADER

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
            
