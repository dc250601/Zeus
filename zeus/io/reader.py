import struct
from .helper import *
from zeus.core.dataset import ContiguousBlocks, ZeusShard
from zeus.archives import ArchiveDataset
from zeus.core.compression import ZeusEncodedObject

from .magic import ZEUS_SHARD_HEADER
from .magic import ZEUS_ARCHIVE_HEADER

def readshard(path):

    validate_zeus_shard_file(path=path)

    with open(path, "rb") as f:
        header_bytes = f.read(ZEUS_SHARD_HEADER.size)

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
        
        
        data_dtype = decode_dtype(data_dtype)
        coordinate_dtype = decode_dtype(coordinate_dtype)
        shape_dtype = decode_dtype(shape_dtype)
        
        data_is_compressed = decode_bool(data_is_compressed)
        coordinate_is_compressed = decode_bool(coordinate_is_compressed)
        shape_is_compressed = decode_bool(shape_is_compressed)
        
        
        ############################################################
        
        data = read_data_chunk(data = f.read(data_size),
                               is_compressed=data_is_compressed,
                               dtype=data_dtype
                               )

        data_offsets = read_data_chunk(data = f.read(data_offsets_size),
                                       is_compressed=False,
                                       dtype="<u8"
                                       )
        


        data_block = ContiguousBlocks(
            data=data,
            offsets=data_offsets,
            is_compressed=data_is_compressed,
            dtype=data_dtype,
        )
        ############################################################
        
        coordinates = read_data_chunk(data = f.read(coordinate_size),
                                      is_compressed=coordinate_is_compressed,
                                      dtype = coordinate_dtype
                                      )
        
        
        coordinate_offsets = read_data_chunk(data = f.read(coordinate_offsets_size),
                                       is_compressed=False,
                                       dtype="<u8"
                                       )
        
        coordinate_block = ContiguousBlocks(
            data=coordinates,
            offsets=coordinate_offsets,
            is_compressed=coordinate_is_compressed,
            dtype=coordinate_dtype,
        )
        ############################################################
        
        shape = read_data_chunk(
            data = f.read(shape_size),
            is_compressed=shape_is_compressed,
            dtype=shape_dtype
        )
        
        shape_offsets = read_data_chunk(
            data = f.read(shape_offsets_size),
            is_compressed=False,
            dtype="<u8"
        )
        
        
        
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

def readArchive(path):
    
    validate_zeus_archive_file(path=path)
    
    with open(path, "rb") as f:
        header_bytes = f.read(ZEUS_ARCHIVE_HEADER.size)

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

        data_points_raw_dtype = decode_dtype(data_points_raw_dtype)
        data_points_encoded_dtype = decode_dtype(data_points_encoded_dtype)
        data_points_lookup_dtype = decode_dtype(data_points_lookup_dtype)
        data_offsets_dtype = decode_dtype(data_offsets_dtype)

        coordinates_dtype = decode_dtype(coordinates_dtype)
        coordinate_offsets_dtype = decode_dtype(coordinate_offsets_dtype)

        shape_dtype = decode_dtype(shape_dtype)
        shapes_offsets_dtype = decode_dtype(shapes_offsets_dtype)


        ############################################################

        data_points_raw = f.read(data_points_raw_size)

        data_points_raw_offsets = np.frombuffer(
            f.read(data_points_raw_offsets_size),
            dtype="<u8"
        )

        data_points_encoded = f.read(data_points_encoded_size)

        data_points_encoded_offsets = np.frombuffer(
            f.read(data_points_encoded_offsets_size),
            dtype="<u8"
        )

        data_points_lookup = np.frombuffer(
            f.read(data_points_lookup_size),
            dtype=data_points_lookup_dtype
        )


        data_points = ZeusEncodedObject.__new__(ZeusEncodedObject)

        data_points.raw = data_points_raw
        data_points.encoded = data_points_encoded
        data_points.lookup = data_points_lookup
        
        ############################################################

        data_offsets = f.read(data_offsets_size)

        ############################################################

        coordinates = f.read(coordinates_size)

        coordinate_offsets = f.read(coordinate_offsets_size)

        ############################################################

        shapes = f.read(shapes_size)

        shapes_offsets = f.read(shapes_offsets_size)

        ############################################################

        archive = ArchiveDataset.__new__(ArchiveDataset)

        archive.data_points = data_points

        archive.data_points_raw_dtype = data_points_raw_dtype
        archive.data_points_encoded_dtype = data_points_encoded_dtype

        archive.data_points_raw_offsets = data_points_raw_offsets
        archive.data_points_encoded_offsets = data_points_encoded_offsets


        archive.data_offsets = data_offsets
        archive.data_offsets_dtype = data_offsets_dtype


        archive.coordinates = coordinates
        archive.coordinates_dtype = coordinates_dtype

        archive.coordinate_offsets = coordinate_offsets
        archive.coordinate_offsets_dtype = coordinate_offsets_dtype


        archive.shapes = shapes
        archive.shape_dtype = shape_dtype

        archive.shapes_offsets = shapes_offsets
        archive.shapes_offsets_dtype = shapes_offsets_dtype


        archive.is_compressed = True

    return archive