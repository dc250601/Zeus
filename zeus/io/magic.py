import struct

#################################################################
# 10s : magic
# Q   : number of samples

# Q   : data payload size
# Q   : data offsets size
# B   : data compression_status
# 16s : data dtype

# Q   : coordinate payload size
# Q   : coordinate offsets size
# B   : coordinate compression_status
# 16s : compression dtype
    
# Q   : shape payload size
# Q   : shape offsets size
# B   : shape compression_status
# 16s : shape dtype

ZEUS_SHARD_MAGIC = b"ZEUS_SHARD"
ZEUS_SHARD_HEADER = struct.Struct("<10sQQQB16sQQB16sQQB16s")
#################################################################



#################################################################
# 10s : magic
#
# 16s : data_points.raw dtype
# 16s : data_points.encoded dtype
# 16s : data_points.lookup dtype
# 16s : data_offsets dtype
#
# 16s : coordinates dtype
# 16s : coordinate_offsets dtype
#
# 16s : shapes dtype
# 16s : shape_offsets dtype
#
# Q   : data_points.raw size
# Q   : data_points.encoded size
# Q   : data_points.lookup size
# Q   : data_offsets size
#
# Q   : coordinates size
# Q   : coordinate_offsets size
#
# Q   : shapes size
# Q   : shape_offsets size

ZEUS_ARCHIVE_MAGIC = b"ZEUS_ARCHV"

ZEUS_ARCHIVE_HEADER = struct.Struct(
    "<10s"
    "16s16s16s16s"
    "16s16s"
    "16s16s"
    "QQQQ"
    "QQ"
    "QQ"
)
#################################################################