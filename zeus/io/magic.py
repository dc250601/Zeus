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