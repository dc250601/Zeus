import enum

class Layout(enum.Enum):
    Sparse = "sparse"
    Dense = "dense"

class Backend(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPI = "mpi"
    