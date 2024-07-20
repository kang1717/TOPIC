class DummyMPI(object):
    def __init__(self):
        self.MPI  = None
        self.comm = None
        self.rank = 0
        self.size = 1

    def barrier(self):
        pass

    def disconnet(self):
        pass

    def free(self):
        pass

    def gather(self, data, root=0):
        return [data]

    def allreduce_max(self, data):
        return data

    def bcast(self, data, root=0):
        return data

class MPI4PY(object):
    def __init__(self):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def barrier(self):
        self.comm.barrier()

    def disconnect(self):
        self.comm.Disconnect()
    def free(self):
        self.comma.Free()

    def gather(self, data, root=0):
        return self.comm.gather(data, root=0)

    def allreduce_max(self, data):
        return self.comm.allreduce(data, op=self.MPI.MAX)

    def bcast(self, data, root=0):
        return self.comm.bcast(data, root=0)
