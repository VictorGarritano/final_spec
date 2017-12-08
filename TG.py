from numpy import *
from numpy.fft import fftfreq,fft,ifft,irfft2,rfft2
from mpi4py import MPI

nu=0.000625
T=0.1
dt=0.01
N=2**7
comm=MPI.COMM_WORLD
num_processes=comm.Get_size()
rank=comm.Get_rank()
Np=N/num_processes
X=mgrid[rank*Np:(rank+1)*Np,:N,:N].astype(float)*2*pi/N
