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
# print('rank {0}, Np {1}, num_processes {2}'.format(rank, Np, num_processes))
X=mgrid[rank*Np:(rank+1)*Np,:N,:N].astype(float)*2*pi/N
U = empty((3, Np, N, N))
U_hat = empty((3, N, Np, N/2 + 1), dtype=complex)
P = empty((Np, N, N))
P_hat = empty((N, Np, N/2 + 1), dtype=complex)
U_hat0 = empty((3, N, Np, N/2+1), dtype=complex)
U_hat1 = empty((3, N, Np, N/2+1), dtype=complex)
dU = empty((3, N, Np, N/2+1), dtype=complex)
Uc_hat = empty((N, Np, N/2+1), dtype=complex)
Uc_hatT = empty((Np, N, N/2+1), dtype=complex)
U_mpi = empty((num_processes, Np, Np, N/2+1), dtype=complex)
curl = empty((3, Np, N, N))
kx = fftfreq(N, 1./N)
kz = kx[:(N/2 + 1)].copy()
kz[-1] *= -1
K = array(meshgrid(kx, kx[rank*Np:(rank+1)*Np], ks, indexing='ij'), dtype=int)
K2 = sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
kmax_dealias = 2./3.*(N/2 + 1)
dealias = array((abs(K[0]) < kmax_dealias)*(abs(K[1]) < kmax_dealias)*\
(abs(K[2]) < kmax_dealias), dtype=bool)
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.0]
