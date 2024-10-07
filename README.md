# SU(2)DMRG
an SU(2) spin rotational symmetric DMRG/MPS algorithm in C++

# Makefile
make main_omp

# Run
mpirun -n 1 ./main_omp -lx 16 -ly 8 -jcoup 0.5 -mkl 2 -psize 4 -sec 0 -kx 0 -ky 0 -qcoup 1 -d 2000 -dr 0 -dge 0 -exci 0 -niter 5 -boundary 2 -memory_flag 0

- lx:
length in open x direction
- ly:
length in periodic y direction
- jcoup:
coupling ratio J2/J1
- mkl:
number of threads used for parallel computing in math kernel library
- psize:
number of parallel threads
- sec:
total spin quantum sector
- kx:
momentum in x direction
- ky:
momentum in y direction
- qcoup:
coupling constant ratio Q/J1
- d:
maximum bond dimension allowed
- dr:
bond dimension of read-in mps
- dge:
bond dimension of the ground state mps if calculating excitations
- exc:
target excited states
- niter:
number of iterations to go
- boundary:
boundary condition code
- memory_flag:
use hard disk for memory swap or not

