.SUFFIXES: .f90 .f .c .cpp .o

FC=mpiifort
F90=mpiifort
CC=mpiicc
CXX=mpiicpc
FFLAGS=-openmp -g -O3
F90FLAGS=-openmp -g -O3
CFLAGS=-openmp -g -O3
CCFLAGS=-openmp -g -O3
CXXFLAGS=-openmp -g -O3
lib=~/program/SU2_real_mpi


main_omp: main_omp.o tensor.o  tensor_dmrg_src.o tensor_su2.o tensor_su2_dmrg_src.o tensor_su2_hamiltonian.o su2bond.o su2struct.o ran.o sort2.o dmrg_su2.o dmrg_su2_omp.o lanczos_su2.o 
	${F90} ${FFLAGS} -o $@ $< tensor.o  tensor_dmrg_src.o tensor_su2.o tensor_su2_dmrg_src.o tensor_su2_hamiltonian.o su2bond.o su2struct.o ran.o sort2.o dmrg_su2.o dmrg_su2_omp.o lanczos_su2.o -L/home/intel/impi/5.0.1.035/intel64/lib/  -lmpicxx -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lstdc++ -nofor_main 

main_mea_enr: main_mea_enr.o tensor.o  tensor_dmrg_src.o tensor_su2.o tensor_su2_dmrg_src.o tensor_su2_hamiltonian.o su2bond.o su2struct.o ran.o sort2.o dmrg_su2.o dmrg_su2_omp_mea_enr.o lanczos_su2.o 
	${F90} ${FFLAGS} -o $@ $< tensor.o  tensor_dmrg_src.o tensor_su2.o tensor_su2_dmrg_src.o tensor_su2_hamiltonian.o su2bond.o su2struct.o ran.o sort2.o dmrg_su2.o dmrg_su2_omp_mea_enr.o lanczos_su2.o -L/home/intel/impi/5.0.1.035/intel64/lib/  -lmpicxx -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lstdc++ -nofor_main 


clean:
	/bin/rm -f *.o 

.f.o:
	${FC} ${F77FLAGS} -c $<  ${FCINCLUDE}
.f90.o:
	${F90} ${F90FLAGS} -c $< ${FCINCLUDE}
.c.o:
	${CC} ${CCFLAGS} -c $< ${CXXINCLUDE}
.cpp.o:
	${CXX} ${CXXFLAGS} -c $< ${CXXINCLUDE}
