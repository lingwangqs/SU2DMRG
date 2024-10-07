#include "lanczos_su2.hpp"
using namespace std;

class dmrg_su2{
private:
  int ly,lx,bondd,ph,ns,read,read2,readinter,exci,max_exci,totspin,xleft,xrght,badposition,phdim,nbb,physpn,nevolution,nfree,ky;
  int **fll,**frr,**hmap,**plqpos,**plqflg,*qfll,*qfrr,**plqind;
  tensor_su2 *uu,*vv,*hh,*dltu,*comp,**opr,**plqopr,*plqplq,**ovlp,**orth,**tran,sigma[4],ring[8],qterm[12],spnswap,spnproj,*overlapvec;
  tensor **step2;
  double gs_enr[100],enr[100],*coup,err,**ww,*wtmp,*bond_enr,*bond_enr_all,wt,vecnorm;
  //this is to store the cgc coefficient

  void initialize_local_operators();
  void lanczos_solve_eigenvector_idmrg(int,int,tensor_su2&);
  void sweep_to_left();
  void sweep_to_right();
  void prepare_site_operator_from_left(int,int);
  void prepare_site_operator_from_right(int,int);
  void prepare_site_operator_from_left(int);
  void prepare_site_operator_from_right(int);
public:
  dmrg_su2(int,int,int,int,int,int);
inline  dmrg_su2(){};
  ~dmrg_su2();
  inline void set_ky(int kk){ky=kk;}
  double get_enr(){return gs_enr[exci];}
  void do_idmrg();
  void prepare_compression();
  void prepare_compression2(int);
  void prepare_compression3(int);
  void initialize_compression2();
  void prepare_sweep();
  void prepare_betadouble();
  void prepare_betadouble2(int);
  void rescale_dltu();
  void sweep();
  void sweep_compression();
  void sweep_compression2(double&,double&);
  void sweep_compression3(double&,double&);
  void hamiltonian_vector_multiplication_idmrg(int,int,tensor_su2&,tensor_su2&);
  void save_mps2(int);
  void save_mps2();
  void save_mps1();
  void save_enr();
  void save_enr2();
  void read_enr(int,int);
  bool read_mps(int,int,int);
  bool read_mps(int,int);
  void save_ww();
  void read_ww(int,int);
  void save_time_evolution();
  void save_time_evolution2(int,double,int);
  void save_compression_error(int,int,double,double);
  int read_time_evolution(int);
  void mea_enr(int);
  void mea_enr();
  void mea_enr(int,double*);
  void mea_enr_all_position();
  void mea_enr_compression();
  void mea_translation();
  void prepare_input_vector(int,int,tensor_su2&);
  void prepare_input_vector_initial(int,int,tensor_su2&);
  void prepare_input_vector_left(int,tensor_su2&);
  void prepare_input_vector_right(int,tensor_su2&);

  void makeup_clebsch_gordan_coefficient_tensors(int);
  void test_clebsch_gordan_coefficient();
  void wavefunc_transformation(int,int);
  void wavefunc_transformation(tensor_su2&,tensor_su2&,int);
  void initialize_prod_identities();
  void make_left_canonical_form2();
  void compute_error();
  void increase_max_dcut();
  void check_error(int,double*,int,int);

  void save_memory(int,int);
  void read_memory(int,int);
  void read_orth(int);
  void release_memory(int,int);
  void release_memory_all(int,int);
  void release_memory2(int,int);
};
