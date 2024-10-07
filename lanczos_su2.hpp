#include "tensor_su2.hpp"
class lanczos_su2{
private:
  int nrep,mlanc,neig;
  double *aal,*nnl;
  double *eig,*vec;
  double enrexp;
  tensor_su2 *ff;

  void diatridiag(int);
  void check_eigenvector(int,int,tensor_su2&,double&,double&,int);
public:
  lanczos_su2();
  ~lanczos_su2();
  void diag_op(int,int,tensor_su2&,tensor_su2&,int);
  void initialize_lanczos(tensor_su2&,int,int);
  void lanczos1(int,int,tensor_su2&);
  void lanczos2(int,int,tensor_su2&,int);
  void lanczos3(int,int,tensor_su2&,int);
  void compute_eigenvector(int,int,tensor_su2&,int,int&,int);
  void compute_eigenvector(tensor_su2&,int);
  void compute_eigenvector(tensor_su2&,int,int);
  double get_eigval(int i){return eig[i];}
  double get_eigval(){return eig[0];}
  void compute_evolution(tensor_su2&,int,int);
  void compute_evolution(tensor_su2&,int,int,int);
};
