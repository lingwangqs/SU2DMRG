using namespace std;

class tensor{
private:
  int nbond,nelement;
  int *bonddim;
  double *telement;
public:
  tensor();
  tensor(int,int*);
  tensor(int,int*,double*);
  ~tensor();
  void copy(int, int*, double*);
  void copy(tensor&);
  void clean();
  void random_init();
  void alloc_space(int,int*);
  void obtaintelement(double*,int,int);
  void obtaintelement(double*);
  void print();
  void rescale(double);
  double rescale();
  double get_norm();
  tensor& operator = (double);
  //tensor& operator = (const tensor&);
  tensor& operator = (tensor&);
  tensor& operator += (tensor&);
  tensor& operator -= (const tensor&);
  tensor& operator *= (double);
  tensor& operator /= (double);
  tensor operator - (const tensor&);
  tensor operator + (const tensor&);
  tensor operator * (double);
  tensor operator / (double);
  bool operator == (const tensor&);
  bool operator != (const tensor&);
  bool check_tensor_null();

  int get_bonddim(int i)const {return bonddim[i];}
  int get_nelement()const {return nelement;}
  int get_nbond()const {return nbond;}
  double get_telement(int i)const {return telement[i];}
  int* get_bonddim_ptr(){return bonddim;}
  double* getptr(int i){return &(telement[i]);}
  double* getptr(){return telement;}
  void get_telement(double*);
  void set_telement(double*);
  void set_telement(int,int,int,double);
  void exchangeindex(int,int);
  void mergeindex(int,int);
  void shift(int,int);
  void separateindex(int,int,int);
  tensor& contract(tensor&,int,tensor&,int);
  tensor& contract_v2(tensor&,int,tensor&,int);
  tensor& contractindex(int,int);
  tensor& contract_dmrg_overlap_initial(tensor&,tensor&,int);
  tensor& contract_dmrg_overlap_transformation(tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_operator_initial(tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_operator_initial(tensor&,tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_operator_transformation(tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_operator_transformation(tensor&,tensor&,tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_operator_transformation_step1(tensor&,tensor&,int,int*,int);
  tensor& contract_dmrg_operator_transformation_step5(tensor&,tensor&,int,int*,int);
  tensor& contract_dmrg_operator_pairup(tensor&,tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_betadouble(tensor&,tensor&,tensor&,tensor&,tensor&);
  tensor& contract_dmrg_permutation(tensor&,tensor&,tensor&,tensor&,tensor&,int);
  tensor& contract_dmrg_hamiltonian_vector_multiplication(tensor&,tensor&,int);


  void multiply_singular_value(int,double*);
  void devide_singular_value(int,double*);
  void svd(tensor&,double,tensor&,double,double*,int&,int);
  void svd(tensor&,double,tensor&,double,double*,int,int&,int);

  void hermitianmatrix_takeinverse();
  void symmetric_matrix_eigenvector(double);
  double inner_prod(tensor&);
  double take_overlap(tensor&);
  double take_trace();
  void direct_sum(int,tensor&,tensor&);
  void direct_sum(int,tensor&);
  void direct_subtract(int,int,tensor&);
  void direct_product(tensor&,tensor&);
  void tensor_product(tensor&,tensor&);
  void calculate_difference(tensor&,double&,double&);
  friend  double compare(tensor&,tensor&);

  void make_cgc(int,int,int);
  void make_cgc2(int,int,int);
  void make_singlet(int);
  void make_identity(int);
  void multiply_cgc(int,int,int,int,int);
  void shift_set_identity(int,int,int*);
  void shift_copy(int,int,int,tensor&);
  bool is_identity();
  bool is_minus_identity();
  bool is_null();
  bool is_zero();
  bool is_one();
  bool is_minus_one();
  bool is_proportional_to(tensor&,double&);
  void shift_copy(int,int,int,int,int,int,tensor&);
};
