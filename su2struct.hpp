#include "su2bond.hpp"

using namespace std;
class su2struct{
private:
  int nbond,nten,locspin;
  int *bonddir,*nmoment,**angularmoment,**cgcdim,**bonddim;
public:
  su2struct();
  ~su2struct();
  su2struct(int,int,su2bond*);
  void set_su2struct(int,int,su2bond*);
  void clean();
  su2struct& operator = (su2struct&);
  bool operator != (su2struct&);
  bool operator == (su2struct&);
  inline  int get_nmoment(int i){return nmoment[i];}
  inline  int get_bonddir(int i){return bonddir[i];}
  inline  int get_angularmoment(int i,int j){return angularmoment[i][j];}
  inline  int get_bonddim(int i,int j){return bonddim[i][j];}
  inline  int get_cgcdim(int i,int j){return cgcdim[i][j];}
  inline  int get_locspin(){return locspin;}
  inline  int get_nbond(){return nbond;}
  inline  int get_nten(){return nten;}
  void get_su2bond(int,su2bond&);
  int get_angularmoment_index(int,int);
  int get_tensor_index(int*);
  bool get_tensor_argument(int,int*,int*,int*);
  bool check_angularmoments(int*);
  void take_conjugate();
  void take_conjugate(int);
  void print();
  void shift(int,int);
  void exchangeindex(int,int);
  void invert_bonddir(int);
  void get_nelement(int&,int&);
  void get_nelement(int&,int&,int*);
  inline  void set_bonddir(int i,int d){bonddir[i]=d;}
  void direct_sum(int,su2struct&,su2struct&);
};
