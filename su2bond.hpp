using namespace std;

class su2bond{
private:
  int bonddir,nmoment,*angularmoment,*bonddim,*cgcdim;
public:
  su2bond();
  ~su2bond();
  su2bond(int,int,int*,int*);
  void set_su2bond(int,int,int*,int*);
  void fuse(su2bond&,su2bond&);
  void fuse(su2bond&,su2bond&,int);
  void fuse_to_multiplet(su2bond&,su2bond&,int);
  void clean();
  su2bond& operator = (const su2bond&);
  bool operator != (const su2bond&);
  bool operator == (const su2bond&);
  inline  int get_bonddir()const{return bonddir;}
  inline  int get_nmoment()const{return nmoment;}
  inline  int get_angularmoment(int i)const{return angularmoment[i];}
  inline  int get_bonddim(int i)const{return bonddim[i];}
  inline  int get_cgcdim(int i)const{return cgcdim[i];}
  inline  void invert_bonddir(){bonddir=-bonddir;}
  void set_bonddir(int);
  bool check_angularmoment(int);
  int get_angularmoment_index(int);
  void print();
  void direct_sum(su2bond&,su2bond&);
};
