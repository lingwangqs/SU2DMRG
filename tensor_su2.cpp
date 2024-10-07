#include "tensor_su2.hpp"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;
extern "C"{
  void dsort2_(int*,double*,int*);
  //int memcmp(char*,char*,int);
  //void memcpy(char*,char*,int);
}
extern int comm_rank,psize,myrank,physpn2;
extern tensor ***spin_op,*cgc_coef_singlet,*identity;
extern double **spin_op_trace,*****fac_operator_onsite_left,*****fac_operator_onsite_rght,******fac_operator_transformation_left,******fac_operator_transformation_rght,*****fac_operator_pairup_left,*****fac_operator_pairup_rght,***fac_hamilt_vec;
extern "C"{
  void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
}
void obtain_symmetric_matrix_eigenvector(double*,double*,int);
bool sum_direct_product(tensor&,tensor&,tensor&,tensor&);
void sum_direct_product(tensor&,tensor&,tensor&,tensor&,tensor&,tensor&);

extern int max_dcut;
//--------------------------------------------------------------------------------------
tensor_su2::tensor_su2(){
//--------------------------------------------------------------------------------------
  nten=0;
  nbond=0;
  locspin=0;
  tarr=NULL;
  parr=NULL;
  tcgc=NULL;
  pcgc=NULL;
}

//--------------------------------------------------------------------------------------
tensor_su2::~tensor_su2(){
//--------------------------------------------------------------------------------------
  clean();
}

//--------------------------------------------------------------------------------------
void tensor_su2::clean(){
//--------------------------------------------------------------------------------------
  if(tarr!=NULL)delete []tarr;
  if(tcgc!=NULL)delete []tcgc;
  if(parr!=NULL)delete []parr;
  if(pcgc!=NULL)delete []pcgc;
  cgc.clean();
  tarr=NULL;
  tcgc=NULL;
  parr=NULL;
  pcgc=NULL;
  nbond=0;
  nten=0;
  locspin=0;
}

//--------------------------------------------------------------------------------------
void tensor_su2::print(){
//--------------------------------------------------------------------------------------
  cgc.print();

  int i,j,*angm,*bdim,*cdim;
  bool check;
  return;
  if(nbond==0)return;
  cout<<"nbond="<<nbond<<endl;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    //if(!parr[i]->is_zero()&&!pcgc[i]->is_zero()){
    if(check&&!parr[i]->is_null()&&!pcgc[i]->is_null()){
      cout<<"****i="<<i<<"****"<<endl;
      for(j=0;j<nbond;j++)
	cout<<"angm["<<j<<"]="<<angm[j]<<endl;
      //parr[i]->print();
      //pcgc[i]->print();
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
su2struct& tensor_su2::get_cgc(){
//--------------------------------------------------------------------------------------
  return cgc;
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator = (double a){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nten;i++)
    if(!parr[i]->is_null()){
      *(parr[i])=a;
    }
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator = (tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i;
  clean();
  cgc=t1.cgc;
  nbond=t1.get_nbond();
  nten=t1.get_nten();
  locspin=t1.get_locspin();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
    if(!t1.is_null(i)){
      tarr[i]=*(t1.get_parr(i));
      tcgc[i]=*(t1.get_pcgc(i));
    }
  }
  return *this;
}

//--------------------------------------------------------------------------------------
bool tensor_su2::operator != (tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i;
  if(cgc!=t1.cgc) return true;
  if(nbond!=t1.get_nbond()) return true;
  if(nten!=t1.get_nten()) return true;
  if(locspin!=t1.get_locspin()) return true;
  for(i=0;i<nten;i++){
    if(!parr[i]->is_null()||!t1.is_null(i)){
      if(*(parr[i])!=*(t1.get_parr(i))){
	parr[i]->print();
	t1.get_parr(i)->print();
	return true;
      }
      if(*(pcgc[i])!=*(t1.get_pcgc(i))){
	return true;
      }
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------
bool tensor_su2::operator == (tensor_su2& t1){
//--------------------------------------------------------------------------------------
  if(*this!=t1)return false;
  else return true;
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator += (tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  bool check;
  if(cgc!=t1.cgc){
    cout<<"tensor_su2::operator += can not operate"<<endl;
    t1.print();
    print();
    exit(0);    
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(!parr[i]->is_null()||!t1.get_parr(i)->is_null()){
      if(t1.get_parr(i)->is_null()||t1.get_pcgc(i)->is_zero())
	continue;
      else if(parr[i]->is_null()||pcgc[i]->is_zero()){
	*(parr[i])=*(t1.get_parr(i));
	*(pcgc[i])=*(t1.get_pcgc(i));
      }
      else if((*(pcgc[i]))==(*(t1.get_pcgc(i))))
	(*(parr[i]))+=(*(t1.get_parr(i)));
      else if(!sum_direct_product(*(parr[i]),*(pcgc[i]),*(t1.get_parr(i)),*(t1.get_pcgc(i)))){
	cout<<"tensor_su2::operator += can not perform"<<endl;
	this->print();
	t1.print();
	exit(0);
      }
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
  return *this;
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator -= (tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  bool check;
  tensor tmp;
  if(cgc!=t1.cgc){
    cout<<"tensor_su2::operator += can not operate"<<endl;
    t1.print();
    print();
    exit(0);    
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(!parr[i]->is_null()||!t1.get_parr(i)->is_null()){
      if(t1.get_parr(i)->is_null()||t1.get_pcgc(i)->is_zero())
	continue;
      else if(parr[i]->is_null()||pcgc[i]->is_zero()){
	*(parr[i])=*(t1.get_parr(i));
	*(pcgc[i])=*(t1.get_pcgc(i));
	*(parr[i])*=-1;
      }
      else{
	tmp=*(t1.get_parr(i));
	tmp*=-1;
	if(!sum_direct_product(*(parr[i]),*(pcgc[i]),tmp,*(t1.get_pcgc(i)))){
	  cout<<"tensor_su2::operator -= can not perform"<<endl;
	  this->print();
	  t1.print();
	  exit(0);
	}
      }
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
  return *this;
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator *= (double fac){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nten;i++)
    if(!parr[i]->is_null())
      *(parr[i])*=fac;
  return *this;
}

//--------------------------------------------------------------------------------------
tensor_su2& tensor_su2::operator /= (double fac){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nten;i++)
    if(!parr[i]->is_null())
      *(parr[i])/=fac;
  return *this;
}

//--------------------------------------------------------------------------------------
void tensor_su2::cgc_make_scalar_operator(){
//--------------------------------------------------------------------------------------
  if(nbond!=2||locspin!=0){
    cout<<"tensor_su2::cgc_check_scalar_operator, not a scalar operator"<<endl;
    cout<<"nbond="<<nbond<<"\tlocspin="<<locspin<<endl;
    exit(0);
  }
  int i;
  double fac;
  for(i=0;i<nten;i++){
    if(!parr[i]->is_null()){
      if(pcgc[i]->is_zero())
	*(parr[i])=0;	
      else{ 
	fac=pcgc[i]->rescale();
	if(pcgc[i]->is_identity())
	  *(parr[i])*=fac;
	else if(pcgc[i]->is_minus_identity())
	  *(parr[i])*=-fac;
	else if(pcgc[i]->get_bonddim(0)!=pcgc[i]->get_bonddim(1)){
	  //else if(pcgc[i]->get_bonddim(0)!=pcgc[i]->get_bonddim(1)&&parr[i]->is_zero()){
	  pcgc[i]->clean();
	  parr[i]->clean();
	}
	else{
	  cout<<"tensor_su2::cgc_make_scalar_operator, cgc is not identity"<<endl;
	  pcgc[i]->print();
	  parr[i]->print();
	  exit(0);
	}
      }
    }
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::cgc_make_cgc(){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  bool check;
  if(nbond!=3){
    cout<<"tensor_su2::cgc_make_cgc, can not perform make_cgc"<<endl;
    exit(0);
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    if(!parr[i]->is_null()){
      check=cgc.get_tensor_argument(i,angm,bdim,cdim);
      pcgc[i]->make_cgc(angm[0],angm[1],angm[2]);
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
  cgc.set_bonddir(0,1);
  cgc.set_bonddir(1,1);
  cgc.set_bonddir(2,-1);
}

//--------------------------------------------------------------------------------------
double tensor_su2::normalize_vector(){
//--------------------------------------------------------------------------------------
  int i,j,*angm,*bdim,*cdim,*flag;
  bool check;
  double nor,nor1,nor2;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  flag=new int[nten];
  nor=0;
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      flag[i]=1;
      nor1=parr[i]->inner_prod(*(parr[i]));
      nor2=pcgc[i]->inner_prod(*(pcgc[i]));      
      //nor+=nor1*nor2;
      nor+=nor1;
    }
    else flag[i]=0;
  }

  nor=sqrt(nor);
  for(i=0;i<nten;i++)
    if(flag[i])
      *(parr[i])/=nor;
  //if(comm_rank==0)cout<<"nor="<<nor<<endl;
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []flag;
  return nor;
}

//--------------------------------------------------------------------------------------
void tensor_su2::multiply_singular_value(int leg,double *w){
//--------------------------------------------------------------------------------------
  int i,j,nmom,*angm,*bdim,*cdim;
  double **ww;
  bool check;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  nmom=cgc.get_nmoment(leg);
  ww=new double*[nmom];
  j=0;
  for(i=0;i<nmom;i++){
    ww[i]=&(w[j]);
    j+=cgc.get_bonddim(leg,i);
  }
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      j=cgc.get_angularmoment_index(leg,angm[leg]);
      parr[i]->multiply_singular_value(leg,ww[j]);
    }
  }
  delete []ww;
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::devide_singular_value(int leg,double *w){
//--------------------------------------------------------------------------------------
  int i,j,nmom,*angm,*bdim,*cdim;
  double **ww;
  bool check;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  nmom=cgc.get_nmoment(leg);
  ww=new double*[nmom];
  j=0;
  for(i=0;i<nmom;i++){
    ww[i]=&(w[j]);
    j+=cgc.get_bonddim(leg,i);
  }
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      j=cgc.get_angularmoment_index(leg,angm[leg]);
      parr[i]->devide_singular_value(leg,ww[j]);
    }
  }
  delete []ww;
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::diagonalize(){
//--------------------------------------------------------------------------------------
  int m,n,a,i,j,nmom1,nmom2;
  double *eig;
  double *aa;  
  if(nbond==2){
    nmom1=get_nmoment(0);
    nmom2=get_nmoment(1);
    if(nmom1!=nmom2){
      cout<<"tensor_su2::diagonalize can not perform"<<endl;
      cout<<"nmom1!=nmom2"<<endl;
      exit(0);
    }
    for(i=0;i<nmom1;i++){
      m=cgc.get_bonddim(0,i);
      n=cgc.get_bonddim(1,i);
      a=cgc.get_angularmoment(0,i);
      if(m!=n){
	cout<<"tensor_su2::diagonalize can not perform"<<endl;
	cout<<"m!=n"<<endl;
	exit(0);
      }
      eig=new double[m];
      aa=new double[m*m];
      parr[i+i*nmom1]->get_telement(aa);
      obtain_symmetric_matrix_eigenvector(aa,eig,m);
      for(j=0;j<m;j++)
	//if(j<5)
	  cout<<"eig["<<j<<"]=\t"<<setw(10)<<eig[j]<<"\t"<<eig[j]*(a+1)<<endl;
      delete []aa;
      delete []eig;
    }
  }
  else{
    cout<<"tensor_su2::diagonalize can not perform"<<endl;
    cout<<"nbond!=2"<<endl;
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
int tensor_su2::get_nmoment(int i){
//--------------------------------------------------------------------------------------
  return cgc.get_nmoment(i);
}
//--------------------------------------------------------------------------------------
int tensor_su2::get_bonddir(int i){
//--------------------------------------------------------------------------------------
  return cgc.get_bonddir(i);
}
//--------------------------------------------------------------------------------------
int tensor_su2::get_angularmoment(int i,int j){
//--------------------------------------------------------------------------------------
  return cgc.get_angularmoment(i,j);
}
//--------------------------------------------------------------------------------------
int tensor_su2::get_bonddim(int i,int j){
//--------------------------------------------------------------------------------------
  return cgc.get_bonddim(i,j);
}
//--------------------------------------------------------------------------------------
int tensor_su2::get_cgcdim(int i,int j){
//--------------------------------------------------------------------------------------
  return cgc.get_cgcdim(i,j);
}

//--------------------------------------------------------------------------------------
void tensor_su2::get_su2bond(int i, su2bond& b){
//--------------------------------------------------------------------------------------
  cgc.get_su2bond(i,b);
}

//--------------------------------------------------------------------------------------
int tensor_su2::get_angularmoment_index(int i0, int angm){
//--------------------------------------------------------------------------------------
  return  cgc.get_angularmoment_index(i0,angm);
}

//--------------------------------------------------------------------------------------
int tensor_su2::get_tensor_index(int *angm){
//--------------------------------------------------------------------------------------
  return cgc.get_tensor_index(angm);
}

//--------------------------------------------------------------------------------------
bool tensor_su2::get_tensor_argument(int i0, int* angm, int* bdim, int* cdim){
//--------------------------------------------------------------------------------------
  return cgc.get_tensor_argument(i0,angm,bdim,cdim);
}

//--------------------------------------------------------------------------------------
tensor* tensor_su2::get_parr(int i){
//--------------------------------------------------------------------------------------
  if(i<nten)
    return parr[i];
  else{
    cout<<"tensor_su2::get_parr tensor index is too large"<<endl;
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
tensor* tensor_su2::get_pcgc(int i){
//--------------------------------------------------------------------------------------
  if(i<nten)
    return pcgc[i];
  else{
    cout<<"tensor_su2::get_pcgc tensor index is too large"<<endl;
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::take_conjugate(){
//--------------------------------------------------------------------------------------
  cgc.take_conjugate();
}

//--------------------------------------------------------------------------------------
void tensor_su2::get_nelement(int &nele1,int &nele2){
//--------------------------------------------------------------------------------------
  int i;

  nele1=0;
  nele2=0;
  for(i=0;i<nten;i++)
    if(!parr[i]->is_null()){
      nele1+=parr[i]->get_nelement();
      nele2+=pcgc[i]->get_nelement();
    }
}

//--------------------------------------------------------------------------------------
void tensor_su2::get_telement(double *tele1, double* tele2){
//--------------------------------------------------------------------------------------
  int i,nele1,nele2;

  nele1=0;
  nele2=0;
  for(i=0;i<nten;i++)
    if(!parr[i]->is_null()){
      parr[i]->get_telement(&(tele1[nele1]));
      pcgc[i]->get_telement(&(tele2[nele2]));
      nele1+=parr[i]->get_nelement();
      nele2+=pcgc[i]->get_nelement();
    }
}

//--------------------------------------------------------------------------------------
void tensor_su2::get_telement(double *tele1, double* tele2, int* iflag){
//--------------------------------------------------------------------------------------
  int i,nele1,nele2;

  nele1=0;
  nele2=0;
  for(i=0;i<nten;i++){
    iflag[i]=0;
    if(!parr[i]->is_null()){
      iflag[i]=1;
      parr[i]->get_telement(&(tele1[nele1]));
      pcgc[i]->get_telement(&(tele2[nele2]));
      nele1+=parr[i]->get_nelement();
      nele2+=pcgc[i]->get_nelement();
    }
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::take_conjugate(int ind){
//--------------------------------------------------------------------------------------
  if(ind>=nbond){
    cout<<"tensor_su2::take_conjugate on bond, bond index wrong\t"<<ind<<endl;
    exit(0);
  }
  int i,j,nmom,*angm,*bdim,*cdim;
  tensor tmp;
  bool check;

  nmom=cgc.get_nmoment(ind);
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check&&!parr[i]->is_null()){
      tmp.contract(*(pcgc[i]),ind,cgc_coef_singlet[angm[ind]],1);
      tmp.shift(0,(ind+1)%nbond);
      *(pcgc[i])=tmp;
    }
  }
  cgc.take_conjugate(ind);
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::conjugate(int ind){
//--------------------------------------------------------------------------------------
  if(ind>=nbond){
    cout<<"tensor_su2::take_conjugate on bond, bond index wrong\t"<<ind<<endl;
    exit(0);
  }
  int i,j,nmom,*angm,*bdim,*cdim;
  tensor *conj,tmp;
  bool check;

  nmom=cgc.get_nmoment(ind);
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check&&!pcgc[i]->is_null()){
      tmp.contract(*(pcgc[i]),ind,cgc_coef_singlet[angm[ind]],0);
      tmp.shift(0,(ind+1)%nbond);
      tmp*=sqrt(angm[ind]+1);
      *(pcgc[i])=tmp;
    }
  }
  cgc.take_conjugate(ind);
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
bool tensor_su2::is_null(int i){
//--------------------------------------------------------------------------------------
  return parr[i]->is_null();
}

//--------------------------------------------------------------------------------------
bool tensor_su2::is_null(){
//--------------------------------------------------------------------------------------
  if(nbond==0)return true;
  else return false;
}

//--------------------------------------------------------------------------------------
bool tensor_su2::check_angularmoments(int* angm){
//--------------------------------------------------------------------------------------
  return cgc.check_angularmoments(angm);
}

//--------------------------------------------------------------------------------------
double tensor_su2::inner_prod(tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i;
  double prod,prod1,prod2;
  if(cgc!=t1.cgc){
    cout<<"tensor_su2::inner_prod t1 has different su2 structure"<<endl;
    exit(0);
  }
  prod=0;
  for(i=0;i<nten;i++){
    if(!parr[i]->is_null()){
      prod1=parr[i]->inner_prod(*(t1.get_parr(i)));
      prod2=pcgc[i]->inner_prod(*(t1.get_pcgc(i)));
      if(fabs(prod2-1.)>1.e-12){
	cout<<"inner_prod wrong"<<endl;
	exit(0);
      }
      prod+=prod1;
    }
  }
  return prod;
}

//--------------------------------------------------------------------------------------
double tensor_su2::ss_inner_prod(tensor_su2& t1){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  double prod,prod1,prod2;
  bool check;
  if(cgc!=t1.cgc){
    cout<<"tensor_su2::ss_inner_prod t1 has different su2 structure"<<endl;
    this->print();
    t1.print();
    exit(0);
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  prod=0;
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check&&!parr[i]->is_null()){
      prod1=parr[i]->take_overlap(*(t1.get_parr(i)));
      if(nbond==3)
	prod2=sqrt(angm[2]+1)/sqrt(angm[0]+1);
      else prod2=1;
      prod+=prod1*prod2;
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
  return prod;
}


//--------------------------------------------------------------------------------------
double tensor_su2::take_trace(){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  double prod,prod1,prod2;
  bool check;
  if(nbond!=2){
    cout<<"tensor_su2::take_trace  wrong su2 tensor"<<endl;
    exit(0);
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  prod=0;
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check&&angm[0]==angm[1]&&!parr[i]->is_null()){
      prod1=parr[i]->take_trace();
      prod2=1;
      prod+=prod1*prod2;
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
  return prod;
}


//--------------------------------------------------------------------------------------
void tensor_su2::left2right_vectran(){
//--------------------------------------------------------------------------------------
  //this function change uu projector from j1,j2 fuse to j3, to j2,j3 fuse to j1
  int i,j,ishift,*angm,*bdim,*cdim,*angm1;
  bool check;
  tensor **parr1;
  tensor **pcgc1;
  su2struct cgc1;
  //three index, shift(1,0)
  ishift=2;
  cgc1=cgc;
  cgc1.shift(1,0);
  parr1=new tensor*[nten];
  pcgc1=new tensor*[nten];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  angm1=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    for(j=0;j<nbond;j++)
      angm1[(j+ishift)%nbond]=angm[j];
    j=cgc1.get_tensor_index(angm1);
    parr1[j]=parr[i];
    pcgc1[j]=pcgc[i];
    if(!parr1[j]->is_null()){
      parr1[j]->shift(1,0);
      pcgc1[j]->make_cgc(angm1[0],angm1[1],angm1[2]);
    }
  }
  cgc=cgc1;
  delete []parr;
  delete []pcgc;
  parr=parr1;
  pcgc=pcgc1;
  delete []angm;
  delete []angm1;
  delete []bdim;
  delete []cdim;
  cgc.set_bonddir(0,1);
  cgc.set_bonddir(1,1);
  cgc.set_bonddir(2,-1);
}

//--------------------------------------------------------------------------------------
void tensor_su2::right2left_vectran(){
//--------------------------------------------------------------------------------------
  //this function change uu projector from j1,j2 fuse to j3, to j2,j3 fuse to j1
  int i,j,ishift,*angm,*bdim,*cdim,*angm1;
  bool check;
  tensor **parr1;
  tensor **pcgc1;
  su2struct cgc1;
  //three index, shift(2,0)
  ishift=1;
  cgc1=cgc;
  cgc1.shift(2,0);
  parr1=new tensor*[nten];
  pcgc1=new tensor*[nten];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  angm1=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    for(j=0;j<nbond;j++)
      angm1[(j+ishift)%nbond]=angm[j];
    j=cgc1.get_tensor_index(angm1);
    parr1[j]=parr[i];
    pcgc1[j]=pcgc[i];
    if(!parr1[j]->is_null()){
      parr1[j]->shift(2,0);
      pcgc1[j]->make_cgc(angm1[0],angm1[1],angm1[2]);
    }
  }
  cgc=cgc1;
  delete []parr;
  delete []pcgc;
  parr=parr1;
  pcgc=pcgc1;
  delete []angm;
  delete []angm1;
  delete []bdim;
  delete []cdim;
  cgc.set_bonddir(0,1);
  cgc.set_bonddir(1,1);
  cgc.set_bonddir(2,-1);
}

//--------------------------------------------------------------------------------------
void tensor_su2::fuse(su2bond& b1, su2bond& b2){
//--------------------------------------------------------------------------------------
  int i,j,k,l,n1,n2,n3,m1,m2,m3,ms,me,d1,d2,d3;
  int *bdim,*pos;
  su2bond *bb;
  clean();
  nbond=3;
  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"tensor_su2::fuse two bonds had different directions"<<endl;
    exit(0);
  }
  bb=new su2bond[nbond];
  bb[0]=b1;
  bb[1]=b2;
  bb[2].fuse(bb[0],bb[1]);

  locspin=0;
  cgc.set_su2struct(3,0,bb);
  nten=cgc.get_nten();

  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];

  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  n1=bb[0].get_nmoment();
  n2=bb[1].get_nmoment();
  n3=bb[2].get_nmoment();

  pos=new int[n3];
  bdim=new int[nbond];

  for(i=0;i<n3;i++)
    pos[i]=0;

  for(j=0;j<n2;j++){
    m2=bb[1].get_angularmoment(j);
    d2=bb[1].get_bonddim(j);
    bdim[1]=d2;
    for(i=0;i<n1;i++){
      m1=bb[0].get_angularmoment(i);
      d1=bb[0].get_bonddim(i);
      bdim[0]=d1;
      ms=abs(m1-m2);
      me=m1+m2;
      for(m3=ms;m3<=me;m3+=2){
	k=bb[2].get_angularmoment_index(m3);
	d3=bb[2].get_bonddim(k);
	bdim[2]=d3;
	l=i+j*n1+k*n1*n2;
	pcgc[l]->make_cgc(m1,m2,m3);
	parr[l]->shift_set_identity(3,pos[k],bdim);
	pos[k]+=d1*d2;
      }
    }
  }
  for(i=0;i<n3;i++){
    if(bb[2].get_bonddim(i)!=pos[i]){
      cout<<"tensor_su2::fuse, wrong in fuse two su2 bond"<<endl;
      exit(0);
    }
  }
  delete []pos;
  delete []bdim;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::fuse(int ind1, int ind2){
//--------------------------------------------------------------------------------------
  int i,j,k,l,m,*angm,*bdim,*cdim,nbond1,nten1,nmom,*angm1,*angm2;
  bool check;
  tensor **parr1,*tarr1,ta,tc,te;
  tensor **pcgc1,*tcgc1,tb,td,tf;
  tensor_su2 tmp;
  su2struct cgc1;
  su2bond *bb,*bb2;
  if(ind2!=ind1+1||cgc.get_bonddir(ind1)!=cgc.get_bonddir(ind2)||ind2>=nbond){
    cout<<"tensor_su2::fuse: can not merge indices that are not next to each other or not have the same bond direction"<<endl;
    exit(0);
  }
  nbond1=nbond-1;
  bb=new su2bond[nbond1];
  bb2=new su2bond[2];
  get_su2bond(ind1,bb2[0]);
  get_su2bond(ind2,bb2[1]);
  bb2[0].invert_bonddir();
  bb2[1].invert_bonddir();
  tmp.fuse(bb2[0],bb2[1]);
  bb[ind1].fuse(bb2[0],bb2[1]);
  for(i=0;i<ind1;i++)
    get_su2bond(i,bb[i]);
  for(i=ind2;i<nbond-1;i++)
    get_su2bond(i+1,bb[i]);
  cgc1.set_su2struct(nbond1,locspin,bb);
  nten1=cgc1.get_nten();
  parr1=new tensor*[nten1];
  pcgc1=new tensor*[nten1];
  tarr1=new tensor[nten1];
  tcgc1=new tensor[nten1];
  for(i=0;i<nten1;i++){
    parr1[i]=&(tarr1[i]);
    pcgc1[i]=&(tcgc1[i]);
  }
  nmom=bb[ind1].get_nmoment();
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  angm1=new int[nbond1];
  angm2=new int[3];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check==false)continue;
    for(j=0;j<ind1;j++)
      angm1[j]=angm[j];
    for(j=ind2;j<nbond-1;j++)
      angm1[j]=angm[j+1];
    angm2[0]=angm[ind1];
    angm2[1]=angm[ind2];
    for(j=0;j<nmom;j++){
      angm1[ind1]=bb[ind1].get_angularmoment(j);
      angm2[2]=angm1[ind1];
      k=cgc1.get_tensor_index(angm1);
      l=tmp.get_tensor_index(angm2);
      if(cgc1.check_angularmoments(angm1)==false)continue;
      if(this->is_null(i)||tmp.is_null(l))continue;
      te=*(parr[i]);
      tf=*(pcgc[i]);
      te.mergeindex(ind1,ind2);
      tf.mergeindex(ind1,ind2);
      ta=*(tmp.get_parr(l));
      tb=*(tmp.get_pcgc(l));
      ta.mergeindex(0,1);
      tb.mergeindex(0,1);
      tc.contract(ta,0,te,ind1);
      td.contract(tb,0,tf,ind1);
      tc.shift(0,ind1);
      td.shift(0,ind1);
      if(parr1[k]->is_null()){
	tarr1[k]=tc;
	tcgc1[k]=td;
      }
      else{
	if(!sum_direct_product(tarr1[k],tcgc1[k],tc,td)){
	  cout<<"tensor_su2::fuse, su2 tensor fuse: wrong"<<endl;
	  exit(0);
	}
      }
    }
  }
  nbond=nbond1;
  nten=nten1;
  cgc=cgc1;
  delete []tarr;
  delete []tcgc;
  delete []parr;
  delete []pcgc;
  parr=parr1;
  pcgc=pcgc1;
  tarr=tarr1;
  tcgc=tcgc1;
  delete []angm;
  delete []angm1;
  delete []angm2;
  delete []bdim;
  delete []cdim;
  delete []bb;
  delete []bb2;
  make_standard_cgc();
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_standard_cgc(){
//--------------------------------------------------------------------------------------
  int i,j,k,l,*angm,*bdim,*cdim;
  bool check;
  tensor tmp;
  double nor;
  if(nbond!=3)return;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check==false)continue;
    tmp.make_cgc(angm[0],angm[1],angm[2]);
    if(pcgc[i]->is_proportional_to(tmp,nor)){
      *(pcgc[i])=tmp;
      *(parr[i])*=nor;
    }    
    else if(pcgc[i]->is_zero()){
      *(pcgc[i])=tmp;
      *(parr[i])*=0;
    }
    else{
      cout<<"tensor_su2::make_standard_cgc cgc is not correct"<<endl;
      pcgc[i]->print();
      tmp.print();
      exit(0);
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::fuse(su2bond& b1, su2bond& b2, int dir){
//--------------------------------------------------------------------------------------
  int i,j,k,l,n0,n1,n2,n3,m0,m1,m2,m3,ms,me,d0,d1,d2,d3;
  int *bdim,*pos,*bdim2;
  su2bond *bb;
  clean();
  nbond=3;

  bb=new su2bond[nbond];
  bb[1]=b1;
  bb[2]=b2;
  bb[0].fuse(bb[1],bb[2],dir);

  locspin=0;
  cgc.set_su2struct(3,0,bb);
  nten=cgc.get_nten();

  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];

  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  n0=bb[0].get_nmoment();
  n1=bb[1].get_nmoment();
  n2=bb[2].get_nmoment();

  pos=new int[n0];
  bdim=new int[nbond];
  bdim2=new int[nbond];

  for(i=0;i<n0;i++)
    pos[i]=0;

  for(k=0;k<n2;k++){
    m2=bb[2].get_angularmoment(k);
    d2=bb[2].get_bonddim(k);
    bdim[2]=d2;
    for(j=0;j<n1;j++){
      m1=bb[1].get_angularmoment(j);
      d1=bb[1].get_bonddim(j);
      bdim[1]=d1;
      ms=abs(m1-m2);
      me=m1+m2;
      for(m0=ms;m0<=me;m0+=2){
	i=bb[0].get_angularmoment_index(m0);
	d0=bb[0].get_bonddim(i);
	bdim[0]=d0;
	l=i+j*n0+k*n0*n1;
	pcgc[l]->make_cgc(m0,m1,m2);
	bdim2[0]=bdim[1];
	bdim2[1]=bdim[2];
	bdim2[2]=bdim[0];
	parr[l]->shift_set_identity(3,pos[i],bdim2);
	parr[l]->shift(2,0);
	pos[i]+=d1*d2;
      }
    }
  }
  for(i=0;i<n0;i++){
    if(bb[0].get_bonddim(i)!=pos[i]){
      cout<<"tensor_su2::fuse, wrong in fuse two su2 bond"<<endl;
      exit(0);
    }
  }
  delete []pos;
  delete []bdim;
  delete []bdim2;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::fuse_to_multiplet(su2bond& b1, su2bond& b2, int angm){
//--------------------------------------------------------------------------------------
  int i,j,k,l,n1,n2,n3,m1,m2,m3,ms,me,d1,d2,d3;
  int *bdim,*pos;
  su2bond *bb;
  clean();
  nbond=3;

  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"tensor_su2::fuse two bonds had different directions"<<endl;
    exit(0);
  }
  bb=new su2bond[nbond];
  bb[0]=b1;
  bb[1]=b2;
  bb[2].fuse_to_multiplet(bb[0],bb[1],angm);

  locspin=0;
  cgc.set_su2struct(3,0,bb);
  nten=cgc.get_nten();

  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];

  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  n1=bb[0].get_nmoment();
  n2=bb[1].get_nmoment();
  n3=bb[2].get_nmoment();

  pos=new int[n3];
  bdim=new int[nbond];

  for(i=0;i<n3;i++)
    pos[i]=0;

  for(j=0;j<n2;j++){
    m2=bb[1].get_angularmoment(j);
    d2=bb[1].get_bonddim(j);
    bdim[1]=d2;
    for(i=0;i<n1;i++){
      m1=bb[0].get_angularmoment(i);
      d1=bb[0].get_bonddim(i);
      bdim[0]=d1;
      ms=abs(m1-m2);
      me=m1+m2;
      for(m3=ms;m3<=me;m3+=2){
	if(m3!=angm)continue;
	k=bb[2].get_angularmoment_index(m3);
	d3=bb[2].get_bonddim(k);
	bdim[2]=d3;
	l=i+j*n1+k*n1*n2;
	pcgc[l]->make_cgc(m1,m2,m3);
	parr[l]->shift_set_identity(3,pos[k],bdim);
	pos[k]+=d1*d2;
      }
    }
  }
  for(i=0;i<n3;i++){
    if(bb[2].get_bonddim(i)!=pos[i]){
      cout<<"tensor_su2::fuse, wrong in fuse two su2 bond"<<endl;
      exit(0);
    }
  }
  delete []pos;
  delete []bdim;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::fuse_to_singlet(su2bond& b1, su2bond& b2){
//--------------------------------------------------------------------------------------
  int i,j,k,l,n1,n2,n3,m1,m2,m3,ms,me,d1,d2,d3;
  int *bdim,*pos;
  su2bond *bb;
  clean();
  nbond=2;

  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"tensor_su2::fuse two bonds had different directions"<<endl;
    exit(0);
  }
  bb=new su2bond[nbond];
  bdim=new int[2];
  bb[0]=b1;
  bb[1]=b2;

  locspin=0;
  cgc.set_su2struct(2,0,bb);
  nten=cgc.get_nten();

  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];

  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  n1=bb[0].get_nmoment();
  n2=bb[1].get_nmoment();

  for(j=0;j<n2;j++){
    m2=bb[1].get_angularmoment(j);
    bdim[1]=bb[1].get_bonddim(j);
    for(i=0;i<n1;i++){
      m1=bb[0].get_angularmoment(i);
      bdim[0]=bb[0].get_bonddim(i);
      if(m1==m2){
	l=i+j*n1;
	pcgc[l]->make_singlet(m1);
	parr[l]->alloc_space(2,bdim);
	(*parr[l])=1;
      }
    }
  }
  delete []bdim;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::contract(tensor_su2& t1, int i1, tensor_su2& t2, int i2){
//--------------------------------------------------------------------------------------
  su2bond *bb,*bb2;
  int i,j,k,l,i0,nbond1,nbond2,nmom;
  int *angm,*bdim,*cdim,*angm1,*angm2;
  tensor tmp1;
  tensor tmp2;
  bool check,check1,check2;

  bb2=new su2bond[2];
  clean();
  t1.get_su2bond(i1,bb2[0]);
  t2.get_su2bond(i2,bb2[1]);
  bb2[1].invert_bonddir();
  if(bb2[0]!=bb2[1]){
    if(comm_rank==0){
      cout<<"tensor_su2::contract, two su2 bonds can not be contracted, check su2bond parameters"<<endl;
      bb2[0].print();
      bb2[1].print();
      //t1.print();
      //t2.print();
    }
    exit(0);
  }
  locspin=t1.get_locspin()+t2.get_locspin();
  nbond1=t1.get_nbond();
  nbond2=t2.get_nbond();
  nbond=nbond1+nbond2-2;
  bb=new su2bond[nbond];
  for(i=0;i<nbond1-1;i++)
    t1.get_su2bond((i1+1+i)%nbond1,bb[i]);
  for(i=0;i<nbond2-1;i++)
    t2.get_su2bond((i2+1+i)%nbond2,bb[i+nbond1-1]);
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  angm=new int[nbond];
  angm1=new int[nbond1];
  angm2=new int[nbond2];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  nmom=bb2[0].get_nmoment();
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    for(k=0;k<nbond1-1;k++)
      angm1[(i1+1+k)%nbond1]=angm[k];
    for(k=0;k<nbond2-1;k++)
      angm2[(i2+1+k)%nbond2]=angm[k+nbond1-1];
    for(j=0;j<nmom;j++){
      angm1[i1]=bb2[0].get_angularmoment(j);
      angm2[i2]=angm1[i1];
      check1=t1.check_angularmoments(angm1);
      check2=t2.check_angularmoments(angm2);
      if(check==false||check1==false||check2==false)continue;
      k=t1.get_tensor_index(angm1);
      l=t2.get_tensor_index(angm2);
      if(t1.get_parr(k)->is_null()||t2.get_parr(l)->is_null())continue;
      if(tarr[i].is_null()){
	tarr[i].contract(*(t1.get_parr(k)),i1,*(t2.get_parr(l)),i2);
	tcgc[i].contract(*(t1.get_pcgc(k)),i1,*(t2.get_pcgc(l)),i2);
      }
      else{
	tmp1.contract(*(t1.get_parr(k)),i1,*(t2.get_parr(l)),i2);
	tmp2.contract(*(t1.get_pcgc(k)),i1,*(t2.get_pcgc(l)),i2);
	if(!sum_direct_product(tarr[i],tcgc[i],tmp1,tmp2)){
	  cout<<"tensor_su2::contract, su2 tensor contraction: not getting unitary tensors"<<endl;
	  tarr[i].print();
	  tcgc[i].print();
	  tmp1.print();
	  tmp2.print();
	  exit(0);
	}
      }
    }
  }
  delete []angm;
  delete []angm1;
  delete []angm2;
  delete []bdim;
  delete []cdim;
  delete []bb;
  delete []bb2;
}

//--------------------------------------------------------------------------------------
void tensor_su2::shift(int i0, int i1){
//--------------------------------------------------------------------------------------
  int i,j,ishift,*angm,*bdim,*cdim,*angm1;
  bool check;
  tensor **parr1;
  tensor **pcgc1;
  su2struct cgc1;
  if(i0==i1)return;
  if(i1>i0)ishift=i1-i0;
  else ishift=nbond-(i0-i1);
  cgc1=cgc;
  cgc1.shift(i0,i1);
  parr1=new tensor*[nten];
  pcgc1=new tensor*[nten];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  angm1=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    for(j=0;j<nbond;j++)
      angm1[(j+ishift)%nbond]=angm[j];
    j=cgc1.get_tensor_index(angm1);
    parr1[j]=parr[i];
    pcgc1[j]=pcgc[i];
    if(!parr1[j]->is_null()){
      parr1[j]->shift(i0,i1);
      pcgc1[j]->shift(i0,i1);
    }
  }
  cgc=cgc1;
  delete []parr;
  delete []pcgc;
  parr=parr1;
  pcgc=pcgc1;
  delete []angm;
  delete []angm1;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::exchangeindex(int ind1, int ind2){
//--------------------------------------------------------------------------------------
  int i,j,*angm,*bdim,*cdim;
  bool check;
  tensor **parr1;
  tensor **pcgc1;
  su2struct cgc1;

  if(ind1==ind2)return;

  cgc1=cgc;
  cgc1.exchangeindex(ind1,ind2);

  parr1=new tensor*[nten];
  pcgc1=new tensor*[nten];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    j=angm[ind1];
    angm[ind1]=angm[ind2];
    angm[ind2]=j;
    j=cgc1.get_tensor_index(angm);
    parr1[j]=parr[i];
    pcgc1[j]=pcgc[i];
    if(!parr1[j]->is_null()){
      parr1[j]->exchangeindex(ind1,ind2);
      pcgc1[j]->exchangeindex(ind1,ind2);
    }
  }
  cgc=cgc1;
  delete []parr;
  delete []pcgc;
  parr=parr1;
  pcgc=pcgc1;
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::set_tensor_su2(int nb, int locsp, su2bond* bb, tensor **pa){
//--------------------------------------------------------------------------------------
  bool check;
  int i,j,*angm,*bdim,*cdim;
  clean();
  nbond=nb;
  locspin=locsp;
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }  
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  j=0;
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tarr[i]=*(pa[j]);
      if(nbond==2)
	tcgc[i].make_identity(angm[0]);
      else if(nbond==3)
	tcgc[i].make_cgc(angm[0],angm[1],angm[2]);
      else{
	cout<<"set_tensor_su2 incorrect parameters"<<endl;
	exit(0);
      }
      j++;
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::set_tensor_su2(su2struct& cgc1, double *tele1, double* tele2,int* iflag){
//--------------------------------------------------------------------------------------
  bool check;
  int i,j1,j2,*angm,*bdim,*cdim;
  clean();
  cgc=cgc1;
  nbond=cgc.get_nbond();
  nten=cgc.get_nten();
  locspin=cgc.get_locspin();
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }  
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  j1=0;
  j2=0;
  for(i=0;i<nten;i++){
    if(iflag[i]){
      check=cgc.get_tensor_argument(i,angm,bdim,cdim);
      parr[i]->copy(nbond,bdim,&(tele1[j1]));
      pcgc[i]->copy(nbond,cdim,&(tele2[j2]));
      j1+=parr[i]->get_nelement();
      j2+=pcgc[i]->get_nelement();
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::set_tensor_su2(su2struct& cgc1, double *tele1, double *tele2){
//--------------------------------------------------------------------------------------
  bool check;
  int i,j1,j2,*angm,*bdim,*cdim;
  clean();
  cgc=cgc1;
  nbond=cgc.get_nbond();
  nten=cgc.get_nten();
  locspin=cgc.get_locspin();
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }  
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  j1=0;
  j2=0;
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      parr[i]->copy(nbond,bdim,&(tele1[j1]));
      pcgc[i]->copy(nbond,cdim,&(tele2[j2]));
      j1+=parr[i]->get_nelement();
      j2+=pcgc[i]->get_nelement();
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::set_tensor_su2(su2struct& cgc1, double *tele1){
//--------------------------------------------------------------------------------------
  bool check;
  int i,j1,j2,*angm,*bdim,*cdim;
  clean();
  cgc=cgc1;
  nbond=cgc.get_nbond();
  nten=cgc.get_nten();
  locspin=cgc.get_locspin();
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }  
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  j1=0;
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      parr[i]->copy(nbond,bdim,&(tele1[j1]));
      j1+=parr[i]->get_nelement();
    }
  }
  if(nbond==3)
    cgc_make_cgc();
  else if(nbond==2&&get_bonddir(0)==1&&get_bonddir(1)==-1)
    cgc_make_identity_cgc();
  else{
    cout<<"can not do set_tensor_su2"<<endl;
    this->print();
    exit(0);
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::cgc_make_identity_cgc(){
//--------------------------------------------------------------------------------------
  if(nbond!=2){
    cout<<"tensor_su2::cgc_make_identity_cgc, can not perform"<<endl;
    exit(0);
  }
  int i,*angm,*bdim,*cdim;
  bool check;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check)
      pcgc[i]->make_identity(angm[0]);
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
int tensor_su2::svd(tensor_su2& tu, double p1, tensor_su2& tv, double p2, double *wout){
//--------------------------------------------------------------------------------------
  int dout;
  svd(tu,p1,tv,p2,dout,wout);
}

//--------------------------------------------------------------------------------------
int tensor_su2::svd(tensor_su2& tu, double p1, tensor_su2& tv, double p2, int& dout, double *wout){
//--------------------------------------------------------------------------------------
  int i,j,k,m,n,n0,a1,a2,m1,m2,nmom1,nmom2,nmom3,*dc,*dim,*bdim,*isort,*amom;
  int nm1,nm2,dcut;
  tensor tmp,**puarr,**pvarr,*uu,*vv;
  tensor_su2 proju,projv,tmp_su2,left,rght,tmp1,tmp2;
  double **ww,*wsort,tol=1.e-64;
  su2bond bb[2],bb2[4],bb3[2];
  //print();
  nm1=tu.get_nmoment(1);
  tu.get_su2bond(0,bb2[0]);
  tu.get_su2bond(1,bb2[1]);
  tv.get_su2bond(0,bb2[2]);
  tv.get_su2bond(1,bb2[3]);

  bb[0].fuse(bb2[0],bb2[1]);
  bb[1].fuse(bb2[2],bb2[3]);

  nmom1=bb[0].get_nmoment();
  nmom2=bb[1].get_nmoment();
  uu=new tensor[nmom1];
  vv=new tensor[nmom1];
  dc=new int[nmom1];
  dim=new int[nmom1];
  bdim=new int[nmom1+2];
  amom=new int[nmom1];
  ww=new double*[nmom1];
  nm1=0;
  for(i=0;i<nmom1;i++){
    ww[i]=new double[bb[0].get_bonddim(i)+max_dcut];
    nm1+=bb[0].get_bonddim(i)+max_dcut;
  }
  wsort=new double[nm1];
  isort=new int[nm1];
  for(i=0;i<nmom1;i++){
    dc[i]=0;
    dim[i]=0;
    bdim[i]=0;
  }

  n=0;
  n0=0;
  for(i=0;i<nmom1;i++){
    a1=bb[0].get_angularmoment(i);
    m1=bb[0].get_bonddim(i);
    for(j=0;j<nmom2;j++){
      a2=bb[1].get_angularmoment(j);
      m2=bb[1].get_bonddim(j);
      if(a1!=a2)continue;
      bdim[0]=m1;
      bdim[1]=m2;
      //tmp.copy(2,bdim,parr[0]->getptr(n0));
      if(parr[i+j*nmom1]->get_nbond()==0)dc[i]=0;
      else{
	tmp.copy(2,bdim,parr[i+j*nmom1]->getptr());
	tmp.svd(uu[i],p1,vv[i],p2,ww[i],dc[i],1);
	for(k=0;k<dc[i];k++){
	  wsort[n+k]=ww[i][k];
	  isort[n+k]=i;
	  //cout<<i<<"\t"<<k<<"\t"<<wsort[n+k]<<endl;
	}
	n0+=m1*m2;
	n+=dc[i];
      }
    }
  }

  dsort2_(&n,wsort,isort);
  if(max_dcut>n)k=0;
  else k=n-max_dcut;
  for(i=n-1;i>=k;i--){
    for(j=0;j<nmom1;j++)
      if(isort[i]==j){
	dim[j]++;
	break;
      }
  }
  dcut=0;
  nmom3=0;
  for(i=0;i<nmom1;i++)
    if(dim[i]!=0){
      nmom3++;
      dcut+=dim[i];
    }
  puarr=new tensor*[nmom3];
  pvarr=new tensor*[nmom3];
  j=0;
  m=0;
  m1=dcut;
  for(i=0;i<nmom1;i++){
    if(dim[i]>0&&dim[i]<dc[i]){
      uu[i].direct_subtract(1,dc[i]-dim[i],tmp);
      vv[i].direct_subtract(1,dc[i]-dim[i],tmp);
    }
    if(dim[i]>0){
      puarr[j]=&(uu[i]);
      pvarr[j]=&(vv[i]);
      bdim[j]=dim[i];
      amom[j]=bb[0].get_angularmoment(i);
      for(k=0;k<dim[i];k++)
	wout[m+k]=ww[i][dc[i]-dim[i]+k];
      for(k=0;k<dc[i]-dim[i];k++)
	wout[m1+k]=ww[i][k];
      m+=dim[i];
      m1+=dc[i]-dim[i];
      j++;
    }
    else{
      for(k=0;k<dc[i];k++)
	wout[m1+k]=ww[i][k];
      m1+=dc[i];
    }
  }
  dout=m1;
  bb3[1].set_su2bond(nmom3,-1,amom,bdim);
  bb3[0]=bb[0];
  bb3[0].invert_bonddir();
  proju.set_tensor_su2(2,0,bb3,puarr);
  bb3[0]=bb[1];
  bb3[0].invert_bonddir();
  projv.set_tensor_su2(2,0,bb3,pvarr);
  left.fuse(bb2[0],bb2[1]);
  rght.fuse(bb2[2],bb2[3]);
  tu.contract(left,2,proju,0);
  tv.contract(rght,2,projv,0);
  nm2=tu.get_nmoment(2);
  delete []uu;
  delete []vv;
  delete []dc;
  delete []dim;
  delete []bdim;
  delete []amom;
  for(i=0;i<nmom1;i++)
    delete []ww[i];
  delete []ww;
  delete []wsort;
  delete []isort;
  delete []puarr;
  delete []pvarr;
}

//--------------------------------------------------------------------------------------
void tensor_su2::svd(tensor_su2& tu, double p1, tensor_su2& tv, double p2, int dcut, int& dout, double *wout){
//--------------------------------------------------------------------------------------
  int i,j,k,m,n,n0,a1,a2,m1,m2,nmom1,nmom2,nmom3,*dc,*dim,*bdim,*isort,*amom;
  int nm1,nm2;
  tensor tmp,*uu,*vv,**puarr,**pvarr;
  tensor_su2 proju,projv,tmp_su2,left,rght,tmp1,tmp2;
  double **ww,*wsort;
  su2bond bb[2],bb2[4],bb3[2];
  this->get_su2bond(0,bb[0]);
  this->get_su2bond(1,bb[1]);
  nmom1=bb[0].get_nmoment();
  nmom2=bb[1].get_nmoment();
  uu=new tensor[nmom1];
  vv=new tensor[nmom1];
  dc=new int[nmom1];
  dim=new int[nmom1];
  bdim=new int[nmom1+2*psize];
  amom=new int[nmom1];
  ww=new double*[nmom1];
  nm1=0;
  for(i=0;i<nmom1;i++){
    ww[i]=new double[cgc.get_bonddim(0,i)*2];
    nm1+=cgc.get_bonddim(0,i)*2;
  }
  wsort=new double[nm1];
  isort=new int[nm1];
  for(i=0;i<nmom1;i++){
    dc[i]=0;
    dim[i]=0;
    bdim[i]=0;
  }

  //#pragma omp parallel for default(shared) private(myrank,i,j,a1,a2,m1,m2,tmp) schedule(dynamic,1)
  for(i=0;i<nmom1;i++){
    myrank=omp_get_thread_num();
    a1=bb[0].get_angularmoment(i);
    m1=bb[0].get_bonddim(i);
    for(j=0;j<nmom2;j++){
      a2=bb[1].get_angularmoment(j);
      m2=bb[1].get_bonddim(j);
      if(a1!=a2)continue;
      bdim[2*myrank+0]=m1;
      bdim[2*myrank+1]=m2;
      //tmp.copy(2,bdim,parr[0]->getptr(n0));
      if(parr[i+j*nmom1]->get_nbond()==0)dc[i]=0;
      else{
	tmp.copy(2,&(bdim[2*myrank]),parr[i+j*nmom1]->getptr());
	tmp.svd(uu[i],p1,vv[i],p2,ww[i],dcut,dc[i],1);
	//cout<<"i="<<i<<"\tm1="<<m1<<"\tm2="<<m2<<"\tdcut="<<dc[i]<<endl;
      }
    }
  }
  n=0;
  n0=0;
  for(i=0;i<nmom1;i++){
    for(k=0;k<dc[i];k++){
      wsort[n+k]=ww[i][k];
      isort[n+k]=i;
      //cout<<i<<"\t"<<k<<"\t"<<wsort[n+k]<<endl;
    }
    n0+=m1*m2;
    n+=dc[i];
  }
  dsort2_(&n,wsort,isort);
  if(dcut>n)k=0;
  else k=n-dcut;
  for(i=n-1;i>=k;i--){
    for(j=0;j<nmom1;j++)
      if(isort[i]==j){
	dim[j]++;
	break;
      }
  }
  nmom3=0;
  for(i=0;i<nmom1;i++)
    if(dim[i]!=0){
      nmom3++;
      //cout<<"i="<<i<<"\tdcut="<<dc[i]<<"\tdim="<<dim[i]<<endl;
    }
  puarr=new tensor*[nmom3];
  pvarr=new tensor*[nmom3];
  j=0;
  m=0;
  m1=dcut;
  for(i=0;i<nmom1;i++){
    if(dim[i]>0&&dim[i]<dc[i]){
      uu[i].direct_subtract(1,dc[i]-dim[i],tmp);
      vv[i].direct_subtract(1,dc[i]-dim[i],tmp);
    }
    if(dim[i]>0){
      puarr[j]=&(uu[i]);
      pvarr[j]=&(vv[i]);
      bdim[j]=dim[i];
      amom[j]=bb[0].get_angularmoment(i);
      for(k=0;k<dim[i];k++)
	wout[m+k]=ww[i][dc[i]-dim[i]+k];
      for(k=0;k<dc[i]-dim[i];k++)
	wout[m1+k]=ww[i][k];
      m+=dim[i];
      m1+=dc[i]-dim[i];
      j++;
    }
    else{
      for(k=0;k<dc[i];k++)
	wout[m1+k]=ww[i][k];
      m1+=dc[i];
    }
  }
  dout=m1;
  
  bb3[1].set_su2bond(nmom3,-bb[0].get_bonddir(),amom,bdim);
  bb3[0]=bb[0];
  tu.set_tensor_su2(2,0,bb3,puarr);
  bb3[1].set_su2bond(nmom3,-bb[1].get_bonddir(),amom,bdim);
  bb3[0]=bb[1];
  tv.set_tensor_su2(2,0,bb3,pvarr);
  delete []uu;
  delete []vv;
  delete []dc;
  delete []dim;
  delete []bdim;
  delete []amom;
  for(i=0;i<nmom1;i++)
    delete []ww[i];
  delete []ww;
  delete []wsort;
  delete []isort;
  delete []puarr;
  delete []pvarr;
}

//--------------------------------------------------------------------------------------
bool sum_direct_product(tensor& ta1,tensor& tb1,tensor& ta2,tensor& tb2){
//--------------------------------------------------------------------------------------
  double nor1;
  double nor2;
  tensor tmp1;
  if(tb1.is_zero()){
    ta1=ta2;
    tb1=tb2;
  }
  else if(tb2.is_zero())
    return true;
  else if(tb1==tb2)
    ta1+=ta2;
  else if(ta1==ta2)
    tb1+=tb2;
  else if(tb2.is_proportional_to(tb1,nor1)){
    tmp1=ta2;
    tmp1*=nor1;
    ta1+=tmp1;
  }
  else if(ta1.is_proportional_to(ta2,nor2)){
    cout<<"this case in sum_direct_product will not occure"<<endl;
    exit(0);
  }
  else{
    cout<<"sum_direct_product(1,2,3,4) can not perform"<<endl;
    ta1.print();
    tb1.print();
    ta2.print();
    tb2.print();
    exit(0);
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------------------
void sum_direct_product(tensor& ta1, tensor& tb1, tensor& ta2, tensor& tb2, tensor& ta3, tensor& tb3){
//--------------------------------------------------------------------------------------
  bool pass1,pass2;
  pass1=sum_direct_product(ta1,tb1,ta2,tb2);  
  pass2=sum_direct_product(ta1,tb1,ta3,tb3);  
  if(!pass1||!pass2){
    cout<<"sum_direct_product can not sum two direct product tensors"<<endl;
    ta1.print();
    ta2.print();
    ta3.print();
    tb1.print();
    tb2.print();
    tb3.print();
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::operator_tensor_product_identity(tensor_su2& op, su2bond& bd_idn){
//--------------------------------------------------------------------------------------
  int i,j,*angm,*bdim,*cdim,nbond1,nten1;
  su2bond *bb;
  bool check;
  tensor iden;
  clean();
  nbond1=op.get_nbond();
  nbond=nbond1+2;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  locspin=op.get_locspin();
  for(i=0;i<nbond1;i++)
    op.get_su2bond(i,bb[i]);
  bb[nbond1]=bd_idn;
  bb[nbond1+1]=bd_idn;
  bb[nbond1].set_bonddir(bb[nbond1-1].get_bonddir());
  bb[nbond1+1].set_bonddir(-bb[nbond1-1].get_bonddir());
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  delete []bb;
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check&&angm[nbond1]==angm[nbond1+1]){
      j=op.get_tensor_index(angm);
      if(op.get_parr(j)->is_null())continue;
      iden.make_identity(bdim[nbond1]-1);
      tarr[i].tensor_product(*(op.get_parr(j)),iden);
      tcgc[i].tensor_product(*(op.get_pcgc(j)),identity[angm[nbond1]]);
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::operator_tensor_product_identity(tensor_su2& op, tensor_su2& ten_idn){
//--------------------------------------------------------------------------------------
  int i,j,k,*angm,*bdim,*cdim,nbond1,nten1;
  su2bond *bb;
  bool check;
  clean();
  nbond1=op.get_nbond();
  nbond=nbond1+2;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  locspin=op.get_locspin();
  for(i=0;i<nbond1;i++)
    op.get_su2bond(i,bb[i]);
  ten_idn.get_su2bond(0,bb[nbond1]);;
  ten_idn.get_su2bond(1,bb[nbond1+1]);;
  bb[nbond1].set_bonddir(bb[nbond1-1].get_bonddir());
  bb[nbond1+1].set_bonddir(-bb[nbond1-1].get_bonddir());
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  delete []bb;
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check&&angm[nbond1]==angm[nbond1+1]){
      j=op.get_tensor_index(angm);
      k=ten_idn.get_tensor_index(&(angm[nbond1]));
      if(op.get_parr(j)->is_null()||ten_idn.get_parr(k)->is_null())continue;
      tarr[i].tensor_product(*(op.get_parr(j)),*(ten_idn.get_parr(k)));
      tcgc[i].tensor_product(*(op.get_pcgc(j)),*(ten_idn.get_pcgc(k)));
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::spin_operator_tensor_product_identity(tensor_su2& op,int physpn){
//--------------------------------------------------------------------------------------
  int i,j,*angm,*bdim,*cdim,nbond1,nten1;
  su2bond *bb;
  bool check;
  clean();
  nbond1=op.get_nbond();
  nten1=op.get_nten();
  angm=new int[nbond1];
  bdim=new int[nbond1+2];
  cdim=new int[nbond1];
  nten=nten1;
  nbond=nbond1+2;
  bb=new su2bond[nbond];
  locspin=op.get_locspin();
  for(i=0;i<nbond1;i++)
    op.get_su2bond(i,bb[i]);
  i=physpn;
  j=1;
  bb[nbond1].set_su2bond(1,-1,&i,&j);
  bb[nbond1+1].set_su2bond(1,1,&i,&j);
  cgc.set_su2struct(nbond,locspin,bb);
  delete []bb;
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  bdim[nbond1]=1;
  bdim[nbond1+1]=1;
  for(i=0;i<nten1;i++){
    check=op.get_tensor_argument(i,angm,bdim,cdim);
    if(check&&!op.get_parr(i)->is_null()){
      tarr[i].copy(nbond,bdim,op.get_parr(i)->getptr());
      tcgc[i].tensor_product(*(op.get_pcgc(i)),identity[physpn]);
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinor_start(int physpn){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *angm,*bdim;
  double *tele;
  //direction -1 out going, direction 1 in going
  //order: down, horizontal, up
  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  tele=new double[1];
  angm[0]=physpn;
  bdim[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=2;
  bdim[0]=1;
  bb[1].set_su2bond(1,-1,angm,bdim);
  angm[0]=physpn;
  bdim[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[1];
  tcgc=new tensor[1];
  parr=new tensor*[1];
  pcgc=new tensor*[1];
  parr[0]=&(tarr[0]);
  pcgc[0]=&(tcgc[0]);

  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  tele[0]=1;
  tarr[0].copy(nbond,bdim,tele);
  tcgc[0].make_cgc(physpn,2,physpn);
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_identity_start(){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *angm,*bdim;
  double *tele;
  //direction -1 out going, direction 1 in going
  //order: down, horizontal, up
  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  tele=new double[1];
  angm[0]=1;
  bdim[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  bdim[0]=1;
  bb[1].set_su2bond(1,-1,angm,bdim);
  angm[0]=1;
  bdim[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[1];
  tcgc=new tensor[1];
  parr=new tensor*[1];
  pcgc=new tensor*[1];
  parr[0]=&(tarr[0]);
  pcgc[0]=&(tcgc[0]);

  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  tele[0]=1;
  tarr[0].copy(nbond,bdim,tele);
  tcgc[0].make_cgc(1,0,1);
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_identity_end(){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *angm,*bdim;
  double *tele;
  tensor tmp;
  //direction -1 out going, direction 1 in going
  //order: down, horizontal, up
  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  tele=new double[1];
  angm[0]=1;
  bdim[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  bdim[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=1;
  bdim[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[1];
  tcgc=new tensor[1];
  parr=new tensor*[1];
  pcgc=new tensor*[1];
  parr[0]=&(tarr[0]);
  pcgc[0]=&(tcgc[0]);

  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  tele[0]=1;
  tarr[0].copy(nbond,bdim,tele);
  tcgc[0].make_cgc(1,0,1);
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_spinor_pull_through_left(){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *angm,*bdim,nele;
  double *tele;
  tensor tmp1,tmp2;
  //direction -1 out going, direction 1 in going
  //order down, right, up, left
  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  tele=new double[1];
  angm[0]=1;
  bdim[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=1;
  bdim[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=2;
  bdim[0]=1;
  bb[2].set_su2bond(1,-1,angm,bdim);
  angm[0]=2;
  bdim[0]=1;
  bb[3].set_su2bond(1,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[1];
  tcgc=new tensor[1];
  parr=new tensor*[1];
  pcgc=new tensor*[1];
  parr[0]=&(tarr[0]);
  pcgc[0]=&(tcgc[0]);

  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  tele[0]=1;
  tarr[0].copy(nbond,bdim,tele);
  tmp1.make_cgc(0,1,1);
  tmp2.make_cgc(0,2,2);
  tcgc[0].contract(tmp1,0,tmp2,0);
  
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_spinor_pull_through_right(){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *angm,*bdim,nele;
  double *tele;
  tensor tmp1,tmp2;
  //direction -1 out going, direction 1 in going
  //order down, right, up, left
  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  tele=new double[1];
  angm[0]=1;
  bdim[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=1;
  bdim[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=2;
  bdim[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);
  angm[0]=2;
  bdim[0]=1;
  bb[3].set_su2bond(1,-1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[1];
  tcgc=new tensor[1];
  parr=new tensor*[1];
  pcgc=new tensor*[1];
  parr[0]=&(tarr[0]);
  pcgc[0]=&(tcgc[0]);

  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  tele[0]=1;
  tarr[0].copy(nbond,bdim,tele);
  tmp1.make_cgc(0,1,1);
  tmp2.make_cgc(0,2,2);
  tcgc[0].contract(tmp1,0,tmp2,0);
  
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::makeup_input_vector(){
//--------------------------------------------------------------------------------------
  int i,j,k,nmom0,nmom1,a0,a1,bdim[2];
  nmom0=cgc.get_nmoment(0);
  nmom1=cgc.get_nmoment(1);
  for(i=0;i<nmom0;i++){
    a0=cgc.get_angularmoment(0,i);
    bdim[0]=cgc.get_bonddim(0,i);
    for(j=0;j<nmom1;j++){
      a1=cgc.get_angularmoment(1,j);
      bdim[1]=cgc.get_bonddim(1,j);
      if(a0==a1&&parr[i+j*nmom0]->is_null()){
	parr[i+j*nmom0]->alloc_space(2,bdim);
	*(pcgc[i+j*nmom0])=cgc_coef_singlet[a0];
      }
    }
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::initialize_input_vector(){
//--------------------------------------------------------------------------------------
  int i,j,k,nmom0,nmom1,a0,a1,bdim[2];
  su2bond bb[2];
  nmom0=cgc.get_nmoment(0);
  nmom1=cgc.get_nmoment(1);
  for(i=0;i<nmom0;i++){
    a0=cgc.get_angularmoment(0,i);
    bdim[0]=cgc.get_bonddim(0,i);
    for(j=0;j<nmom1;j++){
      a1=cgc.get_angularmoment(1,j);
      bdim[1]=cgc.get_bonddim(1,j);
      if(a0==a1){
	(*parr[i+j*nmom0])=1.;
	pcgc[i+j*nmom0]->make_singlet(a0);
      }
    }
  }
}

//--------------------------------------------------------------------------------------
void tensor_su2::random_initialize_tensor(){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,*cdim;
  bool check;
  if(nbond==0)return;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check)
      parr[i]->random_init();
  }
  this->normalize_vector();
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void tensor_su2::direct_sum(int ind, tensor_su2& t1, tensor_su2& t2){
//--------------------------------------------------------------------------------------
  su2struct cgc1,cgc2;
  int i,j,k,l,*angm,*bdim,*cdim,nten1,nten2;  
  bool check;
  clean();
  cgc1=t1.get_cgc();
  cgc2=t2.get_cgc();
  cgc.direct_sum(ind,cgc1,cgc2);
  nbond=cgc.get_nbond();
  nten=cgc.get_nten();
  locspin=cgc.get_locspin();
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  nten1=t1.get_nten();
  nten2=t2.get_nten();
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check==false)continue;
    j=t1.get_tensor_index(angm);
    k=t2.get_tensor_index(angm);
    if(j==nten1&&k!=nten2){
      tarr[i]=*(t2.get_parr(k));
      tcgc[i]=*(t2.get_pcgc(k));
    }
    else if(j!=nten1&&k==nten2){
      tarr[i]=*(t1.get_parr(j));
      tcgc[i]=*(t1.get_pcgc(j));
    }
    else if(j!=nten1&&k!=nten2){
      tarr[i].direct_sum(ind,*(t1.get_parr(j)),*(t2.get_parr(k)));
      tcgc[i]=*(t1.get_pcgc(j));
    }
    else if(j==nten1&&k==nten2){
      cout<<"tensor_su2::direct_sum wrong with cgc"<<endl;
      exit(0);
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

