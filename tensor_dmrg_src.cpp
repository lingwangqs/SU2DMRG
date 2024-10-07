#include <omp.h>
#include "tensor.hpp"
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace std;

extern "C"{
  void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
  void dsyev_(char*,char*,int*,double*,int*,double*,double*,int*,int*);
  void dsyevx_(char*,char*,char*,int*,double*,int*,double*,double*,int*,int*,double*,int*,double*,double*,int*,double*,int*,int*,int*,int*);
  void dsyrk_(char*,char*,int*,int*,double*,double*,int*,double*,double*,int*);
  void dposv_(char*,int*,int*,double*,int*,double*,int*,int*);
  double ran_();
}
void obtain_symmetric_matrix_eigenvector_2(double*,double*,int,int);
void obtain_symmetric_matrix_eigenvector(double*,double*,int);
void get_tensor_index(int&,int,int*,int*);
void get_bond_index(int,int,int*,int*);

extern int max_dcut,myrank,psize;
extern double tolerance;

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_overlap_initial(tensor& t1, tensor& t2, int flag){
//--------------------------------------------------------------------------------------
  //flag=0 contract index 0,1 of t1,t2, flag=1 contract index 1,2 of t1,t2
  if(t1.nbond!=3||t2.nbond!=3){
    cout<<"tensor::contract_dmrg_overlap_initial can not perform"<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  else if(t1.bonddim[0]!=t2.bonddim[0]||t1.bonddim[1]!=t2.bonddim[1]){
    cout<<"tensor::contract_dmrg_overlap_initial the bond dimensions are not consistent"<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  char transa,transb;
  int nele,nbond1,*bdim;
  int m,n,k,i,j;
  double alpha=1,beta=0,*tele;
  k=t1.bonddim[0]*t1.bonddim[1];
  m=t1.nelement/k;
  n=t2.nelement/k;
  nele=m*n;
  tele=new double[nele];
  transa='T';
  transb='N';
  dgemm_(&transa,&transb,&m,&n,&k,&alpha,t1.getptr(),&k,t2.getptr(),&k,&beta,tele,&m);
  nbond1=2;
  bdim=new int[nbond1];
  bdim[0]=t1.get_bonddim(2);
  bdim[1]=t2.get_bonddim(2);
  if(telement!=NULL)delete []telement;
  telement=tele;
  if(bonddim!=NULL)delete []bonddim;
  bonddim=bdim;
  nbond=nbond1;
  nelement=nele;
  return *this;
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_initial(tensor& t1, tensor& t2, tensor& t3, int flag){
//--------------------------------------------------------------------------------------
  if(flag==0&&(t1.bonddim[1]!=t3.bonddim[0]||t2.bonddim[1]!=t3.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_initial bonddim not consistent"<<endl;
    exit(0);
  }
  else if(flag==1&&(t1.bonddim[0]!=t3.bonddim[0]||t2.bonddim[0]!=t3.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_initial bonddim not consistent"<<endl;
    exit(0);
  }
  tensor tmp1,tmp2;
  if(t3.bonddim[0]==1&&t3.bonddim[1]==1&&t3.bonddim[2]==1){
    this->contract_dmrg_overlap_initial(t1,t2,flag);
    (*this)*=t3.getptr()[0];
    this->separateindex(0,bonddim[0],1);
  }
  else if(flag==0){
    tmp1.contract(t1,1,t3,0);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(2,3);
    tmp2=t2;
    tmp2.mergeindex(0,1);
    this->contract(tmp1,2,tmp2,0);
  }
  else if(flag==1){
    tmp1.contract(t1,0,t3,0);
    tmp1.shift(3,0);
    tmp1.mergeindex(0,1);
    tmp2=t2;
    tmp2.mergeindex(0,1);
    this->contract(tmp1,0,tmp2,0);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_transformation(tensor& t1, tensor& t2, tensor& t3, int flag){
//--------------------------------------------------------------------------------------
  if(flag==0&&(t1.bonddim[0]!=t3.bonddim[0]||t2.bonddim[0]!=t3.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_transformation bonddim not consistent"<<endl;
    exit(0);
  }
  else if(flag==1&&(t1.bonddim[1]!=t3.bonddim[0]||t2.bonddim[1]!=t3.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_transformation bonddim not consistent"<<endl;
    exit(0);
  }
  tensor tmp1,tmp2;
  char transa,transb;
  int nele,nele1,nbond1,*bdim;
  int m,n,k;
  double alpha=1,beta=0,*tele,*tele1;
  //resume from here tonight
  if(t3.bonddim[1]==1&&(flag==0&&t1.bonddim[1]==1||flag==1&&t1.bonddim[0]==1)){
    k=t3.bonddim[0];
    m=t1.nelement/k;
    n=t3.nelement/k;
    nele1=m*n;
    tele1=new double[nele1];
    transa='T';
    transb='N';
    dgemm_(&transa,&transb,&m,&n,&k,&alpha,t1.getptr(),&k,t3.getptr(),&k,&beta,tele1,&m);
    k=t3.bonddim[2];
    m=nele1/k;
    n=t2.nelement/k;
    nele=m*n;
    tele=new double[nele];
    transa='N';
    transb='N';
    dgemm_(&transa,&transb,&m,&n,&k,&alpha,tele1,&m,t2.getptr(),&k,&beta,tele,&m);
    nbond1=3;
    bdim=new int[nbond1];
    bdim[0]=t1.bonddim[2];
    bdim[1]=t3.bonddim[1];
    bdim[2]=t2.bonddim[2];
    if(telement!=NULL)delete []telement;
    telement=tele;
    if(bonddim!=NULL)delete []bonddim;
    bonddim=bdim;
    nbond=nbond1;
    nelement=nele;
    delete []tele1;
  }
  else if(flag==0){
    tmp1.contract(t1,0,t3,0);
    tmp1.shift(1,0);
    tmp1.mergeindex(2,3);
    tmp2=t2;
    tmp2.mergeindex(0,1);
    this->contract(tmp1,2,tmp2,0);
  }
  else if(flag==1){
    tmp1.contract(t1,1,t3,0);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(2,3);
    tmp2=t2;
    tmp2.mergeindex(0,1);
    this->contract(tmp1,2,tmp2,0);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_pairup(tensor& uu, tensor& vv, tensor& op1, tensor& op2, int flag){
//--------------------------------------------------------------------------------------
  //this routine realize operator pairup in dmrg procedure
  if(flag==0&&(uu.bonddim[0]!=op1.bonddim[0]||vv.bonddim[0]!=op1.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_pairup bonddim not consistent"<<endl;
    exit(0);
  }
  else if(flag==1&&(uu.bonddim[1]!=op1.bonddim[0]||vv.bonddim[1]!=op1.bonddim[2])){
    cout<<"tensor::contract_dmrg_operator_pairup bonddim not consistent"<<endl;
    exit(0);
  }
  tensor tmp1,tmp2,tmp3;
  if(op1.bonddim[1]==1&&op2.bonddim[0]==1&&op2.bonddim[1]==1&&op2.bonddim[2]==1){
    this->contract_dmrg_operator_transformation(uu,vv,op1,flag);
    this->mergeindex(0,1);
    (*this)*=op2.getptr()[0];
  }
  else if(flag==0){
    tmp1.contract(uu,0,op1,0);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(0,1);
    tmp2=op2;
    tmp2.mergeindex(0,1);
    tmp3.contract(tmp1,0,tmp2,0);
    tmp3.mergeindex(1,2);
    tmp2=vv;
    tmp2.mergeindex(0,1);
    this->contract(tmp3,1,tmp2,0);
  }
  else if(flag==1){
    tmp1.contract(uu,1,op1,0);
    tmp1.shift(1,0);
    tmp1.mergeindex(0,1);
    tmp2=op2;
    tmp2.mergeindex(0,1);
    tmp3.contract(tmp2,0,tmp1,0);
    tmp3.mergeindex(0,1);
    tmp2=vv;
    tmp2.mergeindex(0,1);
    this->contract(tmp3,0,tmp2,0);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_transformation_step1(tensor& t1, tensor& t2,int nb, int* bdim,int flag){
//--------------------------------------------------------------------------------------
  int i,j,k,m1,m2,m3,n1,n2,n3,nr,nc,nele;
  double *aa,*bb,*cc,alpha=1,beta=0;
  clean();
  n1=t1.get_nelement();
  n2=t2.get_nelement();
  m1=t1.get_bonddim(0);
  m2=n2/m1;
  nele=1;
  for(i=0;i<nb;i++)nele*=bdim[i];
  nr=n1/m1;
  nc=m2;
  if(nr*nc!=nele){
    cout<<"wrong bonddim for operator transformation2 step1 2 nele="<<nele<<"\tnr"<<nr<<"\tnc="<<nc<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  aa=new double[nr*nc];
  dgemm_("T","N",&nr,&nc,&m1,&alpha,t1.getptr(),&m1,t2.getptr(),&m1,&beta,aa,&nr);
  this->copy(nb,bdim,aa);
  delete []aa;
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_transformation_step5(tensor& t1, tensor& t2, int nb, int* bdim,int flag){
//--------------------------------------------------------------------------------------
  int i,j,k,m1,m2,m3,n1,n2,n3,nr,nc,nele;
  double *aa,*bb,*cc,alpha=1,beta=0;
  tensor tmp;
  clean();
  if(flag==1){
    tmp=t2;
    tmp.shift(1,0);
  }
  n1=t1.get_nelement();
  n2=t2.get_nelement();
  m1=t1.get_bonddim(3);
  m2=n2/m1;
  nele=1;
  for(i=0;i<nb;i++)nele*=bdim[i];
  nr=n1/m1;
  nc=m2;
  if(nr*nc!=nele){
    cout<<"wrong bonddim for operator transformation2 step5 "<<endl;
    t1.print();
    t2.print();
    for(i=0;i<nb;i++)
      cout<<i<<"\t"<<bdim[i]<<endl;
    exit(0);
  }
  aa=new double[nr*nc];
  if(flag==0)
    dgemm_("N","N",&nr,&nc,&m1,&alpha,t1.getptr(),&nr,t2.getptr(),&m1,&beta,aa,&nr);
  else
    dgemm_("N","N",&nr,&nc,&m1,&alpha,t1.getptr(),&nr,tmp.getptr(),&m1,&beta,aa,&nr);
  this->copy(nb,bdim,aa);
  delete []aa;
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_transformation(tensor& t1, tensor& t2, tensor& t3, tensor& t4, tensor& t5, int flag){
//--------------------------------------------------------------------------------------
  if(flag==0&&(t1.bonddim[0]!=t3.bonddim[0]||t2.bonddim[0]!=t3.bonddim[2]||t3.bonddim[1]!=t4.bonddim[0]||t5.bonddim[0]!=t1.bonddim[1]||t5.bonddim[1]!=t4.bonddim[1]||t5.bonddim[2]!=t2.bonddim[1])){
    cout<<flag<<"\ttensor::contract_dmrg_operator_transformation 2, bonddim not consistent"<<endl;
    t1.print();
    t2.print();
    t3.print();
    t4.print();
    t5.print();
    exit(0);
  }
  if(flag==1&&(t1.bonddim[1]!=t3.bonddim[0]||t2.bonddim[1]!=t3.bonddim[2]||t3.bonddim[1]!=t4.bonddim[1]||t5.bonddim[0]!=t1.bonddim[0]||t5.bonddim[1]!=t4.bonddim[0]||t5.bonddim[2]!=t2.bonddim[0])){
    cout<<flag<<"\ttensor::contract_dmrg_operator_transformation 2, bonddim not consistent"<<endl;
    t1.print();
    t2.print();
    t3.print();
    t4.print();
    t5.print();
    exit(0);
  }
  tensor tmp1,tmp2,tmp3,tmp4,tmp5;
  int n1,n2,n3,m1,m2,m3,nr,bdim[3];
  double *aa,*bb,*cc,alpha=1,beta=0;
  if(t5.bonddim[0]==1&&t5.bonddim[1]==1&&t5.bonddim[2]==1){
    n1=t3.get_bonddim(0);
    n2=t3.get_bonddim(1);
    n3=t3.get_bonddim(2);
    m1=t1.get_bonddim(2);
    m2=t4.get_bonddim(2);
    m3=t2.get_bonddim(2);
    bdim[0]=m1;
    bdim[1]=m2;
    bdim[2]=m3;
    aa=new double[m1*n2*n3];
    nr=n2*n3;
    dgemm_("T","N",&nr,&m1,&n1,&alpha,t3.getptr(),&n1,t1.getptr(),&n1,&beta,aa,&nr);
    bb=new double[m1*m2*n3];    
    nr=m1*n3;
    dgemm_("T","N",&nr,&m2,&n2,&alpha,aa,&n2,t4.getptr(),&n2,&beta,bb,&nr);
    delete []aa;
    cc=new double[m1*m2*m3];    
    nr=m1*m2;
    dgemm_("T","N",&nr,&m3,&n3,&alpha,bb,&n3,t2.getptr(),&n3,&beta,cc,&nr);
    delete []bb;
    this->copy(3,bdim,cc);
    delete []cc;
    (*this)*=t5.telement[0];
    /*
    tmp1=t1;
    tmp2=t2;
    tmp4=t4;
    tmp1.mergeindex(0,1);
    tmp2.mergeindex(0,1);
    tmp4.mergeindex(0,1);
    tmp3.contract(t3,0,tmp1,0);
    tmp1.contract(tmp3,0,tmp4,0);
    this->contract(tmp1,0,tmp2,0);
    (*this)*=t5.telement[0];
    */
  }
  else{
    tmp1=t1;
    tmp2=t2;
    tmp4=t4;
    tmp5=t5;
    if(flag==1){
      tmp1.exchangeindex(0,1);
      tmp2.exchangeindex(0,1);
      tmp4.exchangeindex(0,1);
    }
    tmp1.mergeindex(1,2);
    tmp2.mergeindex(1,2);
    tmp4.mergeindex(1,2);
    tmp5.mergeindex(0,1);
    tmp3.contract(t3,0,tmp1,0);
    tmp1.contract(tmp3,0,tmp4,0);
    tmp3.contract(tmp1,0,tmp2,0);
    tmp3.separateindex(2,t2.bonddim[1-flag],t2.bonddim[2]);
    tmp3.separateindex(1,t4.bonddim[1-flag],t4.bonddim[2]);
    tmp3.separateindex(0,t1.bonddim[1-flag],t1.bonddim[2]);
    tmp3.exchangeindex(1,2);
    tmp3.mergeindex(0,1);
    this->contract(tmp5,0,tmp3,0);
    this->contractindex(0,3);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_operator_initial(tensor& t1, tensor& t2, tensor& t3, tensor& t4, int flag){
//--------------------------------------------------------------------------------------
  if(flag==0&&(t1.bonddim[1]!=t4.bonddim[0]||t2.bonddim[1]!=t4.bonddim[2]||t3.bonddim[1]!=t4.bonddim[1])){
    cout<<"tensor::contract_dmrg_operator_initial 2, bonddim not consistent"<<endl;
    exit(0);
  }
  if(flag==1&&(t1.bonddim[0]!=t4.bonddim[0]||t2.bonddim[0]!=t4.bonddim[2]||t3.bonddim[0]!=t4.bonddim[1])){
    cout<<"tensor::contract_dmrg_operator_initial 2, bonddim not consistent"<<endl;
    exit(0);
  }
  if(flag==0&&(t1.bonddim[0]!=1||t2.bonddim[0]!=1||t3.bonddim[0]!=1)){
    cout<<"wrong situation"<<endl;
    exit(0);
  }
  if(flag==1&&(t1.bonddim[1]!=1||t2.bonddim[1]!=1||t3.bonddim[1]!=1)){
    cout<<"wrong situation"<<endl;
    exit(0);
  }
  tensor tmp1,tmp2,tmp3,tmp4,tmp5;
  if(t4.bonddim[0]==1&&t4.bonddim[1]==1&&t4.bonddim[2]==1){
    tmp1=t1;
    tmp2=t2;
    tmp3=t3;
    tmp1.mergeindex(1,2);
    tmp2.mergeindex(1,2);
    tmp3.mergeindex(1,2);
    tmp1.mergeindex(0,1);
    tmp2.mergeindex(0,1);
    tmp3.mergeindex(0,1);
    tmp5.tensor_product(tmp1,tmp3);
    this->tensor_product(tmp5,tmp2);
    (*this)*=t4.telement[0];
  }
  else{
    tmp1=t1;
    tmp2=t2;
    tmp3=t3;
    tmp4=t4;
    tmp1.mergeindex(1,2);
    tmp2.mergeindex(1,2);
    tmp3.mergeindex(1,2);
    tmp1.mergeindex(0,1);
    tmp2.mergeindex(0,1);
    tmp3.mergeindex(0,1);
    tmp4.mergeindex(0,1);
    tmp5.tensor_product(tmp1,tmp3);
    tmp1.tensor_product(tmp5,tmp2);
    tmp1.separateindex(2,t2.bonddim[1-flag],t2.bonddim[2]);
    tmp1.separateindex(1,t3.bonddim[1-flag],t3.bonddim[2]);
    tmp1.separateindex(0,t1.bonddim[1-flag],t1.bonddim[2]);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(0,1);
    this->contract(tmp4,0,tmp1,0);
    this->contractindex(0,3);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_betadouble(tensor& u1, tensor& u2, tensor& projl, tensor& projr, tensor& projs){
//--------------------------------------------------------------------------------------
  clean();
  int a0,a1,a2,b0,b1,b2,ca0,ca1,ca2,cb0,cb1,cb2,cc2;
  tensor tmp1,tmp2,t1,t2,p1,p2,ps;
  double *pt;
  a0=u1.get_bonddim(0);
  a1=u1.get_bonddim(1);
  a2=u1.get_bonddim(2);
  b0=u2.get_bonddim(0);
  b1=u2.get_bonddim(1);
  b2=u2.get_bonddim(2);
  ca0=projl.get_bonddim(0);
  cb0=projl.get_bonddim(1);
  ca1=projs.get_bonddim(0);
  cb1=projs.get_bonddim(1);
  ca2=projr.get_bonddim(0);
  cb2=projr.get_bonddim(1);
  cc2=projr.get_bonddim(2);
  if(a0!=ca0||b0!=cb0||a2!=ca2||b2!=cb2||a1!=ca1||b1!=cb1){
    cout<<"contract_dmrg_betadouble wrong bond"<<endl;
    exit(0);
  }
  t1=u1;
  t2=u2;
  p1=projl;
  p2=projr;
  ps=projs;
  if(a1==1&&b1==1){
    t1.mergeindex(1,2);
    t2.mergeindex(1,2);
    tmp1.contract(p1,0,t1,0);
    tmp2.contract(tmp1,0,t2,0);
    tmp2.mergeindex(1,2);
    p2.mergeindex(0,1);
    this->contract(tmp2,1,p2,0);
    this->separateindex(1,1,cc2);
    pt=ps.getptr();
    (*this)*=pt[0];
  }
  else{
    tmp1.contract(p1,0,t1,0);
    tmp2.contract(tmp1,0,t2,0);
    tmp2.exchangeindex(2,3);
    tmp2.mergeindex(3,4);
    tmp2.mergeindex(1,2);
    ps.mergeindex(0,1);
    p2.mergeindex(0,1);
    tmp1.contract(tmp2,2,p2,0);
    this->contract(tmp1,1,ps,0);
    this->shift(1,0);
  }
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_permutation(tensor& uu, tensor& vv, tensor& vec, tensor& op1, tensor& op2, int flag){
//--------------------------------------------------------------------------------------
  tensor tmp1,tmp2,tmp3,tmp4;
  int bd1,bd2,bd3,bd4,bd5,bd6,zero=0;
  double fac1,fac2;
  bd1=op1.get_bonddim(0);
  bd2=op1.get_bonddim(1);
  bd3=op1.get_bonddim(2);
  bd4=op2.get_bonddim(0);
  bd5=op2.get_bonddim(1);
  bd6=op2.get_bonddim(2);
  if(bd1==1&&bd2==1&&bd3==1&&bd4==1&&bd5==1&&bd6==1||flag==0&&bd1==1&&bd2==1&&bd3==2&&bd4==1&&bd5==1&&bd6==2||flag==1&&bd1==2&&bd2==1&&bd3==1&&bd4==2&&bd5==1&&bd6==1){
    tmp1=uu;
    tmp2=vec;
    tmp3=vv;
    tmp1.mergeindex(0,1);
    tmp2.mergeindex(0,1);
    tmp3.mergeindex(0,1);
    tmp4.contract(tmp1,0,tmp2,0);
    this->contract(tmp4,1,tmp3,0);
    this->separateindex(0,this->get_bonddim(0),1);
    fac1=op1.inner_prod(op2);
    (*this)*=fac1;
    return *this;
  }
  else if(flag==0){
    tmp1=vv;
    tmp1.exchangeindex(0,1);
    tmp2.contract(op1,0,tmp1,0);
    tmp2.exchangeindex(1,2);
    tmp2.mergeindex(0,1);
    tmp1=vec;
    tmp1.mergeindex(1,2);
    tmp3.contract(tmp1,1,tmp2,0);
    tmp3.mergeindex(0,1);
    tmp1.contract(uu,1,op2,0);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(2,3);
    this->contract(tmp1,2,tmp3,0);
  }
  else if(flag==1){
    tmp1.contract(uu,0,op2,1);
    tmp1.exchangeindex(1,2);
    tmp1.mergeindex(0,1);
    tmp2=vec;
    tmp2.mergeindex(0,1);
    tmp3.contract(tmp1,0,tmp2,0);
    tmp3.mergeindex(1,2);
    tmp1=op1;
    tmp1.exchangeindex(0,1);
    tmp2.contract(tmp1,0,vv,0);
    tmp2.exchangeindex(1,2);
    tmp2.mergeindex(0,1);
    this->contract(tmp3,1,tmp2,0);
  }
  return *this;
  //this->print();
}

//--------------------------------------------------------------------------------------
tensor& tensor::contract_dmrg_hamiltonian_vector_multiplication(tensor& t1, tensor& t2, int flag){
//--------------------------------------------------------------------------------------
  //flag=0 contract index 0,1 of t1,t2, flag=1 contract index 1,2 of t1,t2
  if(t1.nbond!=3||t2.nbond!=3){
    cout<<"tensor::contract_dmrg_overlap_initial can not perform"<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  else if(t1.bonddim[0]!=t2.bonddim[0]||t1.bonddim[1]!=t2.bonddim[1]){
    cout<<"tensor::contract_dmrg_overlap_initial the bond dimensions are not consistent"<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  char transa,transb;
  int nele,nbond1,*bdim;
  int m,n,k,i,j;
  double alpha=1,beta=0,*tele;
  
  k=t1.bonddim[0]*t1.bonddim[1];
  m=t1.nelement/k;
  n=t2.nelement/k;
  nele=m*n;
  tele=new double[nele];
  transa='T';
  transb='N';
  dgemm_(&transa,&transb,&m,&n,&k,&alpha,t1.getptr(),&k,t2.getptr(),&k,&beta,tele,&m);
  nbond1=2;
  bdim=new int[nbond1];
  bdim[0]=t1.get_bonddim(2);
  bdim[1]=t2.get_bonddim(2);
  if(telement!=NULL)delete []telement;
  telement=tele;
  if(bonddim!=NULL)delete []bonddim;
  bonddim=bdim;
  nbond=nbond1;
  nelement=nele;
  return *this;
}
