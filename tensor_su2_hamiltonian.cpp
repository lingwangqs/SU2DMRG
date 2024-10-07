#include <omp.h>
//#include <mpi.h>
#include "dmrg_su2_omp.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <time.h>
using namespace std;

void get_tensor_index(int&,int,int*,int*);
void get_bond_index(int,int,int*,int*);
extern double *****fac_permutation_left,*****fac_permutation_rght,****fac_operator_onsite_left,****fac_operator_onsite_rght;
extern tensor *cgc_coef_singlet;
extern int comm_rank,psize;
//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_Qterm(tensor_su2& iden_op1, tensor_su2& plq1, tensor_su2& iden_op2, tensor_su2& plq2){
//--------------------------------------------------------------------------------------
//construct plqterm
  su2bond bb[3];
  int mom[1],dim[1];
  tensor_su2 tmp,tmp1,tmp2;
  mom[0]=2;
  dim[0]=1;
  bb[0].set_su2bond(1,1,mom,dim);

  tmp.make_spinor_start(1);
  plq1.operator_tensor_product_identity(tmp,bb[0]);
  plq1.exchangeindex(3,4);
  plq1.exchangeindex(2,3);
  plq1.fuse(1,2);
  plq1.get_su2bond(2,bb[1]);
  plq1.get_su2bond(3,bb[2]);
  iden_op1.fuse(bb[1],bb[2]);
  plq1.fuse(2,3);
  plq1.make_standard_cgc();
  plq1*=sqrt(3)/2;
  tmp.conjugate(1);
  plq2.operator_tensor_product_identity(tmp,bb[0]);
  plq2.exchangeindex(2,3);
  plq2.fuse(1,2);
  plq2.exchangeindex(1,3);
  plq2.get_su2bond(2,bb[1]);
  plq2.get_su2bond(3,bb[2]);
  iden_op2.fuse(bb[1],bb[2]);
  plq2.fuse(2,3);
  plq2.make_standard_cgc();
  plq2*=-sqrt(3)/2;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_Qterm_2(tensor_su2& iden_op3, tensor_su2& plq3){
//--------------------------------------------------------------------------------------
//construct plqterm
  su2bond bb[3];
  int mom[1],dim[1];
  tensor_su2 tmp,tmp1,tmp2;
  mom[0]=2;
  dim[0]=1;
  bb[0].set_su2bond(1,1,mom,dim);

  tmp.make_spinor_start(1);
  plq3.operator_tensor_product_identity(tmp,bb[0]);
  plq3.exchangeindex(3,4);
  plq3.exchangeindex(2,3);
  plq3.exchangeindex(1,2);
  plq3.fuse(1,2);
  plq3.get_su2bond(2,bb[1]);
  plq3.get_su2bond(3,bb[2]);
  iden_op3.fuse(bb[1],bb[2]);
  plq3.fuse(2,3);
  plq3.make_standard_cgc();
  plq3*=sqrt(3)/2;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_rightmove(tensor_su2& iden_op){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele; 
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  angm[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[1].set_su2bond(2,-1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[3].set_su2bond(2,1,angm,bdim);
  iden_op.fuse(bb[2],bb[3]);
  
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp2.make_singlet(1);
  tmp3.direct_product(tmp2,tmp1);
  tmp.direct_product(tmp3,tmp2);
  tmp.exchangeindex(3,5);
  tmp.mergeindex(4,5);
  tmp.mergeindex(1,2);
  tmp*=-2;//here multiply -(2s+1)
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[1]);
      tmp2.make_cgc(1,1,angm[3]);
      tmp1.mergeindex(0,1);
      tmp2.mergeindex(0,1);
      tmp3.contract(tmp,1,tmp1,0);
      tcgc[i].contract(tmp3,1,tmp2,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  this->fuse(2,3);
  this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_leftmove(){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele; 
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  angm[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[1].set_su2bond(2,-1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[3].set_su2bond(2,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp3.direct_product(tmp1,tmp1);
  tmp.direct_product(tmp3,tmp1);
  tmp.exchangeindex(1,2);
  tmp.mergeindex(3,4);
  tmp.mergeindex(1,2);
  tmp.exchangeindex(0,1);
  tmp.shift(3,0);
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[1]);
      tmp2.make_cgc(1,1,angm[3]);
      tmp1.mergeindex(0,1);
      tmp2.mergeindex(0,1);
      tmp3.contract(tmp,1,tmp1,0);
      tcgc[i].contract(tmp3,1,tmp2,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  this->fuse(2,3);
  this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_rightmove_4legs(tensor_su2& iden_op){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele;
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  angm[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[1].set_su2bond(2,-1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[3].set_su2bond(2,1,angm,bdim);
  iden_op.fuse(bb[2],bb[3]);
  
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp2.make_singlet(1);
  tmp3.direct_product(tmp2,tmp1);
  tmp.direct_product(tmp3,tmp2);
  tmp.exchangeindex(3,5);
  tmp.mergeindex(4,5);
  tmp.mergeindex(1,2);
  tmp*=-2;//here multiply -(2s+1)
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[1]);
      tmp2.make_cgc(1,1,angm[3]);
      tmp1.mergeindex(0,1);
      tmp2.mergeindex(0,1);
      tmp3.contract(tmp,1,tmp1,0);
      tcgc[i].contract(tmp3,1,tmp2,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  //this->fuse(2,3);
  //this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_leftmove_4legs(){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele; 
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=4;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  angm[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[1].set_su2bond(2,-1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[3].set_su2bond(2,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp3.direct_product(tmp1,tmp1);
  tmp.direct_product(tmp3,tmp1);
  tmp.exchangeindex(1,2);
  tmp.mergeindex(3,4);
  tmp.mergeindex(1,2);
  tmp.exchangeindex(0,1);
  tmp.shift(3,0);
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[1]);
      tmp2.make_cgc(1,1,angm[3]);
      tmp1.mergeindex(0,1);
      tmp2.mergeindex(0,1);
      tmp3.contract(tmp,1,tmp1,0);
      tcgc[i].contract(tmp3,1,tmp2,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  //this->fuse(2,3);
  //this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_leftend(){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele; 
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  
  angm[0]=1;
  bb[0].set_su2bond(1,-1,angm,bdim);
  angm[0]=0;
  angm[1]=2;
  bb[1].set_su2bond(2,-1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp2.make_singlet(1);
  tmp.direct_product(tmp2,tmp1);
  tmp.mergeindex(1,2);
  tmp*=sqrt(2);
  
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[1]);
      tmp1.mergeindex(0,1);
      tcgc[i].contract(tmp,1,tmp1,0);
      tcgc[i].shift(1,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_rightend(){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele; 
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  
  angm[0]=0;
  angm[1]=2;
  bb[0].set_su2bond(2,1,angm,bdim);
  angm[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,-1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp2.make_singlet(1);
  tmp.direct_product(tmp2,tmp1);
  tmp.exchangeindex(1,2);
  tmp.mergeindex(0,1);
  tmp*=-sqrt(2);
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[0]);
      tmp1.mergeindex(0,1);
      tcgc[i].contract(tmp1,0,tmp,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  this->make_standard_cgc();
  this->conjugate(0);
  this->shift(2,0);
  this->make_standard_cgc();
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_permutation_rightend_v2(){
//--------------------------------------------------------------------------------------
  //direction -1 out going, direction 1 in going
  //order: down, right, up, left
  su2bond *bb;
  int i,*angm,*bdim,*cdim,bdim2[2];
  double iden[4];
  double *tele;
  tensor tmp1,tmp2,tmp3,tmp;
  bool check;

  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  tele=new double[1];
  tele[0]=1;
  bdim[0]=1;
  bdim[1]=1;
  bdim[2]=1;
  bdim[3]=1;
  
  angm[0]=0;
  angm[1]=2;
  bb[0].set_su2bond(2,1,angm,bdim);
  angm[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=1;
  bb[2].set_su2bond(1,-1,angm,bdim);

  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }

  iden[0]=1;
  iden[1]=0;
  iden[2]=0;
  iden[3]=1;
  bdim2[0]=2;
  bdim2[1]=2;
  tmp1.copy(2,bdim2,iden);
  tmp2.make_singlet(1);
  tmp.direct_product(tmp2,tmp1);
  tmp.exchangeindex(1,2);
  tmp.mergeindex(0,1);
  tmp*=-sqrt(2);
  for(i=0;i<nten;i++){
    check=cgc.get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      tmp1.make_cgc(1,1,angm[0]);
      tmp1.mergeindex(0,1);
      tcgc[i].contract(tmp1,0,tmp,0);
      tarr[i].copy(nbond,bdim,tele);
    }
  }
  this->make_standard_cgc();
  /*
  this->conjugate(0);
  this->shift(2,0);
  this->make_standard_cgc();
  */
  delete []angm;
  delete []bdim;
  delete []cdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinor_end(int physpn){
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
  angm[0]=2;
  bdim[0]=1;
  bb[0].set_su2bond(1,1,angm,bdim);
  angm[0]=physpn;
  bdim[0]=1;
  bb[1].set_su2bond(1,1,angm,bdim);
  angm[0]=physpn;
  bdim[0]=1;
  bb[2].set_su2bond(1,-1,angm,bdim);

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
  tcgc[0].make_cgc(2,physpn,physpn);
  delete []angm;
  delete []bdim;
  delete []tele;
  delete []bb;
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_on_vec_left(tensor_su2& vec, tensor_su2& sigma, int flag){
//--------------------------------------------------------------------------------------
  tensor_su2 tmp,stmp;
  tmp=vec;
  tmp.exchangeindex(0,1);
  if(flag==0){
    this->contract(tmp,0,sigma,0);
    this->exchangeindex(1,3);
    this->fuse(2,3);
  }
  else if(flag==1){
    this->contract(sigma,2,tmp,0);
    this->exchangeindex(1,2);
    this->fuse(0,1);
  }
  this->make_standard_cgc();
}

//--------------------------------------------------------------------------------------
void tensor_su2::make_spinhalf_on_vec_right(tensor_su2& vec, tensor_su2& sigma, int flag){
//--------------------------------------------------------------------------------------
  tensor_su2 tmp,stmp;
  if(flag==0){
    this->contract(vec,0,sigma,0);
    this->exchangeindex(1,3);
    this->exchangeindex(0,1);
    this->fuse(2,3);
  }
  else if(flag==1){
    this->contract(sigma,2,vec,0);
    this->exchangeindex(1,2);
    this->fuse(0,1);
    this->exchangeindex(0,1);
  }
  this->make_standard_cgc();
}
/*
//--------------------------------------------------------------------------------------
void tensor_su2::permutation(tensor_su2& uu, tensor_su2& vv, tensor_su2& vec, tensor_su2& op1, tensor_su2& op2, int flag){
//--------------------------------------------------------------------------------------
  int a0,a1,a2,b0,b1,b2,c0,c1,c2,d0,d1,d2,e0,e1,e2,nmoma0,nmoma1,nmoma2,nmomb0,nmomb1,nmomb2,nmomc0,nmomc1,nmomc2,nmomd0,nmomd1,nmomd2,nmome0,nmome1,nmome2,i0,i1,i2,j0,j1,j2,k0,k1,k2,l0,l1,l2,s0,s1,s2,m,n,p,q,r,t,i;
  su2bond *bb;
  tensor tmp1,tmp2,tmp3;
  int bdim[5],angm[3];
  double fac;
  clean();
  nbond=3;
  locspin=0;
  bb=new su2bond[nbond];
  uu.get_su2bond(2,bb[0]);
  if(flag==0)op2.get_su2bond(1,bb[1]);
  else if(flag==1)op1.get_su2bond(2,bb[1]);
  vv.get_su2bond(2,bb[2]);
  bb[2].invert_bonddir();
  cgc.set_su2struct(nbond,locspin,bb);
  delete []bb;
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  nmoma0=uu.get_nmoment(0);
  nmoma1=uu.get_nmoment(1);
  nmoma2=uu.get_nmoment(2);
  nmomb0=vv.get_nmoment(0);
  nmomb1=vv.get_nmoment(1);
  nmomb2=vv.get_nmoment(2);
  nmome0=vec.get_nmoment(0);
  nmome1=vec.get_nmoment(1);
  nmome2=vec.get_nmoment(2);
  nmomc0=op1.get_nmoment(0);
  nmomc1=op1.get_nmoment(1);
  nmomc2=op1.get_nmoment(2);
  nmomd0=op2.get_nmoment(0);
  nmomd1=op2.get_nmoment(1);
  nmomd2=op2.get_nmoment(2);

  for(i0=0;i0<nmoma0;i0++){
    a0=uu.get_angularmoment(0,i0);
    for(i1=0;i1<nmoma1;i1++){
      a1=uu.get_angularmoment(1,i1);
      for(i2=0;i2<nmoma2;i2++){
	a2=uu.get_angularmoment(2,i2);
	angm[0]=a0;
	angm[1]=a1;
	angm[2]=a2;
	if(uu.check_angularmoments(angm)==false)continue;

	for(j0=0;j0<nmomb0;j0++){
	  b0=vv.get_angularmoment(0,j0);
	  for(j1=0;j1<nmomb1;j1++){
	    b1=vv.get_angularmoment(1,j1);
	    for(j2=0;j2<nmomb2;j2++){
	      b2=vv.get_angularmoment(2,j2);
	      angm[0]=b0;
	      angm[1]=b1;
	      angm[2]=b2;
	      if(vv.check_angularmoments(angm)==false)continue;

	      for(k0=0;k0<nmomc0;k0++){
		c0=op1.get_angularmoment(0,k0);
		for(k1=0;k1<nmomc1;k1++){
		  c1=op1.get_angularmoment(1,k1);
		  for(k2=0;k2<nmomc2;k2++){
		    c2=op1.get_angularmoment(2,k2);
		    angm[0]=c0;
		    angm[1]=c1;
		    angm[2]=c2;
		    if(op1.check_angularmoments(angm)==false)continue;

		    for(l0=0;l0<nmomd0;l0++){
		      d0=op2.get_angularmoment(0,l0);
		      for(l1=0;l1<nmomd1;l1++){
			d1=op2.get_angularmoment(1,l1);
			for(l2=0;l2<nmomd2;l2++){
			  d2=op2.get_angularmoment(2,l2);
			  angm[0]=d0;
			  angm[1]=d1;
			  angm[2]=d2;
			  if(op2.check_angularmoments(angm)==false)continue;

			  for(s0=0;s0<nmome0;s0++){
			    e0=vec.get_angularmoment(0,s0);
			    for(s1=0;s1<nmome1;s1++){
			      e1=vec.get_angularmoment(1,s1);
			      for(s2=0;s2<nmome2;s2++){
				e2=vec.get_angularmoment(2,s2);
				angm[0]=e0;
				angm[1]=e1;
				angm[2]=e2;
				if(vec.check_angularmoments(angm)==false)continue;
			  
				if(flag==0&&(a1!=d0||b1!=c0||c2!=d2||c1!=e1||a0!=e0||b0!=e2))continue;
				if(flag==1&&(a0!=d1||b0!=c1||c0!=d0||d2!=e1||a1!=e0||b1!=e2))continue;

				m=i0+i1*nmoma0+i2*nmoma0*nmoma1;
				n=j0+j1*nmomb0+j2*nmomb0*nmomb1;
				p=k0+k1*nmomc0+k2*nmomc0*nmomc1;
				q=l0+l1*nmomd0+l2*nmomd0*nmomd1;
				r=s0+s1*nmome0+s2*nmome0*nmome1;
				if(flag==0){
				  i=i2+l1*nmoma2+j2*nmoma2*nmomd1;
				  t=c1/2+((c2-1)/2)*3+(d1/2)*9;
				}
				else if(flag==1){
				  i=i2+k2*nmoma2+j2*nmoma2*nmomc2;
				  t=c2/2+((c0-1)/2)*3+(d2/2)*9;
				}
				if(uu.is_null(m)||vv.is_null(n)||vec.is_null(r)||op1.is_null(p)||op2.is_null(q))continue;
				tmp1.contract_dmrg_permutation(*(uu.get_parr(m)),*(vv.get_parr(n)),*(vec.get_parr(r)),*(op1.get_parr(p)),*(op2.get_parr(q)),flag);
				if(flag==0)
				  fac=fac_permutation_left[a0][a2][b0][b2][t];
				else if(flag==1)
				  fac=fac_permutation_rght[a1][a2][b1][b2][t];
				tmp1*=fac;
				if(tarr[i].is_null()){
				  tarr[i]=tmp1;
				  if(flag==0)tcgc[i].make_cgc(a2,d1,b2);
				  else if(flag==1)tcgc[i].make_cgc(a2,c2,b2);
				}
				else
				  tarr[i]+=tmp1;
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
}
*/

//--------------------------------------------------------------------------------------
void tensor_su2::permutation(tensor_su2& uu, tensor_su2& vv, tensor_su2& vec, tensor_su2& op1, tensor_su2& op2, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,t,nb,nt,nt1,nt2,nt3,nt4,nt5,**angm1,**angm2,**angm3,**angm4,**angm5,**bc1,**bc2,**bc3,**bc4,**bc5,**bd1,**bd2,**bd3,**bd4,**bd5,**bdim;
  double fac,fac1;
  tensor tmp,*tarr_tmp,step1,step2,tmp1,tmp2;
  tensor *tcgc_tmp;
  clean();
  uu.get_su2bond(flag,bb[0]);
  vec.get_su2bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"permutation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_su2bond(flag,bb[0]);
  vec.get_su2bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"permutation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  if(flag==0){
    op1.get_su2bond(2,bb[0]);
    op2.get_su2bond(2,bb[1]);
  }
  else if(flag==1){
    op1.get_su2bond(0,bb[0]);
    op2.get_su2bond(0,bb[1]);
  }
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"permutation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }

  nbond=3;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  vv.get_su2bond(2,bb[2]);
  bb[2].invert_bonddir();
  if(flag==0)
    op2.get_su2bond(1,bb[1]);
  else if(flag==1)
    op1.get_su2bond(2,bb[1]);
  cgc.set_su2struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  tarr=new tensor[nten];
  tcgc=new tensor[nten];
  parr=new tensor*[nten];
  pcgc=new tensor*[nten];
  for(i=0;i<nten;i++){
    parr[i]=&(tarr[i]);
    pcgc[i]=&(tcgc[i]);
  }
  tarr_tmp=new tensor[psize*nten];
  tcgc_tmp=new tensor[psize*nten];

  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bc4=new int*[psize];
  bc5=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bd4=new int*[psize];
  bd5=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  angm4=new int*[psize];
  angm5=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bc4[0]=new int[psize*nb];
  bc5[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bd4[0]=new int[psize*nb];
  bd5[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  angm4[0]=new int[psize*nb];
  angm5[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bc4[i]=&(bc4[0][i*nb]);
    bc5[i]=&(bc5[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bd4[i]=&(bd4[0][i*nb]);
    bd5[i]=&(bd5[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
    angm4[i]=&(angm4[0][i*nb]);
    angm5[i]=&(angm5[0][i*nb]);
  }

  nt1=uu.get_nten();
  nt2=vec.get_nten();
  nt3=vv.get_nten();
  nt4=op1.get_nten();
  nt5=op2.get_nten();
  nt=nt1*nt2;
  nb=3;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,t,fac,fac1,step1,step2,tmp,tmp1,tmp2) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(uu.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(vec.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm2[myrk][0]!=angm1[myrk][flag])continue;
    if(vec.get_parr(k)->is_null()||uu.get_parr(j)->is_null())continue;
    tmp1=(*uu.get_parr(j));
    tmp2=(*vec.get_parr(k));
    tmp1.mergeindex(0,1);
    tmp2.mergeindex(0,1);
    step1.contract(tmp1,0,tmp2,0);
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(angm2[myrk][2]!=angm3[myrk][flag])continue;
      if(vv.get_parr(l)->is_null())continue;
      tmp1=(*vv.get_parr(l));
      tmp1.mergeindex(0,1);
      step2.contract(step1,1,tmp1,0);
      for(m=0;m<nt4;m++){
	if(op1.get_tensor_argument(m,angm4[myrk],bd4[myrk],bc4[myrk])==false)continue;
	if(op1.get_parr(m)->is_null())continue;
	for(n=0;n<nt5;n++){
	  if(op2.get_tensor_argument(n,angm5[myrk],bd5[myrk],bc5[myrk])==false)continue;
	  if(op2.get_parr(n)->is_null())continue;
	  if(flag==0&&angm4[myrk][2]!=angm5[myrk][2]||flag==1&&angm4[myrk][0]!=angm5[myrk][0])continue;	  
	  if(flag==0&&angm4[myrk][1]!=angm2[myrk][1]||flag==1&&angm5[myrk][2]!=angm2[myrk][1])continue;
	  if(angm1[myrk][1-flag]!=angm5[myrk][flag]||angm3[myrk][1-flag]!=angm4[myrk][flag])continue;
	  tmp1=(*op1.get_parr(m));
	  tmp2=(*op2.get_parr(n));
	  fac1=tmp1.inner_prod(tmp2);
	  tmp=step2;
	  tmp*=fac1;
	  tmp.separateindex(0,bd1[myrk][2],1);
	  bdim[myrk][0]=angm1[myrk][2];
	  if(flag==0)
	    bdim[myrk][1]=angm5[myrk][1];
	  else if(flag==1)
	    bdim[myrk][1]=angm4[myrk][2];
	  bdim[myrk][2]=angm3[myrk][2];
	  if(cgc.check_angularmoments(bdim[myrk])==false)continue;
	  p=cgc.get_tensor_index(bdim[myrk]);
	  if(flag==0){
	    //t=c1/2+((c2-1)/2)*3+(d1/2)*9;
	    //fac=fac_permutation_left[a0][a2][b0][b2][t];
	    t=angm4[myrk][1]/2+((angm4[myrk][2]-1)/2)*3+(angm5[myrk][1]/2)*6;
	    fac=fac_permutation_left[angm1[myrk][0]][angm1[myrk][2]][angm3[myrk][0]][angm3[myrk][2]][t];
	  }
	  else if(flag==1){
	    //t=c2/2+((c0-1)/2)*3+(d2/2)*9;
	    //fac=fac_permutation_rght[a1][a2][b1][b2][t];
	    t=angm4[myrk][2]/2+((angm4[myrk][0]-1)/2)*3+(angm5[myrk][2]/2)*6;
	    fac=fac_permutation_rght[angm1[myrk][1]][angm1[myrk][2]][angm3[myrk][1]][angm3[myrk][2]][t];
	  }
	  if(fabs(fac)<1.e-8)continue;
	  tmp*=fac;
	  if(tarr_tmp[nten*myrk+p].is_null()){
	    tarr_tmp[nten*myrk+p]=tmp;
	    tcgc_tmp[nten*myrk+p].make_cgc(bdim[myrk][0],bdim[myrk][1],bdim[myrk][2]);
	  }
	  else
	    tarr_tmp[nten*myrk+p]+=tmp;
	}
      }
    }
  }
  
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!tarr_tmp[nten*j+i].is_null()){
	if(tarr[i].is_null()){
	  tarr[i]=tarr_tmp[nten*j+i];
	  tcgc[i]=tcgc_tmp[nten*j+i];
	}
	else
	  tarr[i]+=tarr_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bc4[0];
  delete []bc4;
  delete []bc5[0];
  delete []bc5;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bd4[0];
  delete []bd4;
  delete []bd5[0];
  delete []bd5;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []angm4[0];
  delete []angm4;
  delete []angm5[0];
  delete []angm5;
  delete []tarr_tmp;
  delete []tcgc_tmp;
}
