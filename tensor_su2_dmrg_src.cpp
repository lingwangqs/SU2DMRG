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
extern int comm_rank,psize,physpn2;
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
void tensor_su2::operator_initial(tensor_su2& uu, tensor_su2& vv, tensor_su2& op, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,nr,nc,m1,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1,tmp1;
  tensor *tcgc_tmp;
  double *aa;
  double alpha=1,beta=0,ctmp;
  clean();
  uu.get_su2bond(1-flag,bb[0]);
  op.get_su2bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_su2bond(1-flag,bb[0]);
  op.get_su2bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  nbond=3;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  op.get_su2bond(1,bb[1]);
  vv.get_su2bond(2,bb[2]);
  bb[2].invert_bonddir();
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
  nb=3;

#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,nr,nc,m1,fac,ctmp,step1,tmp,tmp1,aa) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][1-flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(op.get_parr(j)->get_nelement()==1){
      op.get_parr(j)->get_telement(&ctmp);
      step1=(*uu.get_parr(k));
      step1*=ctmp;
      step1.shift(2,0);
      step1.separateindex(0,bd2[myrk][2],1);
    }
    else{
      step1.contract((*uu.get_parr(k)),1-flag,(*op.get_parr(j)),0);
      if(flag==0)step1.exchangeindex(1,2);
      else if(flag==1)step1.shift(1,0);
    }
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][1-flag])continue;
      if(angm2[myrk][flag]!=angm3[myrk][flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm1[myrk][1];
      bdim[myrk][2]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);
      bdim[myrk][0]=bd2[myrk][2];
      bdim[myrk][1]=bd1[myrk][1];
      bdim[myrk][2]=bd3[myrk][2];
      nr=bdim[myrk][0]*bdim[myrk][1];
      nc=bdim[myrk][2];
      m1=bd3[myrk][0]*bd3[myrk][1];
      tmp1=(*vv.get_parr(l));
      aa=new double[nr*nc];
      dgemm_("N","N",&nr,&nc,&m1,&alpha,step1.getptr(),&nr,tmp1.getptr(),&m1,&beta,aa,&nr);
      tmp.copy(nb,bdim[myrk],aa);
      delete []aa;
      if(flag==0)
	fac=fac_operator_onsite_left[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm1[myrk][0]+angm1[myrk][2]*physpn2];
      else if(flag==1)
	fac=fac_operator_onsite_rght[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm1[myrk][0]+angm1[myrk][2]*physpn2];

      if(fabs(fac)<1.e-8)
	continue;
      tmp*=fac;
      if(tarr_tmp[nten*myrk+m].is_null()){
	tarr_tmp[nten*myrk+m]=tmp;
	tcgc_tmp[nten*myrk+m].make_cgc(angm2[myrk][2],angm1[myrk][1],angm3[myrk][2]);
      }
      else
	tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []tarr_tmp;
  delete []tcgc_tmp;
}

//--------------------------------------------------------------------------------------
void tensor_su2::operator_initial(tensor_su2& uu, tensor_su2& vv, tensor_su2& op,tensor_su2& endt, int flag){
//--------------------------------------------------------------------------------------
  tensor_su2 tmp;
  if(flag==0)
    tmp.contract(endt,0,uu,0);
  else{
    tmp.contract(endt,0,uu,1);
    tmp.shift(2,0);
  }
  this->operator_initial(tmp,vv,op,flag);
}

//--------------------------------------------------------------------------------------
void tensor_su2::operator_transformation(tensor_su2& uu, tensor_su2& vv, tensor_su2& op, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1;
  tensor *tcgc_tmp;
  clean();
  uu.get_su2bond(flag,bb[0]);
  op.get_su2bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_su2bond(flag,bb[0]);
  op.get_su2bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  nbond=3;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  op.get_su2bond(1,bb[1]);
  vv.get_su2bond(2,bb[2]);
  bb[2].invert_bonddir();
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bd4=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  angm4=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bc4[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bd4[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  angm4[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bc4[i]=&(bc4[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bd4[i]=&(bd4[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
    angm4[i]=&(angm4[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
  nb=5;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,fac,step1,tmp) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    step1.contract((*uu.get_parr(k)),flag,(*op.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][flag])continue;
      if(angm2[myrk][1-flag]!=angm3[myrk][1-flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm1[myrk][1];
      bdim[myrk][2]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);

      bdim[myrk][0]=bd2[myrk][(flag+1)%3];
      bdim[myrk][1]=bd2[myrk][(flag+2)%3];
      bdim[myrk][2]=bd1[myrk][1];
      bdim[myrk][3]=bd3[myrk][(flag+1)%3];
      bdim[myrk][4]=bd3[myrk][(flag+2)%3];
      tmp.contract_dmrg_operator_transformation_step5(step1,(*vv.get_parr(l)),nb,bdim[myrk],flag);
      if(bd2[myrk][1-flag]==1&&bd3[myrk][1-flag]==1){
	tmp.mergeindex(3,4);
	tmp.mergeindex(0,1);
      }
      else{
	if(flag==0)tmp.contractindex(0,3);
	else if(flag==1)tmp.contractindex(1,4);
      }
      if(flag==0)
	fac=fac_operator_transformation_left[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm3[myrk][flag]][angm2[myrk][1-flag]];
      else if(flag==1)
	fac=fac_operator_transformation_rght[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm3[myrk][flag]][angm2[myrk][1-flag]];
      if(fabs(fac)<1.e-8)continue;
      tmp*=fac;
      if(tarr_tmp[nten*myrk+m].is_null()){
	tarr_tmp[nten*myrk+m]=tmp;
	tcgc_tmp[nten*myrk+m].make_cgc(angm2[myrk][2],angm1[myrk][1],angm3[myrk][2]);
      }
      else
	tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bd4[0];
  delete []bd4;
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
  delete []tarr_tmp;
  delete []tcgc_tmp;
}


//--------------------------------------------------------------------------------------
void tensor_su2::overlap_initial(tensor_su2& uu, tensor_su2& vv, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1;
  tensor *tcgc_tmp;
  clean();
  uu.get_su2bond(flag,bb[0]);
  vv.get_su2bond(flag,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  uu.get_su2bond(1-flag,bb[0]);
  vv.get_su2bond(1-flag,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  nbond=2;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  vv.get_su2bond(2,bb[1]);
  bb[1].invert_bonddir();
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
  }

  nt1=uu.get_nten();
  nt2=vv.get_nten();
  nt=nt1;
#pragma omp parallel for default(shared) private(myrk,i,j,k,m,tmp) schedule(dynamic,1)
  for(i=0;i<nt1;i++){
    myrk=omp_get_thread_num();
    j=i;
    if(uu.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_parr(j)->is_null())continue;
    for(k=0;k<nt2;k++){
      if(vv.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
      if(vv.get_parr(k)->is_null())continue;
      if(angm1[myrk][0]!=angm2[myrk][0])continue;
      if(angm1[myrk][1]!=angm2[myrk][1])continue;
      bdim[myrk][0]=angm1[myrk][2];
      bdim[myrk][1]=angm2[myrk][2];
      m=cgc.get_tensor_index(bdim[myrk]);

      tmp.contract_dmrg_overlap_initial((*uu.get_parr(j)),(*vv.get_parr(k)),flag);
      if(tarr_tmp[nten*myrk+m].is_null()){
	tarr_tmp[nten*myrk+m]=tmp;
	tcgc_tmp[nten*myrk+m]=identity[angm1[myrk][2]];
      }
      else
	tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []tarr_tmp;
  delete []tcgc_tmp;
}

//--------------------------------------------------------------------------------------
void tensor_su2::overlap_transformation(tensor_su2& uu, tensor_su2& vv, tensor_su2& op, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1;
  tensor *tcgc_tmp;
  clean();
  uu.get_su2bond(flag,bb[0]);
  op.get_su2bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"overlap transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_su2bond(flag,bb[0]);
  op.get_su2bond(1,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  nbond=2;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  vv.get_su2bond(2,bb[1]);
  bb[1].invert_bonddir();
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,step1,tmp) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(flag==0)
      step1.contract((*op.get_parr(j)),0,(*uu.get_parr(k)),flag);
    else{
      if(bd2[myrk][1-flag]==1){
	tmp=(*uu.get_parr(k));
	tmp.mergeindex(0,1);
	step1.contract((*op.get_parr(j)),0,tmp,0);
	step1.separateindex(0,1,bd1[myrk][1]);
      }
      else{
	step1.contract((*op.get_parr(j)),0,(*uu.get_parr(k)),flag);
	step1.shift(2,0);
      }
    }
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][1]!=angm3[myrk][flag])continue;
      if(angm2[myrk][1-flag]!=angm3[myrk][1-flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);

      tmp.contract_dmrg_overlap_initial(step1,(*vv.get_parr(l)),flag);
      if(tarr_tmp[nten*myrk+m].is_null()){
	tarr_tmp[nten*myrk+m]=tmp;
	tcgc_tmp[nten*myrk+m]=identity[angm2[myrk][2]];
      }
      else
	tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []tarr_tmp;
  delete []tcgc_tmp;
}

//--------------------------------------------------------------------------------------
void tensor_su2::operator_pairup(tensor_su2& uu, tensor_su2& vv, tensor_su2& op1, tensor_su2& op2, int flag){
//--------------------------------------------------------------------------------------
  su2bond bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1,step2,tmp1;
  tensor *tcgc_tmp;
  clean();
  uu.get_su2bond(flag,bb[0]);
  op1.get_su2bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_su2bond(flag,bb[0]);
  op1.get_su2bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  op1.get_su2bond(1,bb[0]);
  op2.get_su2bond(1,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }

  nbond=2;
  locspin=0;
  uu.get_su2bond(2,bb[0]);
  vv.get_su2bond(2,bb[1]);
  bb[1].invert_bonddir();
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bd4=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  angm4=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bc4[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bd4[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  angm4[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bc4[i]=&(bc4[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bd4[i]=&(bd4[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
    angm4[i]=&(angm4[0][i*nb]);
  }

  nt1=op1.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt4=op2.get_nten();
  nt=nt1*nt2;
  nb=2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,fac,step1,step2,tmp,tmp1) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op1.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op1.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(flag==0){
      tmp1=(*uu.get_parr(k));
      tmp1.exchangeindex(1,2);
      step1.contract(tmp1,0,(*op1.get_parr(j)),0);
    }
    else if(flag==1)
      step1.contract((*uu.get_parr(k)),flag,(*op1.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][flag])continue;
      tmp1=(*vv.get_parr(l));
      if(flag==1)
	tmp1.exchangeindex(0,1);
      step2.contract(step1,3,tmp1,0);
      step2.mergeindex(2,3);
      step2.mergeindex(1,2);
      step2.exchangeindex(0,1);
      for(n=0;n<nt4;n++){
	if(op2.get_tensor_argument(n,angm4[myrk],bd4[myrk],bc4[myrk])==false)continue;	
	if(angm4[myrk][0]!=angm2[myrk][1-flag]||angm4[myrk][2]!=angm3[myrk][1-flag]||angm4[myrk][1]!=angm1[myrk][1])continue;
	if(op2.get_parr(n)->is_null())continue;

	bdim[myrk][0]=angm2[myrk][2];
	bdim[myrk][1]=angm3[myrk][2];
	if(cgc.check_angularmoments(bdim[myrk])==false)continue;
	m=cgc.get_tensor_index(bdim[myrk]);
	tmp1=(*op2.get_parr(n));
	tmp1.mergeindex(1,2);
	tmp1.mergeindex(0,1);
	bdim[myrk][0]=bd2[myrk][2];
	bdim[myrk][1]=bd3[myrk][2];
	tmp.contract_dmrg_operator_transformation_step1(step2,tmp1,nb,bdim[myrk],flag);
	if(flag==0)
	  fac=fac_operator_pairup_left[angm2[myrk][2]][angm2[myrk][0]][angm3[myrk][0]][angm1[myrk][1]][angm4[myrk][0]+angm4[myrk][2]*physpn2];
	else if(flag==1)
	  fac=fac_operator_pairup_rght[angm2[myrk][2]][angm2[myrk][1]][angm3[myrk][1]][angm1[myrk][1]][angm4[myrk][0]+angm4[myrk][2]*physpn2];
	if(fabs(fac)<1.e-8)continue;
	tmp*=fac;
	if(tarr_tmp[nten*myrk+m].is_null()){
	  tarr_tmp[nten*myrk+m]=tmp;
	  tcgc_tmp[nten*myrk+m]=identity[angm2[myrk][2]];
	}
	else
	  tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bd4[0];
  delete []bd4;
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
  delete []tarr_tmp;
  delete []tcgc_tmp;
}

//--------------------------------------------------------------------------------------
void tensor_su2::hamiltonian_vector_multiplication(tensor_su2& vec, tensor_su2& op1, tensor_su2& op2){
//--------------------------------------------------------------------------------------
  su2bond bb[6];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  tensor tmp,*tarr_tmp,step1;
  tensor *tcgc_tmp;
  tensor_su2 tmpsu2;
  clean();
  if(op1.get_nbond()==2&&op2.get_nbond()==2){
    tmpsu2.contract(op1,0,vec,0);
    this->contract(tmpsu2,1,op2,0);
    return;
  }
  if(op1.get_nbond()==0||op2.get_nbond()==0)return;
  op1.get_su2bond(0,bb[0]);
  vec.get_su2bond(0,bb[1]);
  op2.get_su2bond(0,bb[2]);
  vec.get_su2bond(1,bb[3]);
  op1.get_su2bond(1,bb[4]);
  op2.get_su2bond(1,bb[5]);
  bb[1].invert_bonddir();
  bb[3].invert_bonddir();
  if(bb[0]!=bb[1]||bb[2]!=bb[3]||bb[4]!=bb[5]){
    cout<<"hamiltonian_vector_multiplication bb wrong"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  nbond=2;
  locspin=0;
  op1.get_su2bond(2,bb[0]);
  op2.get_su2bond(2,bb[1]);
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
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op1.get_nten();
  nt2=vec.get_nten();
  nt3=op2.get_nten();
  nt=nt1*nt2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,fac,tmp,step1) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op1.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(vec.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][0])continue;
    if(op1.get_parr(j)->is_null()||vec.get_parr(k)->is_null())continue;
    step1.contract((*vec.get_parr(k)),0,(*op1.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(op2.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(angm3[myrk][0]!=angm2[myrk][1]||angm1[myrk][1]!=angm3[myrk][1])continue;
      bdim[myrk][0]=angm1[myrk][2];
      bdim[myrk][1]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);
      if(op2.get_parr(l)->is_null())continue;
      fac=fac_hamilt_vec[angm1[myrk][0]][angm1[myrk][1]][angm1[myrk][2]];
      if(fabs(fac)<1.e-12)continue;
      tmp.contract_dmrg_hamiltonian_vector_multiplication(step1,(*op2.get_parr(l)),0);
      tmp*=fac;
      if(tarr_tmp[nten*myrk+m].is_null()){
	tarr_tmp[nten*myrk+m]=tmp;
	tcgc_tmp[nten*myrk+m]=cgc_coef_singlet[angm1[myrk][2]];
      }
      else
	tarr_tmp[nten*myrk+m]+=tmp;
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
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []tarr_tmp;
  delete []tcgc_tmp;
}

