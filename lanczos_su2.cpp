#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "dmrg_su2_omp.hpp"
#include <complex>

using namespace std;

extern "C"{
  void dstev_(char*,int*,double*,double*,double*,int*,double*,int*);
  double ran_();
}

extern dmrg_su2* chain_ptr;
extern int comm_rank,nkrylov;
extern double tau;
//--------------------------------------------------------------------------------------
lanczos_su2::lanczos_su2(){
//--------------------------------------------------------------------------------------
}

//--------------------------------------------------------------------------------------
lanczos_su2::~lanczos_su2(){
//--------------------------------------------------------------------------------------
  delete []eig;
  delete []vec;
  delete []nnl;
  delete []aal;
  delete []ff;
}

//--------------------------------------------------------------------------------------
void lanczos_su2::initialize_lanczos(tensor_su2& vin,int ml,int ng){
//--------------------------------------------------------------------------------------
  mlanc=ml;
  neig=ng;
  aal=new double[mlanc];
  nnl=new double[mlanc+1];
  ff=new tensor_su2[mlanc+1];
  eig=new double[mlanc*2];
  vec=new double[mlanc*mlanc];
  ff[0]=vin;
  ff[0].normalize_vector();
}

//--------------------------------------------------------------------------------------
void lanczos_su2::lanczos2(int il, int ir, tensor_su2& vout,int nsite){
//--------------------------------------------------------------------------------------
  int i,j,m;
  double tol=1.e-6,nor;
  double q1,q2;
  tensor_su2 tmp,v2;
  ofstream fout;
  for(i=0;i<mlanc;i++)
    aal[i]=0;
  for(i=0;i<mlanc+1;i++)
    nnl[i]=0;
  diag_op(il,ir,ff[0],ff[1],nsite);
  aal[0]=ff[0].inner_prod(ff[1]);
  tmp=ff[0];
  tmp*=aal[0];
  ff[1]-=tmp;
  nnl[1]=sqrt(ff[1].inner_prod(ff[1]));
  ff[1]/=nnl[1];
  if(mlanc==2){
    if(nnl[1]<tol){
      //cout<<"nnl[1]="<<nnl[1]<<endl;
      diatridiag(mlanc-1);
      //for(i=0;i<mlanc;i++)cout<<"\teig["<<i<<"]="<<eig[i]<<endl;
      compute_evolution(vout,mlanc-1,nsite);	
      return;
    }
    diag_op(il,ir,ff[1],ff[2],nsite);
    aal[1]=ff[1].inner_prod(ff[2]);
    diatridiag(mlanc);
    for(i=0;i<mlanc;i++)cout<<"\teig["<<i<<"]="<<eig[i]<<endl;
    compute_evolution(vout,mlanc,nsite);	
    return;
  }
  for(m=2;m<mlanc;m++){
    diag_op(il,ir,ff[m-1],ff[m],nsite);
    aal[m-1]=ff[m-1].inner_prod(ff[m]);
    tmp=ff[m-1];
    tmp*=aal[m-1];
    ff[m]-=tmp;
    tmp=ff[m-2];
    tmp*=nnl[m-1];
    ff[m]-=tmp;

    for(j=0;j<m;j++){
      q1=ff[m].inner_prod(ff[j]);
      tmp=ff[j];
      tmp*=q1;
      ff[m]-=tmp;
    }

    nnl[m]=sqrt(ff[m].inner_prod(ff[m]));
    ff[m]/=nnl[m];
    if(nnl[m]<tol)cout<<"nnl["<<m<<"]="<<nnl[m]<<endl;
    diatridiag(m);
    compute_evolution(vout,m,nsite);
    vout.normalize_vector();
    if(nnl[m]<tol){
      //for(i=0;i<m;i++)cout<<"\teig["<<i<<"]="<<eig[i]<<endl;
      return;
    }    
    if(0&&m>2){
      nor=v2.inner_prod(vout);
      //cout<<"lanc m="<<m<<"\t"<<setprecision(16)<<1.-fabs(nor)<<endl;
      if(fabs(1.-fabs(nor))<1.e-15)
	return;
    }
    v2=vout;
  }
  diag_op(il,ir,ff[mlanc-1],ff[mlanc],nsite);
  aal[mlanc-1]=ff[mlanc-1].inner_prod(ff[mlanc]);
  diatridiag(mlanc);
  compute_evolution(vout,mlanc,nsite);
}

//--------------------------------------------------------------------------------------
void lanczos_su2::lanczos1(int il, int ir, tensor_su2& vout){
//--------------------------------------------------------------------------------------
  int i,j,m,nsite=2;
  double tol=1.e-6,nor,ee,ov;
  double q1,q2;
  tensor_su2 tmp,v2;
  ofstream fout;
  for(i=0;i<mlanc;i++)
    aal[i]=0;
  for(i=0;i<mlanc+1;i++)
    nnl[i]=0;
  diag_op(il,ir,ff[0],ff[1],nsite);
  aal[0]=ff[0].inner_prod(ff[1]);
  tmp=ff[0];
  tmp*=aal[0];
  ff[1]-=tmp;
  nnl[1]=sqrt(ff[1].inner_prod(ff[1]));
  ff[1]/=nnl[1];
  if(mlanc==2){
    if(nnl[1]<tol){
      diatridiag(mlanc-1);
      compute_eigenvector(vout,mlanc-1,nsite);	
      return;
    }
    diag_op(il,ir,ff[1],ff[2],nsite);
    aal[1]=ff[1].inner_prod(ff[2]);
    diatridiag(mlanc);
    compute_eigenvector(vout,mlanc,nsite);	
    return;
  }
  for(m=2;m<mlanc;m++){
    diag_op(il,ir,ff[m-1],ff[m],nsite);
    aal[m-1]=ff[m-1].inner_prod(ff[m]);
    tmp=ff[m-1];
    tmp*=aal[m-1];
    ff[m]-=tmp;
    tmp=ff[m-2];
    tmp*=nnl[m-1];
    ff[m]-=tmp;

    for(j=0;j<m;j++){
      q1=ff[m].inner_prod(ff[j]);
      tmp=ff[j];
      tmp*=q1;
      ff[m]-=tmp;
    }

    nnl[m]=sqrt(ff[m].inner_prod(ff[m]));
    ff[m]/=nnl[m];
    diatridiag(m);
    if(nnl[m]<tol){
      //for(i=0;i<m;i++)cout<<"\teig["<<i<<"]="<<eig[i]<<endl;
      compute_eigenvector(vout,m,nsite);
      check_eigenvector(il,ir,vout,ee,ov,nsite);
      cout<<"eigenvalue="<<ee<<"\toverlap="<<ov<<endl;
      return;
    }
    if(m%4==0){
      compute_eigenvector(vout,m,nsite);
      check_eigenvector(il,ir,vout,ee,ov,nsite);
      cout<<m<<"\teigenvalue="<<ee<<"\toverlap="<<ov<<endl;
      if(fabs(ov-1.)<1.e-8)return;
    }
  }
  diag_op(il,ir,ff[mlanc-1],ff[mlanc],nsite);
  aal[mlanc-1]=ff[mlanc-1].inner_prod(ff[mlanc]);
  diatridiag(mlanc);
  compute_eigenvector(vout,mlanc,nsite);
  check_eigenvector(il,ir,vout,ee,ov,nsite);
  cout<<mlanc-1<<"\teigenvalue="<<ee<<"\toverlap="<<ov<<endl;
}

//--------------------------------------------------------------------------------------
void lanczos_su2::compute_evolution(tensor_su2& vout, int mlanc_curr, int nsite){
//--------------------------------------------------------------------------------------
  ofstream fout;
  double *eigval,*lambda;
  tensor_su2 tmp;
  int i,j,k;
  eigval=new double[mlanc_curr];
  lambda=new double[mlanc_curr];
  for(i=0;i<mlanc_curr;i++){
    if(i<neig){
      if(nsite==2)
	lambda[i]=exp(-tau*eig[i]);
      else if(nsite==1)
	lambda[i]=exp(tau*eig[i]);
    }
    else
      lambda[i]=0;
  }
  for(i=0;i<mlanc_curr;i++){
    eigval[i]=0;
    for(j=0;j<mlanc_curr;j++)
      eigval[i]+=lambda[j]*vec[i+j*mlanc_curr]*vec[0+j*mlanc_curr];
  }

  for(i=0;i<mlanc_curr;i++)
    if(i==0){
      vout=ff[i];
      vout*=eigval[i];
    }
    else{
      tmp=ff[i];
      tmp*=eigval[i];
      vout+=tmp;
    }
  delete []eigval;
  delete []lambda;
}

//--------------------------------------------------------------------------------------
void lanczos_su2::diatridiag(int n){
//--------------------------------------------------------------------------------------
  double *d,*e,*work;
  int i,j,info;
  char jobz;
  jobz='V';
  d=new double[n];
  e=new double[n];
  work=new double[2*n];
  for(i=0;i<n;i++){
    d[i]=aal[i];
    if(i<n-1)
      e[i]=nnl[i+1];
  }
  dstev_(&jobz,&n,d,e,vec,&n,work,&info);
  for(i=0;i<n;i++){
    eig[i]=d[i];
    //cout<<"eig["<<i<<"]="<<eig[i]<<endl;
  }
  delete []d;
  delete []e;
  delete []work;
}

//--------------------------------------------------------------------------------------
void lanczos_su2::compute_eigenvector(tensor_su2& vout, int mlanc_curr, int nsite){
//--------------------------------------------------------------------------------------
  ofstream fout;
  tensor_su2 tmp;
  int i,j,k;
  for(k=0;k<1;k++){//only compute the gs vector
    for(j=0;j<mlanc_curr;j++)
      if(j==0){
	vout=ff[j];
	vout*=vec[j+k*mlanc_curr];
      }
      else{
	tmp=ff[j];
	tmp*=vec[j+k*mlanc_curr];
	vout+=tmp;
      }
  }
}

//--------------------------------------------------------------------------------------
void lanczos_su2::compute_eigenvector(int il, int ir, tensor_su2& vout, int mlanc_curr, int& stp, int nsite){
//--------------------------------------------------------------------------------------
  ofstream fout;
  double *eigval,*olap;
  int i,j,k;
  char name[100],rank[10];
  tensor_su2 tmp;
  eigval=new double[mlanc];
  olap=new double[mlanc];
  for(k=0;k<1;k++){//only compute the gs vector
    for(j=0;j<mlanc_curr;j++)
      if(j==0){
	vout=ff[j];
	vout*=vec[j+k*mlanc_curr];
      }
      else{
	tmp=ff[j];
	tmp*=vec[j+k*mlanc_curr];
	vout+=tmp;
      }
    check_eigenvector(il,ir,vout,eigval[k],olap[k],nsite);
    if(olap[k]>1-1.e-10){
      stp=1;
      strcpy(name,"out-");
      sprintf(rank,"%d",comm_rank);
      strcat(name,rank);
      strcat(name,".dat");
      fout.open(name,ios::app);
      fout<<il<<"\t"<<ir<<"\t"<<setprecision(12)<<"\t"<<mlanc_curr<<"\t"<<-eigval[k]<<"\t"<<olap[k]<<endl;
      fout.close();
    }
    else stp=0;
  }
  delete []eigval;
  delete []olap;
}

//--------------------------------------------------------------------------------------
void lanczos_su2::check_eigenvector(int il, int ir, tensor_su2& vout, double& eigval, double& olap,int nsite){
//--------------------------------------------------------------------------------------
  tensor_su2 vec;
  double prod,overlap,nor;
  nor=sqrt(vout.inner_prod(vout));
  vout/=nor;
  diag_op(il,ir,vout,vec,nsite);
  prod=sqrt(vec.inner_prod(vec));
  overlap=vec.inner_prod(vout);
  overlap/=prod;
  eigval=prod;
  olap=fabs(overlap);
}

//------------------------------------------------------------------------------
void lanczos_su2::diag_op(int i0,int i1, tensor_su2& v1, tensor_su2& v2,int nsite){
//------------------------------------------------------------------------------
  chain_ptr->hamiltonian_vector_multiplication_idmrg(i0,i1,v1,v2);
}
