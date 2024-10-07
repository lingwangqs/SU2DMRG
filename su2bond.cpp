#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "su2bond.hpp"

using namespace std;
extern "C"{
  void isort2_(int*,int*,int*);
  //int memcmp(char*,char*,int);
  void memcpy(char*,char*,int);
}

//--------------------------------------------------------------------------------------
su2bond::su2bond(){
//--------------------------------------------------------------------------------------
  angularmoment=NULL;
  bonddim=NULL;
  cgcdim=NULL;
  nmoment=0;
  bonddir=0;
}

//--------------------------------------------------------------------------------------
su2bond::su2bond(int nm,int dir,int *mom,int *dim){
//--------------------------------------------------------------------------------------
  int i;
  nmoment=nm;
  bonddir=dir;
  angularmoment=new int[nmoment];
  bonddim=new int[nmoment];
  cgcdim=new int[nmoment];
  for(i=0;i<nmoment;i++){
    angularmoment[i]=mom[i];
    bonddim[i]=dim[i];
    cgcdim[i]=mom[i]+1;
  }
}

//--------------------------------------------------------------------------------------
void su2bond::clean(){
//--------------------------------------------------------------------------------------
  if(angularmoment!=NULL)delete []angularmoment;
  if(bonddim!=NULL)delete []bonddim;
  if(cgcdim!=NULL)delete []cgcdim;
  angularmoment=NULL;
  bonddim=NULL;
  cgcdim=NULL;
  nmoment=0;
  bonddir=0;
}

//--------------------------------------------------------------------------------------
su2bond::~su2bond(){
//--------------------------------------------------------------------------------------
  clean();
}

//--------------------------------------------------------------------------------------
void su2bond::set_su2bond(int nm,int dir,int *mom,int *dim){
//--------------------------------------------------------------------------------------
  int i;
  clean();
  nmoment=nm;
  bonddir=dir;
  angularmoment=new int[nmoment];
  bonddim=new int[nmoment];
  cgcdim=new int[nmoment];
  isort2_(&nm,mom,dim);
  for(i=0;i<nmoment;i++){
    angularmoment[i]=mom[i];
    bonddim[i]=dim[i];
    cgcdim[i]=mom[i]+1;
  }
}

//--------------------------------------------------------------------------------------
su2bond& su2bond::operator = (const su2bond& b1){
//--------------------------------------------------------------------------------------
  int i;
  clean();
  nmoment=b1.get_nmoment();
  bonddir=b1.get_bonddir();
  angularmoment=new int[nmoment];
  bonddim=new int[nmoment];
  cgcdim=new int[nmoment];
  for(i=0;i<nmoment;i++){
    angularmoment[i]=b1.get_angularmoment(i);
    bonddim[i]=b1.get_bonddim(i);
    cgcdim[i]=b1.get_cgcdim(i);
  }
  return *this;
}

//--------------------------------------------------------------------------------------
bool su2bond::operator != (const su2bond& b1){
//--------------------------------------------------------------------------------------
  int i;
  if(nmoment!=b1.get_nmoment()||bonddir!=b1.get_bonddir())
    return true;
  for(i=0;i<nmoment;i++)
    if(angularmoment[i]!=b1.get_angularmoment(i)||bonddim[i]!=b1.get_bonddim(i))
      return true;
  return false;
}

//--------------------------------------------------------------------------------------
bool su2bond::operator == (const su2bond& b1){
//--------------------------------------------------------------------------------------
  if(*this!=b1)return false;
  else return true;
}

//--------------------------------------------------------------------------------------
void su2bond::print(){
//--------------------------------------------------------------------------------------
  int i;
  cout<<"su2bond nmoment="<<nmoment<<"\tbonddir="<<bonddir<<endl;
  for(i=0;i<nmoment;i++)
    cout<<angularmoment[i]<<"\t"<<bonddim[i]<<"\t"<<cgcdim[i]<<endl;    
}

//--------------------------------------------------------------------------------------
void su2bond::set_bonddir(int dir){
//--------------------------------------------------------------------------------------
  if(dir>0)bonddir=1;
  else if(dir<0)bonddir=-1;
  else{
    cout<<"something wrong with set_bonddir"<<endl;
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
bool su2bond::check_angularmoment(int angm){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nmoment;i++)
    if(angularmoment[i]==angm)return true;
  return false;
}

//--------------------------------------------------------------------------------------
int su2bond::get_angularmoment_index(int angm){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nmoment;i++){
    if(angularmoment[i]==angm)
      return i;
  }
  cout<<"get_angularmoment_index: couldn't find the martching angularmoment"<<endl;
  exit(0);
}

//--------------------------------------------------------------------------------------
void su2bond::fuse(su2bond& b1,su2bond& b2){
//--------------------------------------------------------------------------------------
  int i,j,k,l,m,max_num,flag,num,*ang,*dim,*angc,*dimc;
  int mom1,mom2,dim1,dim2,n1,n2,moms,mome;
  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"su2bond::fuse two bonds of different directions"<<endl;
    exit(0);
  }
  n1=b1.get_nmoment();
  n2=b2.get_nmoment();
  num=0;
  max_num=100;
  ang=new int[max_num];
  dim=new int[max_num];
  for(i=0;i<n1;i++){
    mom1=b1.get_angularmoment(i);
    dim1=b1.get_bonddim(i);
    for(j=0;j<n2;j++){
      mom2=b2.get_angularmoment(j);
      dim2=b2.get_bonddim(j);
      moms=abs(mom1-mom2);
      mome=abs(mom1+mom2);
      if(mome>38)mome=38;
      for(k=moms;k<=mome;k+=2){
	flag=1;
	for(l=0;l<num;l++){
	  if(ang[l]==k){
	    dim[l]+=dim1*dim2;
	    flag=0;
	    break;
	  }
	}
	if(flag){
	  ang[num]=k;
	  dim[num]=dim1*dim2;
	  num++;
	  if(num==max_num){
	    max_num*=2;
	    angc=new int[max_num];
	    dimc=new int[max_num];
	    memcpy((char*)angc,(char*)ang,num*sizeof(int));
	    memcpy((char*)dimc,(char*)dim,num*sizeof(int));
	    delete []ang;
	    delete []dim;
	    ang=angc;
	    dim=dimc;
	  }
	}
      }
    }
  }
  isort2_(&num,ang,dim);
  set_su2bond(num,-b1.get_bonddir(),ang,dim);
  delete []ang;
  delete []dim;
}

//--------------------------------------------------------------------------------------
void su2bond::fuse(su2bond& b1,su2bond& b2,int dir){
//--------------------------------------------------------------------------------------
  int i,j,k,l,m,max_num,flag,num,*ang,*dim,*angc,*dimc;
  int mom1,mom2,dim1,dim2,n1,n2,moms,mome;
  n1=b1.get_nmoment();
  n2=b2.get_nmoment();
  num=0;
  max_num=100;
  ang=new int[max_num];
  dim=new int[max_num];
  for(i=0;i<n1;i++){
    mom1=b1.get_angularmoment(i);
    dim1=b1.get_bonddim(i);
    for(j=0;j<n2;j++){
      mom2=b2.get_angularmoment(j);
      dim2=b2.get_bonddim(j);
      moms=abs(mom1-mom2);
      mome=mom1+mom2;
      if(mome>38)mome=38;
      for(k=moms;k<=mome;k+=2){
	flag=1;
	for(l=0;l<num;l++){
	  if(ang[l]==k){
	    dim[l]+=dim1*dim2;
	    flag=0;
	    break;
	  }
	}
	if(flag){
	  ang[num]=k;
	  dim[num]=dim1*dim2;
	  num++;
	  if(num==max_num){
	    max_num*=2;
	    angc=new int[max_num];
	    dimc=new int[max_num];
	    memcpy((char*)angc,(char*)ang,num*sizeof(int));
	    memcpy((char*)dimc,(char*)dim,num*sizeof(int));
	    delete []ang;
	    delete []dim;
	    ang=angc;
	    dim=dimc;
	  }
	}
      }
    }
  }
  isort2_(&num,ang,dim);
  set_su2bond(num,dir,ang,dim);
  delete []ang;
  delete []dim;
}

//--------------------------------------------------------------------------------------
void su2bond::fuse_to_multiplet(su2bond& b1,su2bond& b2,int tot_angm){
//--------------------------------------------------------------------------------------
  int i,j,k,l,m,max_num,flag,num,*ang,*dim,*angc,*dimc;
  int mom1,mom2,dim1,dim2,n1,n2,moms,mome;
  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"su2bond::fuse two bonds of different directions"<<endl;
    exit(0);
  }
  n1=b1.get_nmoment();
  n2=b2.get_nmoment();
  num=1;
  max_num=num;
  ang=new int[max_num];
  dim=new int[max_num];
  ang[0]=tot_angm;
  dim[0]=0;
  for(i=0;i<n1;i++){
    mom1=b1.get_angularmoment(i);
    dim1=b1.get_bonddim(i);
    for(j=0;j<n2;j++){
      mom2=b2.get_angularmoment(j);
      dim2=b2.get_bonddim(j);
      moms=abs(mom1-mom2);
      mome=mom1+mom2;
      if(mome>38)mome=38;
      for(k=moms;k<=mome;k+=2){
	if(k!=tot_angm)continue;
	dim[0]+=dim1*dim2;
      }
    }
  }
  set_su2bond(num,-b1.get_bonddir(),ang,dim);
  delete []ang;
  delete []dim;
}

//--------------------------------------------------------------------------------------
void su2bond::direct_sum(su2bond& b1,su2bond& b2){
//--------------------------------------------------------------------------------------
  int i,j,nmom,nmom1,nmom2,*aa,*bb,*cc,angm,bdim,flag;
  clean();
  if(b1.get_bonddir()!=b2.get_bonddir()){
    cout<<"su2bond::direct_sum can not apply direct sum"<<endl;
    exit(0);
  }
  bonddir=b1.get_bonddir();
  nmom1=b1.get_nmoment();
  nmom2=b2.get_nmoment();
  nmom=nmom1+nmom2;
  aa=new int[nmom];
  bb=new int[nmom];
  for(i=0;i<nmom1;i++){
    aa[i]=b1.get_angularmoment(i);
    bb[i]=b1.get_bonddim(i);
  }
  nmoment=nmom1;
  for(i=0;i<nmom2;i++){
    angm=b2.get_angularmoment(i);
    bdim=b2.get_bonddim(i);
    flag=0;
    for(j=0;j<nmoment;j++)
      if(angm==aa[j]){
	bb[j]+=bdim;
	flag=1;
	break;
      }
    if(flag==0){
      aa[nmoment]=angm;
      bb[nmoment]=bdim;
      nmoment++;
    }
  }
  isort2_(&nmoment,aa,bb);
  angularmoment=new int[nmoment];
  bonddim=new int[nmoment];
  cgcdim=new int[nmoment];
  for(i=0;i<nmoment;i++){
    angularmoment[i]=aa[i];
    bonddim[i]=bb[i];
    cgcdim[i]=angularmoment[i]+1;
  }
  delete []aa;
  delete []bb;
}
