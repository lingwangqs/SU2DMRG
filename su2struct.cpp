#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "su2struct.hpp"

using namespace std;
extern "C"{
  void isort2_(int*,int*,int*);
}

//--------------------------------------------------------------------------------------
su2struct::su2struct(){
//--------------------------------------------------------------------------------------
  bonddir=NULL;
  nmoment=NULL;
  angularmoment=NULL;
  cgcdim=NULL;
  bonddim=NULL;
  nbond=0;
  nten=0;
  locspin=0;
}

//--------------------------------------------------------------------------------------
su2struct::su2struct(int nb, int locsp, su2bond* barr){
//--------------------------------------------------------------------------------------
  int i,j;
  if(locsp<0||nb<1){
    cout<<"su2struct::su2struct input a wrong locspin/nbond, locspin"<<locsp<<"\tnbond="<<nb<<endl;
    exit(0);
  }
  nbond=nb;
  locspin=locsp;
  bonddir=new int[nbond];
  nmoment=new int[nbond];
  angularmoment=new int*[nbond];
  cgcdim=new int*[nbond];
  bonddim=new int*[nbond];
  nten=1;
  for(i=0;i<nbond;i++){
    bonddir[i]=barr[i].get_bonddir();
    nmoment[i]=barr[i].get_nmoment();
    nten*=nmoment[i];
    angularmoment[i]=new int[nmoment[i]];
    cgcdim[i]=new int[nmoment[i]];
    bonddim[i]=new int[nmoment[i]];
    for(j=0;j<nmoment[i];j++){
      angularmoment[i][j]=barr[i].get_angularmoment(j);
      bonddim[i][j]=barr[i].get_bonddim(j);
      cgcdim[i][j]=barr[i].get_cgcdim(j);
    }
  }
}

//--------------------------------------------------------------------------------------
su2struct::~su2struct(){
//--------------------------------------------------------------------------------------
  clean();
}

//--------------------------------------------------------------------------------------
void su2struct::clean(){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nbond;i++){
    if(angularmoment[i]!=NULL)delete []angularmoment[i];
    if(cgcdim[i]!=NULL)delete []cgcdim[i];
    if(bonddim[i]!=NULL)delete []bonddim[i];
  }
  if(angularmoment!=NULL)  delete []angularmoment;
  if(cgcdim!=NULL)  delete []cgcdim;
  if(bonddim!=NULL)  delete []bonddim;
  if(nmoment!=NULL)  delete []nmoment;
  if(bonddir!=NULL)  delete []bonddir;
  angularmoment=NULL;
  cgcdim=NULL;
  bonddim=NULL;
  nmoment=NULL;
  bonddir=NULL;
  nbond=0;
  nten=0;
  locspin=0;
}

//--------------------------------------------------------------------------------------
void su2struct::set_su2struct(int nb, int locsp, su2bond* barr){
//--------------------------------------------------------------------------------------
  int i,j;
  if(locsp<0||nb<1){
    cout<<"su2struct::set_su2struct input a wrong locspin/nbond, locspin"<<locsp<<"\tnbond="<<nb<<endl;
    exit(0);
  }
  nbond=nb;
  locspin=locsp;
  bonddir=new int[nbond];
  nmoment=new int[nbond];
  angularmoment=new int*[nbond];
  cgcdim=new int*[nbond];
  bonddim=new int*[nbond];
  nten=1;
  for(i=0;i<nbond;i++){
    bonddir[i]=barr[i].get_bonddir();
    nmoment[i]=barr[i].get_nmoment();
    nten*=nmoment[i];
    angularmoment[i]=new int[nmoment[i]];
    cgcdim[i]=new int[nmoment[i]];
    bonddim[i]=new int[nmoment[i]];
    for(j=0;j<nmoment[i];j++){
      angularmoment[i][j]=barr[i].get_angularmoment(j);
      bonddim[i][j]=barr[i].get_bonddim(j);
      cgcdim[i][j]=barr[i].get_cgcdim(j);
    }
  }
}

//--------------------------------------------------------------------------------------
void su2struct::print(){
//--------------------------------------------------------------------------------------
  int i,j;
  cout<<"su2struct:locspin="<<locspin<<endl;
  cout<<"su2struct:nbond="<<nbond<<endl;
  cout<<"su2struct:nten="<<nten<<endl;
  for(i=0;i<nbond;i++){
    cout<<"su2struct:bonddir["<<i<<"]="<<bonddir[i]<<"\tnmoment["<<i<<"]="<<nmoment[i]<<endl;
    for(j=0;j<nmoment[i];j++)
      cout<<angularmoment[i][j]<<"\t"<<bonddim[i][j]<<"\t"<<cgcdim[i][j]<<endl;
  }
  cout<<"============="<<endl;
}

//--------------------------------------------------------------------------------------
su2struct& su2struct::operator = (su2struct& cgc){
//--------------------------------------------------------------------------------------
  int i,j;
  clean();
  locspin=cgc.get_locspin();
  nbond=cgc.get_nbond();
  bonddir=new int[nbond];
  nmoment=new int[nbond];
  angularmoment=new int*[nbond];
  cgcdim=new int*[nbond];
  bonddim=new int*[nbond];
  nten=1;
  for(i=0;i<nbond;i++){
    bonddir[i]=cgc.get_bonddir(i);
    nmoment[i]=cgc.get_nmoment(i);
    nten*=nmoment[i];
    angularmoment[i]=new int[nmoment[i]];
    cgcdim[i]=new int[nmoment[i]];
    bonddim[i]=new int[nmoment[i]];
    for(j=0;j<nmoment[i];j++){
      angularmoment[i][j]=cgc.get_angularmoment(i,j);
      bonddim[i][j]=cgc.get_bonddim(i,j);
      cgcdim[i][j]=cgc.get_cgcdim(i,j);
    }
  }
  return *this;
}

//--------------------------------------------------------------------------------------
bool su2struct::operator != (su2struct& cgc){
//--------------------------------------------------------------------------------------
  if(nbond!=cgc.get_nbond()||locspin!=cgc.get_locspin()||nten!=cgc.get_nten())return true;
  int i,j;
  for(i=0;i<nbond;i++){
    if(bonddir[i]!=cgc.get_bonddir(i)||nmoment[i]!=cgc.get_nmoment(i))return true;
    for(j=0;j<nmoment[i];j++){
      if(angularmoment[i][j]!=cgc.get_angularmoment(i,j)||bonddim[i][j]!=cgc.get_bonddim(i,j))return true;
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------
bool su2struct::operator == (su2struct& cgc){
//--------------------------------------------------------------------------------------
  if(*this!=cgc)return false;
  else return true;
}

//--------------------------------------------------------------------------------------
void su2struct::take_conjugate(){
//--------------------------------------------------------------------------------------
  int i;
  for(i=0;i<nbond;i++){
    bonddir[i]=-bonddir[i];
  }
}

//--------------------------------------------------------------------------------------
void su2struct::take_conjugate(int ind){
//--------------------------------------------------------------------------------------
  if(ind>=nbond){
    cout<<"su2struct::take_conjugate bond, bond index wrong\t"<<ind<<endl;
    exit(0);
  }
  bonddir[ind]=-bonddir[ind];
}

//--------------------------------------------------------------------------------------
void su2struct::invert_bonddir(int ind){
//--------------------------------------------------------------------------------------
  bonddir[ind]=-bonddir[ind];  
}

//--------------------------------------------------------------------------------------
void su2struct::get_su2bond(int i0, su2bond& bb){
//--------------------------------------------------------------------------------------
  int i,*angm,*bdim,nmom,bdir;
  if(i0>=nbond){
    cout<<"su2struct::get_su2bond, can not get_su2bond i0="<<i0<<"\tnbond="<<nbond<<endl;
    exit(0);
  }
  nmom=nmoment[i0];
  bdir=bonddir[i0];
  angm=new int[nmom];
  bdim=new int[nmom];
  for(i=0;i<nmom;i++){
    angm[i]=angularmoment[i0][i];
    bdim[i]=bonddim[i0][i];
  }
  bb.set_su2bond(nmom,bdir,angm,bdim);
  delete []angm;
  delete []bdim;
}

//--------------------------------------------------------------------------------------
int su2struct::get_angularmoment_index(int i0, int angm){
//--------------------------------------------------------------------------------------
  int i;
  if(i0<nbond){
    for(i=0;i<nmoment[i0];i++)
      if(angularmoment[i0][i]==angm)return i;
    return -1;
  }
  else{
    cout<<"su2struct::get_angularmoment_index: bond index i0 too large "<<i0<<endl;
    exit(0);
  }
}

//--------------------------------------------------------------------------------------
int su2struct::get_tensor_index(int* angm){
//--------------------------------------------------------------------------------------
  int i0,i,*aa;
  aa=new int[nbond];
  for(i=0;i<nbond;i++){
    aa[i]=get_angularmoment_index(i,angm[i]);
    if(aa[i]==-1){
      delete []aa;
      return nten;
    }
  }
  i0=0;
  for(i=nbond-1;i>=0;i--){
    i0*=nmoment[i];
    i0+=aa[i];
  }
  delete []aa;
  return i0;
}

//--------------------------------------------------------------------------------------
bool su2struct::get_tensor_argument(int i0, int* angm, int* bdim, int* cdim){
//--------------------------------------------------------------------------------------
  int i,j;
  bool check;
  if(i0>=nten){
    cout<<"su2struct::get_tensor_argument, something wrong with get_tensor_argument i0>=nten="<<nten<<endl;
    exit(0);
  }
  for(i=0;i<nbond;i++){
    j=i0%nmoment[i];
    i0/=nmoment[i];
    bdim[i]=bonddim[i][j];
    cdim[i]=cgcdim[i][j];
    angm[i]=angularmoment[i][j];
  }
  if(nbond==1&&angm[0]!=locspin)return false;
  else if(nbond==1&&angm[0]==locspin)return true;
  su2bond *bb;
  int *mom,*dim;
  bb=new su2bond[nbond+1];
  mom=new int[1];
  dim=new int[1];
  for(i=0;i<nbond;i++){
    dim[0]=1;
    mom[0]=angm[i];
    bb[i].set_su2bond(1,1,mom,dim);
  }
  for(i=1;i<nbond;i++){
    bb[nbond].fuse(bb[i-1],bb[i]);
    bb[nbond].set_bonddir(1);
    bb[i]=bb[nbond];
  }
  check=bb[nbond].check_angularmoment(locspin);
  delete []dim;
  delete []mom;
  delete []bb;
  if(check) return true;
  else return false;
}

//--------------------------------------------------------------------------------------
bool su2struct::check_angularmoments(int* angm){
//--------------------------------------------------------------------------------------
  su2bond *bb;
  int *mom,*dim,i;
  bool check;
  if(nbond==1)
    if(locspin==angm[0])return true;
    else return false;
  bb=new su2bond[nbond+1];
  mom=new int[1];
  dim=new int[1];
  for(i=0;i<nbond;i++){
    dim[0]=1;
    mom[0]=angm[i];
    bb[i].set_su2bond(1,1,mom,dim);
  }
  for(i=1;i<nbond;i++){
    bb[nbond].fuse(bb[i-1],bb[i]);
    bb[nbond].set_bonddir(1);
    bb[i]=bb[nbond];
  }
  check=bb[nbond].check_angularmoment(locspin);
  delete []dim;
  delete []mom;
  delete []bb;
  if(check) return true;
  else return false;
}

//--------------------------------------------------------------------------------------
void su2struct::shift(int i0, int i1){
//--------------------------------------------------------------------------------------
  int ishift,i,nb,totspin;
  su2bond *barr;
  if(i0==i1)
    return;
  if(i1>i0)ishift=i1-i0;
  else ishift=nbond-(i0-i1);
  barr=new su2bond[nbond];
  for(i=0;i<nbond;i++)
    get_su2bond(i,barr[(i+ishift)%nbond]);
  nb=nbond;
  totspin=locspin;
  set_su2struct(nb,totspin,barr);
  delete []barr;
}

//--------------------------------------------------------------------------------------
void su2struct::exchangeindex(int i0, int i1){
//--------------------------------------------------------------------------------------
  int i,nb,totspin;
  su2bond *barr;
  if(i0==i1)
    return;
  barr=new su2bond[nbond];
  for(i=0;i<nbond;i++)
    if(i!=i0&&i!=i1)
      get_su2bond(i,barr[i]);
  get_su2bond(i0,barr[i1]);
  get_su2bond(i1,barr[i0]);
  nb=nbond;
  totspin=locspin;
  set_su2struct(nb,totspin,barr);
  delete []barr;
}

//--------------------------------------------------------------------------------------
void su2struct::get_nelement(int &n1, int &n2){
//--------------------------------------------------------------------------------------
  int i,j,k1,k2,*angm,*bdim,*cdim;
  bool check;
  n1=0;
  n2=0;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    check=get_tensor_argument(i,angm,bdim,cdim);
    if(check){
      k1=1;
      k2=1;
      for(j=0;j<nbond;j++){
	k1*=bdim[j];
	k2*=cdim[j];
      }
      n1+=k1;
      n2+=k2;
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void su2struct::get_nelement(int &n1, int &n2, int* iflag){
//--------------------------------------------------------------------------------------
  int i,j,k1,k2,*angm,*bdim,*cdim;
  bool check;
  n1=0;
  n2=0;
  angm=new int[nbond];
  bdim=new int[nbond];
  cdim=new int[nbond];
  for(i=0;i<nten;i++){
    if(iflag[i]){
      check=get_tensor_argument(i,angm,bdim,cdim);
      k1=1;
      k2=1;
      for(j=0;j<nbond;j++){
	k1*=bdim[j];
	k2*=cdim[j];
      }
      n1+=k1;
      n2+=k2;
    }
  }
  delete []angm;
  delete []bdim;
  delete []cdim;
}

//--------------------------------------------------------------------------------------
void su2struct::direct_sum(int ind, su2struct& cgc1, su2struct& cgc2){
//--------------------------------------------------------------------------------------
  su2bond bb1,bb2,*bb;
  int nbond1,nbond2,i,locspin1,locspin2;
  clean();
  nbond1=cgc1.get_nbond();
  nbond2=cgc2.get_nbond();
  locspin1=cgc1.get_locspin();
  locspin2=cgc2.get_locspin();
  if(nbond1!=nbond2||locspin1!=locspin2){
    cout<<"su2struct::direct_sum can not sum two su2structure 1"<<endl;
    exit(0);
  }
  nbond=nbond1;
  locspin=locspin1;
  bb=new su2bond[nbond];
  for(i=0;i<nbond;i++){
    cgc1.get_su2bond(i,bb1);
    cgc2.get_su2bond(i,bb2);
    if(i!=ind&&bb1==bb2) bb[i]=bb1;
    else if(i==ind) bb[i].direct_sum(bb1,bb2);
    else if(i!=ind&&bb1!=bb2){
      cout<<"su2struct::direct_sum can not sum two su2structure 2"<<endl;
      delete []bb;
      exit(0);
    }
  }
  set_su2struct(nbond,locspin,bb);
  delete []bb;
}
