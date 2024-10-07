//#include <mpi.h>
#include <omp.h>
#include <mkl.h>
#include "dmrg_su2_omp.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <sys/types.h>
#include <time.h>
#include <cstring>

using namespace std;
int max_dcut=8000,myrank=0,myrk,psize=12,mkl=1,comm_rank=0,comm_size,thread,bdry=2,jobid,memory_flag=0;
double delta=0,qdelta=1,alpha,tau;
double t1,t2,preparetime=0,svdtime=0,lanczostime=0;
dmrg_su2* chain_ptr;
tensor ***spin_op,***cgc_coef_left,***cgc_coef_rght,*cgc_coef_singlet,*identity;
double **spin_op_trace,*****fac_operator_onsite_left,*****fac_operator_onsite_rght,******fac_operator_transformation_left,******fac_operator_transformation_rght,*****fac_operator_pairup_left,*****fac_operator_pairup_rght,***fac_hamilt_vec,*****fac_permutation_left,*****fac_permutation_rght,*fac_betadouble,*fac_comp_left,*fac_comp_rght,*fac_init_left,*fac_init_rght;

double cosk,coskminus;
double pi=M_PI;
double pi2=pi*2;


extern "C"{
  double ran_();
  void initran_(int*);  
}

int main(int argc,char** argv){
  int i,sec,read,read2,d,dr,exci,niter,lx,ly,kx,ky,nstep;
  bool pass;
  double t3, t4,enr1,enr2;
  ofstream fout;
  char name[100],id[10],dk[10];

  t1=clock();
  i=(long)t1;
  initran_(&i);
  for(i=1;i<argc;i++){
    if(strcmp(argv[i],"-d")==0){
      max_dcut=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"max_dcut="<<max_dcut<<endl;
    }
    else if(strcmp(argv[i],"-dr")==0){
      read=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"read="<<read<<endl;
    }
    else if(strcmp(argv[i],"-dge")==0){
      read2=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"read2="<<read2<<endl;
    }
    else if(strcmp(argv[i],"-boundary")==0){
      bdry=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"boundary="<<bdry<<endl;
    }
    else if(strcmp(argv[i],"-lx")==0){
      lx=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"lx="<<lx<<endl;
    }
    else if(strcmp(argv[i],"-ly")==0){
      ly=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"ly="<<ly<<endl;
    }
    else if(strcmp(argv[i],"-kx")==0){
      kx=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"kx="<<kx<<endl;
    }
    else if(strcmp(argv[i],"-ky")==0){
      ky=atoi(argv[i+1]);
      cout<<"ky="<<ky<<endl;
    }
    else if(strcmp(argv[i],"-sec")==0){
      sec=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"sec="<<sec<<endl;
    }
    else if(strcmp(argv[i],"-exci")==0){
      exci=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"exci="<<exci<<endl;
    }
    else if(strcmp(argv[i],"-niter")==0){
      niter=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"niter="<<niter<<endl;
    }
    else if(strcmp(argv[i],"-jcoup")==0){
      delta=atof(argv[i+1]);
      if(comm_rank==0)
	cout<<"jcoup="<<delta<<endl;
    }
    else if(strcmp(argv[i],"-qcoup")==0){
      qdelta=atof(argv[i+1]);
      if(comm_rank==0)
	cout<<"qcoup="<<qdelta<<endl;
    }
    else if(strcmp(argv[i],"-invalpha")==0){
      alpha=atof(argv[i+1]);
      alpha=1/alpha;
      cout<<"alpha="<<alpha<<endl;
    }
    else if(strcmp(argv[i],"-jobid")==0){
      jobid=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"jobid="<<jobid<<endl;
    }
    else if(strcmp(argv[i],"-memory_flag")==0){
      memory_flag=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"memory_flag="<<memory_flag<<endl;
    }
    else if(strcmp(argv[i],"-psize")==0){
      psize=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"psize="<<psize<<endl;
    }
    else if(strcmp(argv[i],"-mkl")==0){
      mkl=atoi(argv[i+1]);
      if(comm_rank==0)
	cout<<"mkl="<<mkl<<endl;
    }
  }
  if(ly>1){
    cosk=cos(pi2*ky/ly);
    coskminus=cosk;
    cout<<"cosk="<<cosk<<"\tcoskminus="<<coskminus<<endl;
  }
  else if(0&&lx>1){
    cosk=cos(pi2*kx/lx);
    coskminus=cosk;
  }
  
  sprintf(dk,"%d",comm_rank+4);
  strcpy(name,"mkdir /data");
  strcat(name,dk);
  system(name);
  strcat(name,"/");
  sprintf(id,"%d",jobid);
  strcat(name,id);
  system(name);

  omp_set_num_threads(psize);
  mkl_set_num_threads(mkl);
  omp_set_nested(1);
  omp_set_max_active_levels(2);
  mkl_set_dynamic(1);

  dmrg_su2 chain(lx,ly,sec,read,read2,exci);
  chain_ptr=&chain;
  chain.set_ky(ky);
  if(read==0&&exci==0){
    chain.do_idmrg();
  }
  else{
    pass=chain.read_mps(read,read2);
    cout<<comm_rank<<" pass="<<pass<<endl;
    chain.read_enr(read,read2);
    chain.read_ww(read,read2);
    t3=omp_get_wtime();
    if(pass) chain.prepare_sweep();
    /*
    if(pass){
      chain.mea_translation();
      //chain.mea_enr();
      return 0;
    }
    */
    else chain.do_idmrg();

    t4=omp_get_wtime();
    if(comm_rank==0)cout<<"presweep time="<<t4-t3<<endl;
  }
  enr1=chain.get_enr();
  for(i=0;i<niter;i++){
    t3=omp_get_wtime();
    chain.sweep();
    t4=omp_get_wtime();
    if(comm_rank==0){
      fout.open("time.dat",ios::app);
      if(fout.is_open()){
	fout<<jobid<<"\tsweep="<<i<<"\tmemory_flag="<<memory_flag<<"\ttime="<<t4-t3<<endl;
	fout.close();
      }
    }
    enr2=chain.get_enr();
    cout<<setprecision(12)<<"iter="<<i<<"\tenr1="<<enr1<<"\tenr2="<<enr2<<endl;
    if(fabs((enr1-enr2))<1.e-6){
      cout<<"energy converged"<<endl;
      break;
    }
    enr1=enr2;
  }

  char base[100],len[10],dim[10],secchar[10],pos[10],exc[10],kk[10];
  sprintf(len,"%d",lx*ly);
  sprintf(secchar,"%d",sec);
  sprintf(dim,"%d",max_dcut);
  sprintf(exc,"%d",exci);
  sprintf(id,"%d",jobid);
  sprintf(dk,"%d",comm_rank+4);
  sprintf(kk,"%d",ky);

  strcpy(name,"rm -f /data");
  strcat(name,dk);
  strcat(name,"/");
  strcat(name,id);
  strcpy(base,name);
  
  for(i=0;i<7;i++){
    strcpy(name,base);
    if(i==0)
      strcat(name,"/uu-");
    else if(i==1)
      strcat(name,"/hh-");
    else if(i==2)
      strcat(name,"/tran-");
    else if(i==3)
      strcat(name,"/ovlp-");
    else if(i==4)
      strcat(name,"/orth-");
    else if(i==5)
      strcat(name,"/opr-");
    else if(i==6)
      strcat(name,"/plqopr-");
    strcat(name,len);
    strcat(name,"-");
    strcat(name,dim);
    strcat(name,"-");
    strcat(name,secchar);
    strcat(name,"-");
    strcat(name,kk);
    strcat(name,"-");
    strcat(name,exc);
    strcat(name,"-*");
    system(name);
    cout<<name<<endl;
  }
}
