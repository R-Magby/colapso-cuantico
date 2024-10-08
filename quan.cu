#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#include <cufft.h>
#include <omp.h>
#define order 3
#define consta_pi 3.1415
#define consta_var 12
double host_coefficient_adelantada[order];
double host_coefficient_centrada[order];
double host_coefficient_atrasada[order];
double  host_b_i[4] = {1.0/6.0 , 1.0/3.0 , 1.0/3.0 , 1.0/6.0};
double host_c_i[4] = {0.0 , 0.5 , 0.5 , 1.0};
double host_a_ij[4] = {0.0 , 0.5 , 0.5 , 1.0};

__device__ double coefficient_adelantada[3] = {-1.5 ,2.0 ,-0.5};
__device__ double coefficient_centrada[3] = {-0.5, 0.0, 0.5};
__device__ double coefficient_atrasada[3] = {1.5, -2.0, 0.5};
__device__ double b_i[4] = {1.0/6.0 , 1.0/3.0 , 1.0/3.0 , 1.0/6.0};
__device__ double c_i[4] = {0.0 , 0.5 , 0.5 , 1.0};
__device__ double a_ij[4] = {0.0 , 0.5 , 0.5 , 1.0};

void guardar_salida_phi(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_2.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_alpha(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_lapso.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);}
void guardar_salida_Kb(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_Kb.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);}
void guardar_salida_chi(double *data,int Nr, int T) {
  FILE *fp = fopen("dr_campo_escalar.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void cargar_coeff_centradas_df(double *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_centrada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(double) , N , arch );
  fclose(arch);

}
void cargar_coeff_atrasada_df(double *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_atrasada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(double) , N , arch );
  fclose(arch);

}
void cargar_coeff_adelantada_df(double *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_adelantada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(double) , N , arch );
  fclose(arch);

}

void difference_tenth(double *df, double *f, double h, int N ){
  double temp;

  for(int idx=0; idx<N;idx++){
    temp=0.0;

    if (idx<(int)(order-1)/2 ){
      for (int m=0;m<order;m++){
        temp += host_coefficient_adelantada[m]*f[idx+m];
      }
      df[idx]=temp/h; 
    }
    else if (idx > N-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += host_coefficient_atrasada[-m]*f[idx+m];
      }
      df[idx]=temp/h;
    }
    else{
      for (int m=0;m<order;m++){
        temp += host_coefficient_centrada[m]*f[idx-(int)(order-1)/2+m];
      }
      df[idx]=temp/h;
    }
  }
}


double absorbente(double *g,int idx,double dr,double dt,int Nr){
    double dgdt, c;
    c=1.0;
    dgdt=-1.0*c*(3.0*g[idx]-4.0*g[idx-1] + g[idx-2])/(2.0*dr);

    return g[idx] + dgdt*dt;
}

double Kreiss_Oliger(double *G,int idx,int Nr,double epsilon){

    if(idx==0){
      return epsilon*(G[2]-2.0*G[1]+G[0]);
    }
    else if(idx==1){
      return epsilon*(G[3]-3.0*G[2]+3.0*G[1]-G[0]);
    }
    else if(idx==Nr-1){
      return epsilon*(G[Nr-3]-2.0*G[Nr-2]+G[Nr-1]);
    }
    else if(idx==Nr-2){
      return epsilon*(-G[Nr-4]+3.0*G[Nr-3]-3.0*G[Nr-2]+G[Nr-1]);
    }
    else{
      return epsilon*(G[idx+2] - 4.0*G[idx+1] + 6.0*G[idx] - 4.0*G[idx-1]+G[idx-2]);
    }
}





void calculate_rho(double *rho,double *PI,double *chi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){
    rho[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);}

}
void calculate_ja(double *ja, double *PI, double *chi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){

    ja[idx] = -PI[idx]*chi[idx] / (sqrt(A[idx])*B[idx]);}

}
void calculate_SA(double *SA, double *PI, double *chi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){
    SA[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);}

}
void calculate_SB(double *SB, double *PI, double *chi, double *A, double *B, int Nr ){
  for(int idx=0;idx<Nr;idx++){

  SB[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) - chi[idx]*chi[idx])/(2.0*A[idx]);}

}
double conservacion(double *rho, double dr, int Nr){
    float kn,sum_par,sum_impar;
    sum_par=0.0;
    sum_impar=0.0;

    for(int i=2; i<Nr-1; i=2+i){
        sum_par+=rho[i]*i*i*dr*dr;
    }
    for(int i=1; i<Nr-1; i=2+i){
        sum_impar+=rho[i]*i*i*dr*dr;
    }
    return (rho[0]*0.0 +4.0*sum_impar + 2.0*sum_par +  rho[Nr-1]*(Nr-1)*(Nr-1)*dr*dr)*(4.0*3.1415*dr/3.0);
            



}

__device__ double derivate_RK( double *f,double *Kf, int idx,int s, int id_RK, double dr,double dt, int Nr, int symmetric){
  double temp ;
  temp=0.0;
    if (idx==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[id_RK+m]);
        }
      }
    }
    /*else if (idx<(int)(order-1)/2 && idx>0){
      for (int m=0;m<order;m++){
        temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }*/
    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f[idx+m] + dt*a_ij[s]*Kf[id_RK+m]);
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m] + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m]);
      }
    }
    return temp/dr;
}
__device__ cufftDoubleComplex derivate_complex_RK( cufftDoubleComplex *f,cufftDoubleComplex *Kf, int idx,int s, int id_RK, double dr,double dt, int Nr, int symmetric){
  cufftDoubleComplex temp ;
  temp.x = 0.0;
  temp.y = 0.0;
    if (idx==0){
      if (symmetric==0){
        temp.x = 0.0;
        temp.y = 0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp.x += coefficient_adelantada[m]*(f[idx+m].x + dt*a_ij[s]*Kf[id_RK+m].x);
          temp.y += coefficient_adelantada[m]*(f[idx+m].y + dt*a_ij[s]*Kf[id_RK+m].y);

        }
      }
    }
    /*else if (idx<(int)(order-1)/2 && idx>0){
      for (int m=0;m<order;m++){
        temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }*/
    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp.x += coefficient_atrasada[-m]*(f[idx+m].x + dt*a_ij[s]*Kf[id_RK+m].x);
        temp.y += coefficient_atrasada[-m]*(f[idx+m].y + dt*a_ij[s]*Kf[id_RK+m].y);

      }
    }
    else{
      for (int m=0;m<order;m++){
        temp.x += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].x + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m].x);
        temp.y += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].y + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m].y);

      }
    }
    temp.x /=dr;
    temp.y /=dr;



    return temp;
}
__device__ cufftDoubleComplex derivate_complex( cufftDoubleComplex *f, int idx,  double dr, int Nr, int symmetric){
  cufftDoubleComplex temp ;
  temp.x = 0.0;
  temp.y = 0.0;
    if (idx==0){
      if (symmetric==0){
        temp.x = 0.0;
        temp.y = 0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp.x += coefficient_adelantada[m]*(f[idx+m].x);
          temp.y += coefficient_adelantada[m]*(f[idx+m].y );

        }
      }
    }
    /*else if (idx<(int)(order-1)/2 && idx>0){
      for (int m=0;m<order;m++){
        temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }*/
    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp.x += coefficient_atrasada[-m]*(f[idx+m].x );
        temp.y += coefficient_atrasada[-m]*(f[idx+m].y );

      }
    }
    else{
      for (int m=0;m<order;m++){
        temp.x += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].x );
        temp.y += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].y );

      }
    }
    temp.x /=dr;
    temp.y /=dr;




  

    return temp;
}

__device__ double Kb_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda,
                         double *U,  double *RK_C , double *Sa, double *rho, double radio, int id, int s, double dt, double dr, int Nr, int t, int error){
  double f1,f2,f3;

  f1 = 0.5*(U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 2.0* (lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/
          (A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) - (Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ]) - (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ]) - (lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ]);
  
  f2 = -0.5*(Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ]) * (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])  - 0.5*derivate_RK(Db,RK_C, id, s,7,dr,dt,Nr,1) +0.25*(Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])*
        ((U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 4.0*(lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])) +
        (A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);

  f3 = Sa[id] - rho[id];



  if (error==0 && t==0 && id == 50){
    printf("Kb_dot == check\n");
  }



  return  (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) /(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) *(f1/radio+f2) + (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) / 2.0 * f3;
}

__device__ double K_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda, 
                        double *U, double *RK_C , double *Sa, double *rho, double *Sb, double radio, int id, int s, double dt,double dr, int Nr,  int t, int error){

  double f1,f2,f3;

  f1 = (K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 4.0*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])* (Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) +
        6.0*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ])*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);

  f2 =  derivate_RK(Da,RK_C, id, s,6*Nr + id,dr,dt,Nr,1) + (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])+ 2.0*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])/radio
        -0.5*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*((U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 4.0*(lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]));

  f3 =  rho[id] +Sa[id] + 2.0*Sb[id];



  if (error==0 && t==0 && id == 50){
    printf("K_dot == check\n");
  }


  return  (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * 
          ( f1 - f2/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) ) + 
          (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) / 2.0 * f3;
}

__device__ double lambda_dot(double *A, double *B, double *alpha, double *Db, double * K, double *Kb,  double *RK_C, double *Ja, int id, int s, double dt,double dr, int Nr,  int t, int error){

  double f1;
  
  f1  =   derivate_RK(Kb,RK_C, id, s, 9*Nr + id ,dr,dt,Nr,1) - 0.5* ( Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])*(( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 3.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) ) +
          Ja[id]/2.0;




          if (error==0 && t==0 && id == 50){
            printf("lambda_dot == check\n");
          }
        
  return 2.0*(alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * ( A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])/( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ]) * f1;

}

__device__ double U_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda, 
                        double *RK_C, double *Ja, int id, int s, double dt,double dr, int Nr,  int t, int error){
  double f1, f2;

  f1  =   derivate_RK(K,RK_C, id, s, 8*Nr + id ,dr,dt,Nr,1) -  ( Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*(( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 4.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) );

  f2  =   -2.0*( ( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 3.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) ) *(  ( Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])  - 
           2.0*  ( lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ]) * ( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/ ( A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) );
  


    if (error==0 && t==0 && id == 50){
      printf("U_dot == check\n");
    }
        

  return 2.0*(alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * ( f1 + f1 - 2.0*Ja[id] );

}

__device__ double f_Db_dot(double *Kb, double *alpha,  double *RK_C, int id, int s, double dt,int Nr,  int t, int error){



  if (error==0 && t==0 && id == 50){
    printf("f_Db_dot == check\n");
  }


  return (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) *( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);
}
__device__ double Db_dot(double *Kb, double *alpha, int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp;
  temp=0.0;

    if (idx==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f_Db_dot(Kb,alpha,RK_C,idx + m ,s,dt ,Nr ,   t, error));
        }
      }
    }

    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_Db_dot(Kb,alpha,RK_C,idx + m ,s,dt ,Nr , t,  error));
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_Db_dot(Kb,alpha,RK_C,idx-(int)(order-1)/2+m,s,dt ,Nr , t ,  error));
      }
    }


    if (error==0 && t==0 && idx == 50){
      printf("Db_dot == check\n");
    }
  

    return temp/dr;
}
__device__ double f_PI_dot(double *chi, double *PI, double *A, double *B,double *alpha, double *RK_C, int id, int s, double dt,double dr,  int Nr,  int t, int error){
      double radio;
      double epsilon=dr/2.0;
      if ( id==0 ){
        radio = id*dr +epsilon;
      }
      else{
        radio = id*dr ;
      }


      if (error==0 && t==0 && id == 50){
        printf("f_PI_dot == check\n");
      }
    

  return (chi[id] + dt * a_ij[s] * RK_C[ 1*Nr + id ]) * (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) *( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])*
          radio*radio/(sqrt(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]));
}

__device__ double PI_dot(double *chi, double *PI, double *A, double *B,double *alpha, double r2, int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp;
  temp=0.0;
    if (idx==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f_PI_dot(chi, PI,A,B,alpha,RK_C,idx + m ,s,dt,dr ,Nr , t , error));
        }
      }
    }

    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_PI_dot(chi, PI,A,B,alpha,RK_C,idx+m,s,dt,dr ,Nr , t , error));
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_PI_dot(chi, PI,A,B,alpha,RK_C,idx-(int)(order-1)/2+m,s,dt,dr ,Nr , t , error));
      }
    }


    if (error==0 && t==0 && idx == 50){
      printf("PI_dot == check\n");
    }
  
    return temp/dr*(1.0/(r2));;
}


__device__ double f_chi_dot(double *PI, double *A, double *B,double *alpha, double *RK_C, int id, int s, double dt, int Nr,  int t, int error){


  if (error==0 && t==0 && id == 50){
    printf("f_chi_dot == check\n");
  }


  return (PI[id] + dt * a_ij[s] * RK_C[ 2*Nr + id ]) * (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ])/
          (sqrt(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])*( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ]));
}

__device__ double chi_dot(double *PI, double *A, double *B,double *alpha,int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr, int t, int error){
  double temp;
  temp=0.0;

    if (idx==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f_chi_dot(PI,A,B,alpha,RK_C,idx + m ,s,dt ,Nr, t ,  error));
        }
      }
    }

    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_chi_dot(PI,A,B,alpha,RK_C,idx+m,s,dt, Nr , t , error));
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_chi_dot(PI,A,B,alpha,RK_C,idx-(int)(order-1)/2+m,s,dt,Nr, t , error ));
      }
    }



    if (error==0 && t==0 && idx == 50){
      printf("chi_dot == check\n");
    }
  
    return temp/dr;
}

__device__ cufftDoubleComplex f_dr_u_dot(cufftDoubleComplex *pi, double *A, double *B,double *alpha, double *RK_C, cufftDoubleComplex *RK_Q, int id_nodo, int idr , int space, int s, double dt ,int Nr, int t, int error){

  cufftDoubleComplex Z;

  Z.x= (pi[id_nodo].x + dt * a_ij[s] * RK_Q[ 2*space +  id_nodo ].x) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));
  Z.y= (pi[id_nodo].y + dt * a_ij[s] * RK_Q[ 2*space + id_nodo ].y) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));


  if (error==0 && t==0 && id_nodo == 50){
    printf("f_dr_u_dot == check\n");
  }


  return Z;
}

__device__ cufftDoubleComplex dr_u_dot(cufftDoubleComplex *pi, double *A, double *B,double *alpha,int id_nodo, int idr,int space, int s, double dr, double dt, double *RK_C, cufftDoubleComplex *RK_Q, 
                                      int symmetric,int Nr,  int t, int error){
  cufftDoubleComplex temp;
  temp.x=0.0;
  temp.y=0.0;

    if (idr==0){
      if (symmetric==0){
        temp.x=0.0;
        temp.y=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp.x += coefficient_adelantada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m  ,idr+ m, space, s,dt, Nr , t , error)).x;
          temp.y += coefficient_adelantada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m ,idr+ m, space, s,dt, Nr, t , error )).y;

        }
      }
    }

    else if (idr > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp.x += coefficient_atrasada[-m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m , idr + m, space , s,dt, Nr, t , error)).x;
        temp.y += coefficient_atrasada[-m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m , idr + m, space , s,dt, Nr, t , error)).y;

      }
    }
    else{
      for (int m=0;m<order;m++){
        temp.x += coefficient_centrada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo-(int)(order-1)/2+m , idr-(int)(order-1)/2+m , space , s,dt, Nr , t , error)).x;
        temp.y += coefficient_centrada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo-(int)(order-1)/2+m , idr-(int)(order-1)/2+m , space , s,dt, Nr , t , error)).y;

      }
    }
 


  
    return temp;
}
__device__ double f_metric_dot(double *A, double *B,double *alpha, double *RK_C, int idr , int s, double dt, int Nr,  int t,int error){

  return  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ]));
}

__device__ double metric_pi_dot( double *A, double *B,double *alpha, int idr, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp;
  temp=0.0;

    if (idr==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f_metric_dot(A,B,alpha,RK_C  ,idr+ m,s,dt , Nr, t , error));
        }
      }
    }

    else if (idr > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_metric_dot(A,B,alpha,RK_C, idr + m,s,dt, Nr, t , error));
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_metric_dot(A,B,alpha,RK_C,  idr-(int)(order-1)/2+m,s,dt , Nr, t , error));
      }
    }




  
    return temp/dr;
}
__device__ cufftDoubleComplex f_pi_dot(cufftDoubleComplex *u, cufftDoubleComplex *dr_u,double *A, double *B,double *alpha, double *lambda, double *RK_C, cufftDoubleComplex *RK_Q, 
                                      double radio,  int id_nodo, int idr , int space, int s, int k, int l, double dt,double dr, int Nr,  int t, int error){

  double f1, f2, f3;
  cufftDoubleComplex Z;


  f1 =  l/radio * (u[id_nodo].x + dt*a_ij[s] * RK_Q[ 0*space + id_nodo].x) + (dr_u[id_nodo].x + dt*a_ij[s] * RK_Q[ space + id_nodo].x) ;
  f2 = (2.0*l+2.0)/radio * (dr_u[id_nodo].x + dt* a_ij[s] * RK_Q[ space + id_nodo].x)  + derivate_complex_RK(dr_u, RK_Q, id_nodo, s , space + id_nodo,  dr, dt,Nr,1).x;
  f3 = l*(l+1)/radio *(lambda[idr] + dt * a_ij[s] * RK_C[ 10*Nr + idr ])*(u[id_nodo].x + dt*a_ij[s] * RK_Q[ 0*space + id_nodo].x);

  Z.x = metric_pi_dot(A,B,alpha,idr,s,dr,dt, RK_C ,0,Nr, t , error) * f1 +  
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f2 +
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] *  RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f3;


  f1 =  l/radio * (u[id_nodo].y + dt*a_ij[s] * RK_Q[ 0*space + id_nodo].y) + (dr_u[id_nodo].y + dt*a_ij[s] * RK_Q[ space + id_nodo].y) ;
  f2 = (2.0*l+2.0)/radio * (dr_u[id_nodo].y + dt*a_ij[s] * RK_Q[ space + id_nodo].y)  + derivate_complex_RK(dr_u, RK_Q, id_nodo, s , space + id_nodo,  dr, dt,Nr,1).y;
  f3 = l*(l+1)/radio *(lambda[idr] + dt * a_ij[s] * RK_C[ 10*Nr + idr ])*(u[id_nodo].y + dt*a_ij[s] * RK_Q[ 0*space + id_nodo].y);

  Z.y = metric_pi_dot(A,B,alpha,idr,s,dr,dt, RK_C ,0,Nr , t ,error) * f1 +  
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f2 +
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f3;

  if (error==0 && t==0 && id_nodo == 50){
    printf("f_pi_dot == check\n");
  }

  return  Z;
}
//( phi , chi , PI, A, B, alpha, Da, Db, K, Kb, lambda, U, u_nodos, pi, dr_u, Nr, Nk, Nl, dr, k_min, dt, t);
__global__ void evolucion_var(double *phi, double *chi, double *PI, double *A, double *B, double *alpha, double *Da, double *Db, 
                              double * K, double *Kb, double *lambda, double *U, cufftDoubleComplex *u, cufftDoubleComplex *u_p1,  cufftDoubleComplex *pi, cufftDoubleComplex *dr_u,
                              double *SA, double *SB, double *ja, double *rho, 
                              int Nr, int Nk, int Nl, double dr, double dk, double dt, int t,  double *RK_C, cufftDoubleComplex* RK_Q , int error){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double radio;
    float epsilon = dr/2;
    
    double RK_temp[12];
    double RK_sum[12];

    cufftDoubleComplex RK_temp_Q[3];
    cufftDoubleComplex RK_sum_Q[3];

    for( int i=0 ; i<12 ; i++){
        RK_temp[i]=0.0;
    }
    for( int i=0 ; i<3 ; i++){

        RK_temp_Q[i].x=0.0;

        RK_temp_Q[i].y=0.0;
    }
    if(idx <Nr && t==0){
      for( int i=0 ; i<12 ; i++){
        RK_C[ i*Nr + idx ] = 0.0;
      }
    }
    
    if (idx<Nk*Nr*Nl && t==0){
      for( int i=0 ; i<3 ; i++){
        RK_Q[ i*Nr*Nk*Nl + idx ].x = 0.0;
        RK_Q[ i*Nr*Nk*Nl + idx ].y = 0.0;

      }
    }

    for (int s=0; s<4; s++){
      __syncthreads();

      //clasic...

      if (idx < Nr){
        if ( idx==0 ){
          radio = idx*dr +epsilon;
        }
        else{
          radio = idx*dr ;
        }
       //phi
        RK_temp[0] = (PI[idx] + dt * a_ij[s] * RK_C[ 2*Nr + idx ]) * (alpha[idx] + dt * a_ij[s] * RK_C[ 5*Nr + idx ])/(sqrt(A[idx] + dt * a_ij[s] * RK_C[ 3*Nr + idx ])*( B[idx] + dt * a_ij[s] * RK_C[ 4*Nr + idx ]));
        //chi


        RK_temp[1] = chi_dot( PI, A, B, alpha, idx, s, dr, dt, RK_C, 1,Nr, t , error);
        //PI
        RK_temp[2] =  PI_dot( chi, PI, A, B, alpha, radio*radio , idx, s, dr, dt, RK_C, 1, Nr, t , error);
        //A
        RK_temp[3] = -2.0*(alpha[idx]+ dt * a_ij[s] * RK_C[5*Nr + idx] ) * (A[idx]+ dt * a_ij[s] * RK_C[3*Nr + idx] )
                      * ((K[idx]+ dt * a_ij[s] * RK_C[8*Nr + idx] ) - 2.0*(Kb[idx]+ dt * a_ij[s] * RK_C[9*Nr + idx] ));
        //B
        RK_temp[4] = -2.0*(alpha[idx] + dt * a_ij[s] * RK_C[5*Nr + idx] ) * (B[idx] + dt * a_ij[s] * RK_C[4*Nr + idx] ) *  (Kb[idx] + dt * a_ij[s] * RK_C[9*Nr + idx] );
        //alpha
        RK_temp[5] = -2.0*(alpha[idx] + dt * a_ij[s] * RK_C[5*Nr + idx] ) * (K[idx] + dt * a_ij[s] * RK_C[8*Nr + idx] ) ;
        //Da
        RK_temp[6] = -2.0*derivate_RK(Kb,RK_C, idx, s, 9*Nr + idx,dr,dt,Nr,1);
        //Db

        RK_temp[7] = -2.0*Db_dot(Kb, alpha, idx, s, dr, dt, RK_C,1,Nr,t , error);
        //K
        
        RK_temp[8] = K_dot( A, B, alpha, Da, Db, K, Kb, lambda, U, RK_C, SA, rho, SB, radio, idx,s,dt,dr,Nr, t ,error);
        //Kb
        RK_temp[9] = Kb_dot( A, B, alpha, Da, Db, K, Kb, lambda, U, RK_C, SA, rho, radio, idx,s,dt,dr,Nr, t ,error);
        //lambda 

        RK_temp[10] = lambda_dot( A, B, alpha, Db, K, Kb, RK_C, ja, idx,s,dt,dr,Nr, t ,error);
        //U
        RK_temp[11] = U_dot( A, B, alpha, Da, Db, K, Kb,  lambda, RK_C, ja, idx,s,dt,dr,Nr, t ,error);

          //if (s==1){
        //printf("id = %d, A=%lf  \n", idx ,A[idx]);
          //}

        for( int i=0 ; i<12 ; i++){
          if(s==0){
            RK_sum[i]=b_i[s]*RK_temp[i];
            RK_C[ i*Nr + idx] =RK_temp[i];
          }
          else{
            RK_sum[i] += b_i[s]*RK_temp[i];
            RK_C[ i*Nr + idx] = RK_temp[i];

          }
        }
        /*
        if(idx==  50){
          printf("phi 0,17: %.15f\n",RK_sum[0]);
          printf("PI 0,17: %.15f\n",RK_sum[1]);
          printf("chi 0,17: %.15f\n",RK_sum[2]);
          printf("A 0,17: %.15f\n",RK_sum[3]);
          printf("B 0,17: %.15f\n",RK_sum[4]);
          printf("alpha 0,17: %.15f\n",RK_sum[5]);
          printf("Da 0,17: %.15f\n",RK_sum[6]);
          printf("Db 0,17: %.15f\n",RK_sum[7]);
          printf("K 0,17: %.15f\n",RK_sum[8]);
          printf("Kb 0,17: %.15f\n",RK_sum[9]);
          printf("lambda 0,17: %.15f\n",RK_sum[10]);
          printf("U 0,17: %.15f\n",RK_sum[11]);
        }*/

      }

      __syncthreads();

      //quantum...
      
        if (idx < Nr*Nk*Nl){
          int idr = idx%Nr;
          int id_nodo = (int)idx/Nr;
          cufftDoubleComplex temp_Q;
          //u
          RK_temp_Q[0].x = (pi[idx].x + dt * a_ij[s] * RK_Q[ 2*Nk*Nr*Nl + idx ].x) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));
          RK_temp_Q[0].y = (pi[idx].y + dt * a_ij[s] * RK_Q[ 2*Nk*Nr*Nl + idx ].y) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));

          //dr_u
          temp_Q = dr_u_dot( pi, A, B, alpha, idx , idr ,  Nk*Nr*Nl ,s, dr, dt, RK_C, RK_Q, 1,Nr, t ,error);
          RK_temp_Q[1].x =  temp_Q.x;
          RK_temp_Q[1].y =  temp_Q.y;

          //pi
          int k,l;
          k = id_nodo%Nk;
          l = id_nodo/Nl;
          /*
          if(idx==300){
            printf("id_nodo = %d , k = %d , l = %d \n",id_nodo,k,l);
          }
            */
          temp_Q = f_pi_dot( u ,dr_u , A, B, alpha, lambda ,  RK_C, RK_Q, radio,  idx, idr, Nk*Nr*Nl,s, k, l, dt, dr, Nr, t , error);
          RK_temp_Q[2].x =  temp_Q.x;
          RK_temp_Q[2].y =  temp_Q.y;
         /* if(s==0){
              printf("idr = %d, id = %d, Rk=%lf + %lf i \n", idr, idx , RK_temp_Q[0].x,RK_temp_Q[0].y);
              
          }*/

          for( int i=0 ; i<3 ; i++){
            if(s==0){
              RK_sum_Q[i].x = b_i[s]*RK_temp_Q[i].x;
              RK_Q[ i*Nr*Nk*Nl + idx].x =RK_temp_Q[i].x;

              RK_sum_Q[i].y = b_i[s]*RK_temp_Q[i].y;
              RK_Q[ i*Nr*Nk*Nl + idx].y =RK_temp_Q[i].y;

            }
            else{
              RK_sum_Q[i].x += b_i[s]*RK_temp_Q[i].x;
              RK_Q[ i*Nr*Nk*Nl + idx].x =RK_temp_Q[i].x;

              RK_sum_Q[i].y +=b_i[s]*RK_temp_Q[i].y;
              RK_Q[ i*Nr*Nk*Nl + idx].y =RK_temp_Q[i].y;
  
            }
          }
          if(idx== 17*Nk*Nr + 0*Nr + 50){
            printf("RKQ0 0,17: %.15f\n",RK_sum_Q[0].x);
            printf("RKQ1 0,17: %.15f\n",RK_sum_Q[1].x);
            printf("RKQ2 0,17: %.15f\n",RK_sum_Q[2].x);

          }
      }
      __syncthreads();

    
    }
    if(idx < Nr){
      phi[(t+1)*Nr +idx] = phi[t*Nr +idx] + dt*RK_sum[0];
      PI[idx]        += dt*RK_sum[1];
      chi[idx]       += dt*RK_sum[2];
      A[idx]         += dt*RK_sum[3];
      B[idx]         += dt*RK_sum[4];
      alpha[idx]     += dt*RK_sum[5];
      Da[idx]        += dt*RK_sum[6];
      Db[idx]        += dt*RK_sum[7];
      K[idx]         += dt*RK_sum[8];
      Kb[idx]        += dt*RK_sum[9];
      lambda[idx]    += dt*RK_sum[10];
      U[idx]         += dt*RK_sum[11];
    }
    __syncthreads();

    if( idx < Nr*Nk*Nl ){
      u_p1[idx].x    = u[idx].x +  dt*RK_sum_Q[0].x;
      u_p1[idx].y    = u[idx].y +  dt*RK_sum_Q[0].y;

      dr_u[idx].x += dt*RK_sum_Q[1].x;
      dr_u[idx].y += dt*RK_sum_Q[1].y;

      pi[idx].x   += dt*RK_sum_Q[2].x;
      pi[idx].y   += dt*RK_sum_Q[2].y;

    }

    __syncthreads();
    if (idx==2){
      //A[idx-1]=B[idx-1];
      //Kb[idx-1]=K[idx-1]/3.0;
      phi[t*Nr + 0]=(4.0*phi[t*Nr + 1]-phi[t*Nr + 2])/3.0;
      alpha[0]=(4.0*alpha[1]-alpha[2])/3.0;
      PI[0]=(4.0*PI[1]-PI[2])/3.0;
      A[0]=(4.0*A[1]-A[2])/3.0;
      B[0]=(4.0*B[1]-B[2])/3.0;
      K[0]=(4.0*K[1]-K[2])/3.0;
      Kb[0]=(4.0*Kb[1]-Kb[2])/3.0;

      //antisimetricas

      //simetricas
    }
      __syncthreads();
    
    if(idx== 50){
      printf("phi t0 : %.15f\n",phi[t*Nr + idx]);
      printf("phi t1 : %.15f\n",phi[(t+1)*Nr + idx]);

      printf("PI : %.15f\n",PI[idx]);
      printf("chi : %.15f\n",chi[idx]);
      printf("alpha : %.15f\n",alpha[idx]);
  
      printf("A : %.15f\n",A[idx]);
      printf("B : %.15f\n",B[idx]);
      printf("K_past : %.15f\n",K[idx-1]);
      printf("KB_past : %.15f\n",Kb[idx-1]);
      printf("K : %.15f\n",K[idx]);
      printf("KB : %.15f\n",Kb[idx]);
      printf("K_last : %.15f\n",K[idx+1]);
      printf("KB_last : %.15f\n",Kb[idx+1]);
      printf("DB : %.15f\n",Db[idx]);
      printf("Da : %.15f\n",Da[idx]);
      printf("U : %.15f\n",U[idx]);
      printf("lambda : %.15f\n",lambda[idx]);
      printf("rho : %.15f\n",rho[idx]);
      printf("ja : %.15f\n",ja[idx]);
      printf("SA : %.15f\n",SA[idx]);
      printf("SB : %.15f\n",SB[idx]);

      printf("Re(u) 0,17: %.15f\n",u[17*Nk*Nr + 0*Nr + idx].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[17*Nk*Nr + 0*Nr + idx].x);

      printf("RKQ 0,17: %.15f\n",u[17*Nk*Nr + 0*Nr + idx].x + dt*RK_sum_Q[0].x);

      printf("Re(dr_u) : %.15f\n",dr_u[17*Nk*Nr + 0*Nr + idx].x);
      printf("Re(pi)  : %.15f\n",pi[17*Nk*Nr + 0*Nr + idx].x);
      printf("Im(u) : %.15f\n",u[17*Nk*Nr + 0*Nr + idx].y);
      printf("Im(u_p1) : %.15f\n",u_p1[17*Nk*Nr + 0*Nr + idx].y);

      printf("Im(dr_u) : %.15f\n",dr_u[17*Nk*Nr + 0*Nr + idx].y);
      printf("Im(pi) : %.15f\n",pi[17*Nk*Nr + 0*Nr + idx].y);
  }
}

__global__ void fluctuation( cufftDoubleComplex *u, cufftDoubleComplex *u_p1, cufftDoubleComplex *pi, double *temp_array ,  double *temp_fluct,   double dr, double dk, double dt, int Nr, int Nk , int Nl, int l, int t ){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;
  float epsilon = dr/2;
  //temp_fluct (5xNr)
  double cte= 1.0/(4.0*3.1415);
  double u_snake;
  double d_u_temp_x;
  double d_u_temp_y;
  if (idx < Nr){
    if(idx==0){
      radio=dr/2.0;
    }
    else{
      radio=idx*dr;
    }

    double suma_par = 0.0 , suma_impar = 0.0 ;

    temp_fluct[ 0*Nr + idx] = 0.0;

    for (int k=0 ; k < Nk ; k++){
      /*if(t==0){
        d_u_temp_x =pi[  l*Nk*Nr + k*Nr + idx ].x;
        d_u_temp_y =pi[  l*Nk*Nr + k*Nr + idx ].y;

      }*/
      d_u_temp_x = ( u_p1[  l*Nk*Nr + k*Nr + idx].x * pow(radio , l) - u[l*Nk*Nr + k*Nr + idx].x * pow(radio , l) )/dt ;
      d_u_temp_y = ( u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) - u[l*Nk*Nr + k*Nr + idx].y * pow(radio , l) )/dt ;
      if(idx==50 && l==17){
        printf("d_u_temp_x = %.15f  |  k = %d\n",d_u_temp_x,k);
          printf("u = %.15lf * radio = %f|\n",u[l*Nk*Nr + k*Nr + idx].x,pow(radio , l),k);
          printf("u_p1 = %.15f * radio = %f|\n", u_p1[  l*Nk*Nr + k*Nr + idx].x,pow(radio , l),k);

        
      }
      if (k==0){
        temp_fluct[ 0*Nr + idx] = d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y;
      }
      else if (k%2==0 && k>0 &&k<Nk-1){
        suma_par += 4.0*( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }
      else if (k%2==1 && k>0 &&k<Nk-1){
        suma_impar += 2-0*( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }
      else if (k==Nk-1){
        temp_fluct[ 0*Nr + idx] += ( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }

    }
    temp_fluct[ 0*Nr + idx] += suma_impar + suma_par ;
    temp_fluct[ 0*Nr + idx] = temp_fluct[ 0*Nr + idx] * (2.0*l + 1.0) * cte * dk/3.0;
    if(idx==50){
      printf("temp_fluct_0 = %1.5f |  l = %d\n",temp_fluct[ 0*Nr + idx],l);
    }

    suma_par = 0.0 ;
    suma_impar = 0.0 ;
    cufftDoubleComplex res;

    for (int k=0 ; k < Nk ; k++){
      res = derivate_complex(u_p1,  l*Nk*Nr + k*Nr + idx, dr,Nr,0 );
      d_u_temp_x = res.x;
      d_u_temp_y = res.y ;

      if (k==0){
        temp_fluct[ 1*Nr + idx] = d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y;
      }
      else if (k%2==0 && k>0 &&k<Nk-1){
        suma_par += 4.0*( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }
      else if (k%2==1 && k>0 &&k<Nk-1){
        suma_impar += 2-0*( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }
      else if (k==Nk-1){
        temp_fluct[ 1*Nr + idx] += ( d_u_temp_x*d_u_temp_x + d_u_temp_y*d_u_temp_y);
      }

    }
    temp_fluct[ 1*Nr + idx] += suma_impar + suma_par ;
    temp_fluct[ 1*Nr + idx] = temp_fluct[ 1*Nr + idx] * (2*l + 1) * cte * dk/3.0;
    if(idx==50){
      printf("temp_fluct_1 = %1.5f |  l = %d\n",temp_fluct[ 1*Nr + idx],l);
    }


    suma_par = 0.0 ;
    suma_impar = 0.0 ;

    for (int k=0 ; k < Nk ; k++){
      res = derivate_complex(u_p1, l*Nk*Nr + k*Nr + idx, dr,Nr,0 );
      d_u_temp_x = res.x * ( u_p1[  l*Nk*Nr + k*Nr + idx].x * pow(radio , l) - u[l*Nk*Nr + k*Nr + idx].x * pow(radio , l) )/dt ; 
      d_u_temp_y = res.y * ( u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) - u[l*Nk*Nr + k*Nr + idx].y * pow(radio , l) )/dt ;

      if (k==0){
        temp_fluct[ 2*Nr + idx] = d_u_temp_x - d_u_temp_y;
      }
      else if (k%2==0 && k>0 &&k<Nk-1){
        suma_par += 4.0*(  d_u_temp_x - d_u_temp_y );
      }
      else if (k%2==1 && k>0 &&k<Nk-1){
        suma_impar += 2-0*(  d_u_temp_x - d_u_temp_y );
      }
      else if (k==Nk-1){
        temp_fluct[ 2*Nr + idx] += ( d_u_temp_x - d_u_temp_y );
      }

    }
    temp_fluct[ 2*Nr + idx] += suma_impar + suma_par ;
    temp_fluct[ 2*Nr + idx] = temp_fluct[ 2*Nr + idx]  * (2*l + 1) * cte* dk/3.0;
    if(idx==50){
      printf("temp_fluct_2 = %1.5f |  l = %d\n",temp_fluct[ 2*Nr + idx],l);
    }


    suma_par = 0.0 ;
    suma_impar = 0.0 ;

    for (int k=0 ; k < Nk ; k++){
      u_snake = u_p1[ l*Nk*Nr + k*Nr + idx].x * pow(radio , l) *u_p1[  l*Nk*Nr + k*Nr + idx].x * pow(radio , l) 
                + u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) * u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) ;

      if (k==0){
        temp_fluct[ 3*Nr + idx] = u_snake;
      }
      else if (k%2==0 && k>0 &&k<Nk-1){
        suma_par += 4.0*( u_snake );
      }
      else if (k%2==1 && k>0 &&k<Nk-1){
        suma_impar += 2-0*( u_snake );
      }
      else if (k==Nk-1){
        temp_fluct[ 3*Nr + idx] += ( u_snake );
      }

    }
    temp_fluct[ 3*Nr + idx] += suma_impar + suma_par ;
    temp_fluct[ 3*Nr + idx] = temp_fluct[ 3*Nr + idx] * 0.5 * (l+1)*(2*l + 1) * cte* dk/3.0;

    suma_par = 0.0 ;
    suma_impar = 0.0 ;

    double angle;
    angle = acos( idx*(Nr-1)/(idx*dr) );

    for (int k=0 ; k < Nk ; k++){
      u_snake = u_p1[  l*Nk*Nr + k*Nr + idx].x * pow(radio , l) *u_p1[  l*Nk*Nr + k*Nr + idx].x * pow(radio , l) 
                + u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) * u_p1[  l*Nk*Nr + k*Nr + idx].y * pow(radio , l) ;

      if (k==0){
        temp_fluct[ 4*Nr + idx] = u_snake;
      }
      else if (k%2==0 && k>0 &&k<Nk){
        suma_par += 4.0*( u_snake );
      }
      else if (k%2==1&& k>0 &&k<Nk){
        suma_impar += 2-0*( u_snake );
      }
      else if (k==Nk-1){
        temp_fluct[ 4*Nr + idx] += ( u_snake );
      }

    }
    temp_fluct[ 4*Nr + idx] += suma_impar + suma_par ;
    temp_fluct[ 4*Nr + idx] = temp_fluct[ 4*Nr + idx] * cte * (l +1)*(2*l +1) *sin(angle)*sin(angle)* dk/3.0; 


    temp_array[ 0*Nl*Nr + l*Nr + idx ]=temp_fluct[ 0*Nr + idx];
    temp_array[ 1*Nl*Nr + l*Nr + idx ]=temp_fluct[ 1*Nr + idx];
    temp_array[ 2*Nl*Nr + l*Nr + idx ]=temp_fluct[ 2*Nr + idx];
    temp_array[ 3*Nl*Nr + l*Nr + idx ]=temp_fluct[ 3*Nr + idx];
    temp_array[ 4*Nl*Nr + l*Nr + idx ]=temp_fluct[ 4*Nr + idx];
  }
}


__global__  void suma_fluct(double *temp_array , int Nl, int Nr){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < Nr){
    for (int l = 1 ; l < Nl ; l++ ){

      temp_array[ idx ] += temp_array[ 0*Nl*Nr + l*Nr + idx ];
      temp_array[ 1*Nl*Nr + idx ] += temp_array[ 1*Nl*Nr + l*Nr + idx ];
      temp_array[ 2*Nl*Nr + idx ] += temp_array[ 2*Nl*Nr + l*Nr + idx ];
      temp_array[ 3*Nl*Nr + idx ] += temp_array[ 3*Nl*Nr + l*Nr + idx ];
      temp_array[ 4*Nl*Nr + idx ] += temp_array[ 4*Nl*Nr + l*Nr + idx ];

    }
  }
}
__global__ void stress_energy(double *SA, double *SB, double *rho, double *ja, double *phi, double *A, double *B, double *alpha, double *temp_array , double dr, double dt, int Nr, int Nk, int Nl, int t){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  float epsilon = dr/2;
  double dt_phi, dr_phi;
  if(idx < Nr){

    dt_phi = (phi[(t+1)*Nr + idx ] - phi[(t)*Nr +idx])/dt;

    if((t*Nr + idx)%Nr == 0){
      dr_phi = ( -1.5*phi[(t+1)*Nr + idx] + 2.0*phi[(t+1)*Nr + idx+1] - 0.5*phi[(t+1)*Nr +idx +2])/dr;
    }
    else if ((t*Nr + idx)%Nr == Nr-1){
      dr_phi = ( 0.5*phi[(t+1)*Nr + idx] - 2.0*phi[(t+1)*Nr + idx+1] + 1.5*phi[(t+1)*Nr +idx +2])/dr;
    }
    else{
      dr_phi = ( 0.5*phi[(t+1)*Nr + idx + 1]  - 0.5*phi[(t+1)*Nr +idx - 1])/dr;
    }

    double  radio;
    if(idx==0){
      radio=idx*dr + dr/2.0;
    }
    else{
      radio=idx*dr;
    }

    //variables

    double espectation_dt_phi;
    double espectation_dr_phi;
    double espectation_dr_dt_phi;
    double espectation_dtheta_phi;
    double espectation_dphi_phi;


    espectation_dt_phi = dt_phi*dt_phi + temp_array[ idx ];
    espectation_dr_phi = dr_phi*dr_phi + temp_array[ 1*Nl*Nr + idx ];
    espectation_dr_dt_phi = dt_phi*dr_phi + temp_array[ 2*Nl*Nr + idx ];
    espectation_dtheta_phi = temp_array[ 3*Nl*Nr + idx ];
    // espectation_dphi_phi = temp_array[ 4*Nl*Nr + idx ];


    double xPI2x , xchi2x, xpichix;
    xPI2x = A[idx]*B[idx]*B[idx]/(alpha[idx]*alpha[idx]) * espectation_dt_phi;
    xchi2x = espectation_dr_phi;
    xpichix = sqrt(A[idx])*B[idx]/alpha[idx] * espectation_dr_dt_phi;
    //Stress-Energy

    double mu = 1.0 + 4.0*sqrt(3.0);

    rho[idx] =  1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xchi2x) + 1.0/(B[idx]*radio*radio)*espectation_dtheta_phi - 0.5*mu*phi[(t+1)*Nr + idx]*phi[(t+1)*Nr + idx];

    ja[idx] = - xpichix/(sqrt(A[idx])*B[idx]);

    SA[idx] = 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xchi2x) - 1.0/(B[idx]*radio*radio)*espectation_dtheta_phi - 0.5*mu*phi[(t+1)*Nr + idx]*phi[(t+1)*Nr + idx];

    SB[idx] = 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) - xchi2x)  - 0.5*mu*phi[(t+1)*Nr + idx]*phi[(t+1)*Nr + idx];
    if(idx== 50){
      printf("Stress_Energy:\n"); 
        printf("phi t0 : %.15f\n",phi[t*Nr + idx]);
        printf("phi t1 : %.15f\n",phi[(t+1)*Nr + idx]);
        printf(" <x|PI|x>^2 = %1.5f\n",xPI2x);
        printf(" <x|chi|x>^2 = %1.5f\n",xchi2x);
        printf(" <x|PI chi|x>^2 = %1.5f\n",xpichix);
    
        printf("alpha : %.15f\n",alpha[idx]);
    
        printf("A : %.15f\n",A[idx]);
        printf("B : %.15f\n",B[idx]);
    
        printf("rho : %.15f\n",rho[idx]);
        printf("ja : %.15f\n",ja[idx]);
        printf("SA : %.15f\n",SA[idx]);
        printf("SB : %.15f\n",SB[idx]);
    
    
      }
  }

}
__global__ void cambio_u(cufftDoubleComplex *u, cufftDoubleComplex *u_p1, int Nr, int Nk, int Nl){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  if(idx < Nr*Nk*Nl){
    u[idx].x = u_p1[idx].x;
    u[idx].y=u_p1[idx].y;

  }  
  __syncthreads();

}


/*Define j0(z) */
double b0(double z){
  return sin(z)/z;
}

/*Define j1(z) */
double b1(double z){
  return sin(z)/(z*z)-cos(z)/z;
}

/*Define jn(z) */
double sjn(double z,int n){
  double out;
  if (n==0){
      out = b0(z);
  }
  else if(n==1){
      out = b1(z);
  }
  /*using recurrence relation */
  else{
      out = (2*n-1)*sjn(z,n-1)/z-sjn(z,n-2);
  }
  return out;
}
double f_sinc(double z){
  return sin(z)/z;
}

void sjn(double *f,double z,double dr,int Nr,int n){
  double out;
  for(int m=0 ; m<n ; m++){
    for(int idx=0 ; idx <Nr; idx++){
      if (idx==0){
        f[idx] = (-1.5*f[idx] + 2.0*f[idx+1] -0.5*f[idx+2] )/dr;
        f[idx] = f[idx]/(z*0.5);

      }

      else if (idx == Nr-1){
        f[idx] = (1.5*f[idx] - 2.0*f[idx-1] +0.5*f[idx-2] )/dr;  
        f[idx] = f[idx]/(z*idx);

      }
      else{
        f[idx] = (0.5*f[idx+1] - 0.5*f[idx-1] )/dr;
        f[idx] = f[idx]/(z*idx);
      }
    }
  }
}



void u_initial(cufftDoubleComplex *u,int k, int l,int Nk,int Nl,int Nr,double dr){
    int id_u=l*Nk*Nr + k*Nr ;
    double k_min = consta_pi/15.0;
    double omega=k_min*(k+1);
    double *temp;
    temp = (double *)malloc(Nr*sizeof(double));

    for (int r=0 ; r<Nr ; r++){
      if (r==0){
        temp[r]=f_sinc(k_min*(k+1)*dr*0.5);
      }
      else{
        temp[r]=f_sinc(k_min*(k+1)*dr*r);
      }
    }
    
    sjn(temp, k_min*(k+1)*dr ,dr,Nr,l);

    for (int r=0 ; r<Nr ; r++){
      if (r==0){        
        temp[r]= pow(-1,l)*pow(k_min*(k+1)*dr*0.5 , l)*temp[r];
      }
      else{
        temp[r]= pow(-1,l)*pow(k_min*(k+1)*dr*r , l)*temp[r];
      }
    }
    for (int r=0 ; r<Nr ; r++){
        if (r==0){
            u[id_u + r].x =  (k_min*(k+1)/sqrt(consta_pi * omega)) * temp[r] / (pow(dr*r+dr/2.0,l));
            u[id_u + r].y = 0.0;
        }
        else{
            u[id_u + r].x = (k_min*(k+1)/sqrt(consta_pi * omega)) * temp[r] / pow(dr*r,l);
            u[id_u + r].y=0.0;

        }
    }
}

void pi_initial(cufftDoubleComplex* pi,int k, int l,int Nk,int Nl,int Nr,double dr){
    int id_u=l*Nk*Nr + k*Nr ;
    double k_min = consta_pi/15.0;
    double omega=k_min*(k+1);

    for (int r=0 ; r<Nr ; r++){
        if (r==0){
            pi[id_u + r].x = 0.0;
            pi[id_u + r].y = -omega*(k_min*(k+1)/sqrt(consta_pi * omega)) * sjn(  0 , l ) / (pow(0,l));


        }
        else{
            pi[id_u + r].x= 0.0;
            pi[id_u + r].y = -omega*(k_min*(k+1)/sqrt(consta_pi * omega)) * sjn(  k_min*(k+1)*dr*r , l ) / pow(dr*r,l);

        }
    }
}
void dr_u_initial(cufftDoubleComplex* dr_u,int k, int l,int Nk,int Nl,int Nr,double dr){
    int id_u=l*Nk*Nr + k*Nr ;
    double k_min = consta_pi/15.0;
    double omega=k_min*(k+1);
    double dr_j;
    for (int r=0 ; r<Nr ; r++){
        if (r==0){
            dr_j= (-1.5*sjn(  0 , l ) + 2.0*sjn(  k_min*(k+1)*dr*(r+1) , l ) - 0.5*sjn(  k_min*(k+1)*dr*(r+2) , l ))/(dr);

            dr_u[id_u + r].x = (k_min*(k+1)/sqrt(consta_pi * omega)) * ( dr_j  / (pow(0.0,l)) -  sjn(  0.0 , l ) * l/ (pow(0.0,l+1)));
            dr_u[id_u + r].y = 0.0;

        }
        else if (r==Nr-1){
            dr_j= (1.5*sjn(  k_min*(k+1)*dr*r , l )- 2.0*sjn(  k_min*(k+1)*dr*(r+1) , l ) + 0.5*sjn(  k_min*(k+1)*dr*(r+2) , l ))/(dr);

            dr_u[id_u + r].x = (k_min*(k+1)/sqrt(consta_pi * omega)) * ( dr_j / (pow(dr*r,l)) - sjn(  k_min*(k+1)*dr*r , l ) * l / (pow(dr*r,l+1)));
            dr_u[id_u + r].y = 0.0;

        }
        else{
            dr_j= (sjn(  k_min*(k+1)*dr*(r+1) , l ) - sjn(  k_min*(k+1)*dr*(r-1) , l ))/(2.0*dr);

            dr_u[id_u + r].x = (k_min*(k+1)/sqrt(consta_pi * omega)) *  ( dr_j / (pow(dr*r,l)) - sjn(  k_min*(k+1)*dr*r , l ) * l / (pow(dr*r,l+1)));
            dr_u[id_u + r].y = 0.0;

        }
    }
}
void inicial_phi(double *phi, double dr,int Nr){
  double a=0.2;
  double std=1.5;
  for (int i=0;i<Nr;i++){
    //phi[i]=a;
    phi[i]=a*(i*dr-20.0)/std*exp(-( (i*dr-20.0) /std)*((i*dr-20.0) /std));

    //phi[i]=a*exp(-( (i*dr-20.0) /std)*((i*dr-20.0) /std));
  }
 }
 double dfr(double r,double a,double std){
  //return a*exp(-( (r-20.0) /std)*((r-20.0) /std));
  return a*(r-20.0)/std*exp(-( (r-20.0) /std)*((r-20.0) /std));
 }
void iniciar_A(double *A,double *chi, double dr,int Nr){
  double *ks;
  ks=(double *)malloc(sizeof(double)*5);
  double sumas=0.0;
  double A0,rs,chi0;
  double epsilon=dr*0.2;
    //double Mp=1.0/sqrt(8.0*3.1415);
    double Mp=1.0;
        double a=0.2;
  double std=1.5;
  A[0]=1.0;
  ks[0]=0.0;


  for (int idx=1;idx<Nr;idx++){

    sumas=0.0;
    for (int s=0 ; s<4 ;s++){
      A0=(A[idx-1]+dr*host_a_ij[s]*ks[s]);

      //ks[s+1]= A0*((1.0/rs)*(1.0-A0)+rs*chi[idx-1]*chi[idx-1]*0.5 + 0.0);
      //chi0=0.01*(rs-5.0)*(-2.0/(0.2*0.2))*exp(-1.0*((rs-5.0)/0.2)*((rs-5.0)/0.2));
        double temp;
        temp=0.0;

            if (idx==1){
              for (int m=0;m<order;m++){
                  rs= dr*((idx-1)+m)+dr*host_c_i[s];

                temp += host_coefficient_adelantada[m]*dfr(rs,a,std);
              } 

            }
            else if (idx > Nr-(int)(order-1)/2-1){
              for (int m=-order+1;m<1;m++){
              rs= dr*((idx-1)+m)+dr*host_c_i[s];

              temp +=  host_coefficient_atrasada[-m]*dfr(rs,a,std);
              }
            }
            else{
            for (int m=0;m<order;m++){
              rs= dr*((idx-1)-(int)(order-1)/2+m)+dr*host_c_i[s];

              temp += host_coefficient_centrada[m]*dfr(rs,a,std);
            }
            
          }
        if (idx-1==0){
        rs= dr*(idx-1)+dr*host_c_i[s] + epsilon;
      }
      else{
        rs= dr*(idx-1)+dr*host_c_i[s];
      }
            chi0= temp/dr; 
      ks[s+1]= A0*((1.0/rs)*(1.0-A0)+rs*chi0*chi0*0.5 + 0.0);

      sumas += host_b_i[s]*ks[s+1];
    }
   A[idx]=A[idx-1]+dr*sumas;
  }
}
void rellenar(double *f, int Nr,double num){
  for (int i=0;i<Nr;i++){
    f[i]=num;
  }
}


/*notas: primera sol buena fue con Nr=500 ,dr=50.0/Nr y dt=5/20.000, con amplitud 0.0002 y std=1.5
Cuando aumento la amplitud por ejemplo a a=0.02 y a =0.002 las simulaciones divergen con Nr=500 dr=50/r y dt=5.0/80.000
divergen en el rebote...*/
int main(){
    double time1, timedif;
    time1 = (double) clock();            /* get initial time */
    time1 = time1 / CLOCKS_PER_SEC;  
  int Nr=500;
  int Nt=4000;
  int minit= Nt/100;
  // Defino los array del host
  double *A,*B,*alpha,*phi,*chi,*PI,*lambda,*K,*Kb,*U, *Da, *Db, *temp_phi, *temp_Kb,*temp_alpha;
  double *rho,*ja,*SA,*SB;


  double *cuda_A,*cuda_B,*cuda_alpha,*cuda_phi,*cuda_chi,*cuda_PI,*cuda_lambda,*cuda_K,*cuda_Kb,*cuda_U, *cuda_Da, *cuda_Db, *cuda_temp_phi, *cuda_temp_Kb,*cuda_temp_alpha;
  double *cuda_rho,*cuda_ja,*cuda_SA,*cuda_SB;

  //double *K_s,*b_i, *a_ij, *c_i;
  double *coef_atrasada,*coef_centrada,*coef_adelantada;

  //deltas
  double dr=20.0/Nr;
  //double dt=5.0/10000;
  double dt=dr/4.0; //el bueno
  //double dt=4.0/20000;
  printf("dr=%f , dt=%f",dr,dt);
// mallocs
phi=(double *)malloc(Nr*Nt*sizeof(double));
temp_Kb=(double *)malloc(Nt*Nr*sizeof(double));
temp_phi=(double *)malloc(Nr*sizeof(double));
temp_alpha=(double *)malloc(Nt*Nr*sizeof(double));

  PI=(double *)malloc(Nr*sizeof(double));
  chi=(double *)malloc(Nr*sizeof(double));

  A=(double *)malloc(Nr*sizeof(double));
  B=(double *)malloc(Nr*sizeof(double));
  lambda=(double *)malloc(Nr*sizeof(double));
  alpha=(double *)malloc(Nr*sizeof(double));
  K=(double *)malloc(Nr*sizeof(double));
  Kb=(double *)malloc(Nr*sizeof(double));
  U=(double *)malloc(Nr*sizeof(double));
  Db=(double *)malloc(Nr*sizeof(double));
  Da=(double *)malloc(Nr*sizeof(double));

  ja=(double *)malloc(Nr*sizeof(double));
  rho=(double *)malloc(Nr*sizeof(double));
  SA=(double *)malloc(Nr*sizeof(double));
  SB=(double *)malloc(Nr*sizeof(double));


  coef_centrada=(double *)malloc(order*sizeof(double));
  coef_atrasada=(double *)malloc(order*sizeof(double));
  coef_adelantada=(double *)malloc(order*sizeof(double));




cargar_coeff_centradas_df(coef_centrada,order);
cargar_coeff_atrasada_df(coef_atrasada,order);
cargar_coeff_adelantada_df(coef_adelantada,order);
for (int i=0; i < order ;i++){
  /*
  coefficient_centrada[i]=coef_centrada[i];
  coefficient_adelantada[i]=coef_adelantada[i];
  coefficient_atrasada[i]=coef_atrasada[i];
*/
  host_coefficient_centrada[i]=coef_centrada[i];
  host_coefficient_adelantada[i]=coef_adelantada[i];
  host_coefficient_atrasada[i]=coef_atrasada[i];
}

printf("Diferencias finitas: check\n");

//condiciones iniciales
inicial_phi(phi,dr,Nr);
inicial_phi(temp_phi,dr,Nr);

difference_tenth(chi,phi,dr,Nr); 

printf("Condiciones iniciales de phi y chi: check\n");

rellenar(PI, Nr, 0.0);
rellenar(K,Nr,0.0);
rellenar(Kb,Nr,0.0);
rellenar(temp_Kb,Nr,0.0);

rellenar(Da,Nr,0.0);
rellenar(Db,Nr,0.0);
rellenar(alpha, Nr, 1.0);
//rellenar(alpha, Nr, 1.0);

rellenar(B,Nr,1.0);
iniciar_A(A,chi,dr,Nr);
printf("Condiciones iniciales parte 2: check\n");

calculate_rho(rho,PI,chi,A,B,Nr);
calculate_SA(SA,PI,chi,A,B,Nr);
calculate_SB(SB,PI,chi,A,B,Nr);
calculate_ja(ja,PI,chi,A,B,Nr);
printf("Valores iniciales de Tuv: check\n");
double *dr_A0;
dr_A0=(double *)malloc(Nr*sizeof(double));
difference_tenth(dr_A0,A,dr,Nr);
for(int idx=0;idx<Nr;idx++){
  if (idx==0){
    lambda[idx]=(1.0-A[idx]/B[idx])/(idx*dr+dr*0.5);      
  }
  else{
    lambda[idx]=(1.0-A[idx]/B[idx])/(idx*dr);
  }

  U[idx]= (dr_A0[idx]-4.0*lambda[idx])/A[idx];


}
guardar_salida_chi(A,Nr,1.0);

    timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
    printf("The elapsed time is %lf seconds\n", timedif);
    printf("inicio de la parte cuantica:\n");
//Quantum...
cufftDoubleComplex *u_nodos;
cufftDoubleComplex *u_nodos_p1;

cufftDoubleComplex *pi;
cufftDoubleComplex *dr_u;

cufftDoubleComplex *cuda_u_nodos;
cufftDoubleComplex *cuda_u_nodos_p1;

cufftDoubleComplex *cuda_pi;
cufftDoubleComplex *cuda_dr_u;

int Nk=20,Nl=20;
  u_nodos=(cufftDoubleComplex*)malloc(Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  u_nodos_p1=(cufftDoubleComplex*)malloc(Nk*Nl*Nr*sizeof(cufftDoubleComplex));

  pi=(cufftDoubleComplex*)malloc(       Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  dr_u=(cufftDoubleComplex*)malloc(     Nk*Nl*Nr*sizeof(cufftDoubleComplex));

//pendiente inicial del A...
// cuda mallocs
    time1 = (double) clock();            /* get initial time */
    time1 = time1 / CLOCKS_PER_SEC; 
    int idk;
    #pragma omp parallel for 
    for (idk=0;idk<Nk;idk++){
      for (int idl=0;idl<Nl;idl++){
          u_initial(u_nodos,idk,idl,Nk, Nl, Nr,dr);
          pi_initial(pi,idk,idl,Nk, Nl, Nr,dr);
          dr_u_initial(dr_u,idk,idl,Nk, Nl, Nr,dr);
      }
    }
    #pragma omp parallel for
    for (idk=0;idk<Nk;idk++){
      for (int idl=0;idl<Nl;idl++){
        for (int idx = 0 ; idx < Nr ; idx++){
          u_nodos_p1[idl*Nk*Nr + idk *Nr + idx].x = pi[idl*Nr*Nk + idk*Nr + idx].x;
          u_nodos_p1[idl*Nk*Nr + idk *Nr + idx].y = pi[idl*Nr*Nk + idk*Nr + idx].y;
        }
      }
    }


    timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
    printf("The elapsed time is %lf seconds, opm\n", timedif);


    for (int idk=0;idk<Nk;idk++){
      for (int idl=0;idl<Nl;idl++){
        printf("Reu_%d,%d = %.15lf \n",idk,idl,u_nodos[ idl*Nk*Nr + idk*Nr + 50 ].x);

      }
    }
/*
for (int idk=0;idk<Nk;idk++){
  for (int idl=0;idl<Nl;idl++){
    printf("Reu_%d,%d = %lf \n",idk,idl,u_nodos[ idl*Nk*Nl + idk*Nr + 50 ].x);
    printf("Repi_%d,%d = %lf \n",idk,idl,pi[ idl*Nk*Nl + idk*Nr + 50 ].x);
    printf("Redr_u_%d,%d = %lf \n",idk,idl,dr_u[ idl*Nk*Nl + idk*Nr + 50 ].x);
    printf("Imu_%d,%d = %lf \n",idk,idl,u_nodos[ idl*Nk*Nl + idk*Nr + 50 ].y);
    printf("Impi_%d,%d = %lf \n",idk,idl,pi[ idl*Nk*Nl + idk*Nr + 50 ].y);
    printf("Imdr_u_%d,%d = %lf \n",idk,idl,dr_u[ idl*Nk*Nl + idk*Nr + 50 ].y);
  }
}
*/
printf("Condiciones iniciales: check\n");

//dr_Db = (double *)malloc(Nr*sizeof(double));
//dr_Da = (double *)malloc(Nr*sizeof(double));
//dr_K = (double *)malloc(Nr*sizeof(double));
//dr_Kb = (double *)malloc(Nr*sizeof(double));
//dr_temp = (double *)malloc(Nr*sizeof(double));


//El U y lambda son los unicos que no aparecen en derivadas
//es necesario que cada K tenga dimensiones  [ stage x Nr ]
//para hacer las diferencias finitas.


printf("Rho antes de la simulacion rho = %.15f\n",conservacion(rho,dr,Nr));

//Cuda...


cudaMalloc((void **)&cuda_phi, Nt*Nr*sizeof(double));
cudaMalloc((void **)&cuda_chi, Nr*sizeof(double));
cudaMalloc((void **)&cuda_PI, Nr*sizeof(double));

cudaMalloc((void **)&cuda_A, Nr*sizeof(double));
cudaMalloc((void **)&cuda_B, Nr*sizeof(double));
cudaMalloc((void **)&cuda_alpha, Nr*sizeof(double));
cudaMalloc((void **)&cuda_Da, Nr*sizeof(double));
cudaMalloc((void **)&cuda_Db, Nr*sizeof(double));
cudaMalloc((void **)&cuda_K, Nr*sizeof(double));
cudaMalloc((void **)&cuda_Kb, Nr*sizeof(double));
cudaMalloc((void **)&cuda_lambda, Nr*sizeof(double));
cudaMalloc((void **)&cuda_U, Nr*sizeof(double));


cudaMalloc((void **)&cuda_u_nodos, Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&cuda_u_nodos_p1, Nr*Nk*Nl*sizeof(cufftDoubleComplex));

cudaMalloc((void **)&cuda_dr_u, Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&cuda_pi, Nr*Nk*Nl*sizeof(cufftDoubleComplex));


cudaMalloc((void **)&cuda_rho, Nr*sizeof(double));
cudaMalloc((void **)&cuda_SA, Nr*sizeof(double));
cudaMalloc((void **)&cuda_SB, Nr*sizeof(double));
cudaMalloc((void **)&cuda_ja, Nr*sizeof(double));


//cuda memcpy

//CLASSIC
cudaMemcpy(cuda_phi, phi, Nr*Nt*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_chi, chi, Nr*sizeof(double), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_A, A, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_B, B, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_alpha, alpha, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_K, K, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_Kb, Kb, Nr*sizeof(double), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_lambda, lambda, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_U, U, Nr*sizeof(double), cudaMemcpyHostToDevice);

//QUAMTUN

cudaMemcpy(cuda_u_nodos,  u_nodos,      Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_u_nodos_p1,  u_nodos_p1,   Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_dr_u,     dr_u,         Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_pi,       pi,           Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

//SEMICLASSIC

cudaMemcpy(cuda_SA, SA, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_SB, SB, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_ja, ja, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_rho, rho, Nr*sizeof(double), cudaMemcpyHostToDevice);



//RK
double *RK_C;
cufftDoubleComplex *RK_Q;

cudaMalloc((void **)&RK_C, consta_var * Nr*sizeof(double ));

cudaMalloc((void **)&RK_Q, 3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));

double *temp_array, *temp_fluct;
cudaMalloc((void **)&temp_array, 5 * Nl*Nr*sizeof(double));
cudaMalloc((void **)&temp_fluct, 5 * Nr*sizeof(double));

//Evolution...
time1 = (double) clock();            /* get initial time */
time1 = time1 / CLOCKS_PER_SEC; 

    int size_cuda;
    if (Nr > Nk*Nl){
      size_cuda = Nr;
    }
    else{
      size_cuda = Nk*Nl;
    }

    int thread=64;
    dim3 bloque(thread);
    dim3 grid((int)ceil((float)(Nr*Nk*Nl)/thread));


    double k_min = consta_pi/15.0;

printf("thread = %d , block = %d , k_min = %lf ", thread , (int)ceil((float)(Nr*Nk*Nl)/thread) , k_min);
  for (int t = 0 ; t < 5   ; t++ ){

    evolucion_var<<< grid , bloque >>>( cuda_phi , cuda_chi , cuda_PI, cuda_A, cuda_B, cuda_alpha, cuda_Da, cuda_Db, cuda_K, cuda_Kb, cuda_lambda,
                                        cuda_U, cuda_u_nodos, cuda_u_nodos_p1, cuda_pi, cuda_dr_u, cuda_SA, cuda_SB, cuda_ja, cuda_rho, Nr, Nk, Nl, dr, k_min, dt, t, RK_C, RK_Q, 1);

    cudaDeviceSynchronize();

    for (int l = 0 ; l < Nl; l++ ){

      fluctuation<<<  (int)ceil((float)(Nr)/thread) , bloque>>>( cuda_u_nodos, cuda_u_nodos_p1, cuda_pi, temp_array, temp_fluct, dr, k_min,dt, Nr, Nk, Nl ,l,t);
      cudaDeviceSynchronize();

    }

    cudaDeviceSynchronize();
    suma_fluct<<< (int)ceil((float)(Nr)/thread) , bloque>>>(temp_array, Nl,Nr);
    cudaDeviceSynchronize();

    stress_energy<<< (int)ceil((float)(Nr)/thread) , bloque>>>(cuda_SA, cuda_SB, cuda_rho, cuda_ja, cuda_phi, cuda_A, cuda_B, cuda_alpha, temp_array,dr,dt,Nr,Nk, Nl,t);
    cudaDeviceSynchronize();
    cambio_u<<< grid , bloque >>>(cuda_u_nodos, cuda_u_nodos_p1, Nr,Nk,Nl);
    cudaDeviceSynchronize();

  }
timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
printf("The elapsed time is %lf seconds, GPU\n", timedif);

cudaMemcpy(phi, cuda_phi, Nr*Nt*sizeof(double), cudaMemcpyDeviceToHost);
/*
  for(int r=0; r<Nr; r++){
    printf("phi_%d : %.15f\n",r,phi[5*Nr + r]);
  }
*/
cudaError_t err = cudaGetLastError();
printf("Error: %s\n",cudaGetErrorString(err));

guardar_salida_phi(phi,Nr,Nt);


  free(phi);free(chi);free(PI);free(K);free(Kb);free(U);free(A);free(B);free(alpha);free(lambda);

}

// Compilar con : nvcc quan.cu -o quan.cux -Xcompiler /openmp


//nota: sacar el puntero en las funcinoes, si quiero optimiazr, solo dejar el phi + dt a_ij K en la funciones  por ejemplo en vez de punteros en todas las funciones.
