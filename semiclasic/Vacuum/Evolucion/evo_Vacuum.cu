#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#include <cufft.h>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>

#include <cuda_runtime.h>



#define consta_pi 3.1415926535897932384626433832795
#define consta_var 12
#define M 20.0



//Derivate
#define order 11
#define host_order 3

#define half_order_right (int)(order+1)/2
#define half_order_left (int)(order-1)/2

//GRID
#define Origin 0

//Parameter Scalar field
#define amplitude 0.0
#define width 1.5

//Parameter Quatum
#define Nk 50
#define Nl 50


//coeficientes de las diferencias finitas y RK4 para el host y el device

double host_coefficient_adelantada[3] =  {-1.5 ,2.0 ,-0.5};
double host_coefficient_centrada[3]  = {-0.5, 0.0, 0.5};
double host_coefficient_atrasada[3] = {1.5, -2.0, 0.5};
double  host_b_i[4] = {1.0/6.0 , 1.0/3.0 , 1.0/3.0 , 1.0/6.0};
double host_c_i[4] = {0.0 , 0.5 , 0.5 , 1.0};
double host_a_ij[4] = {0.0 , 0.5 , 0.5 , 1.0};

//__device__ double coefficient_adelantada[3] = {-1.5 ,2.0 ,-0.5};
//__device__ double coefficient_adelantada_cuarto[5] = {-25 ,48 ,-36, 16,-3};

//__device__ double coefficient_centrada[3] = {-0.5, 0.0, 0.5};
//__device__ double coefficient_atrasada[3] = {1.5, -2.0, 0.5};

__device__ double coefficient_adelantada[5] = {-25.0/12.0 ,4.0, -3.0, 4.0/3.0 ,-1.0/4.0};
//__device__ double coefficient_centrada[5] = {1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0,-1.0/12.0};
//__device__ double coefficient_centrada[7] = {-1.0/60.0, 3.0/20.0,-3.0/4.0, 0.0, 3.0/4.0,-3.0/20.0, 1.0/60.0};
__device__ double coefficient_centrada[11] ={-7.93650794e-04,  9.92063492e-03, -5.95238095e-02,  2.38095238e-01,
  -8.33333333e-01,  0.00000000e+00,  8.33333333e-01, -2.38095238e-01,
   5.95238095e-02, -9.92063492e-03,  7.93650794e-04,
};
__device__ double coefficient_atrasada[5] = {25.0/12.0 ,-4.0, 3.0, -4.0/3.0 ,1.0/3.0};

__device__ double b_i[4] = {1.0/6.0 , 1.0/3.0 , 1.0/3.0 , 1.0/6.0};
__device__ double c_i[4] = {0.0 , 0.5 , 0.5 , 1.0};
__device__ double a_ij[4] = {0.0 , 0.5 , 0.5 , 1.0};


//guardado de diferentes salidas.
void guardar_salida_rho(double *data,int Nr, int T) {
  FILE *fp = fopen("rho.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_SA(double *data,int Nr, int T) {
    FILE *fp = fopen("SA.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_ja(double *data,int Nr, int T) {
    FILE *fp = fopen("ja.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_SB(double *data,int Nr, int T) {
    FILE *fp = fopen("SB.dat", "wb");
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
void guardar_salida_psi(double *data,int Nr, int T) {
  FILE *fp = fopen("dr_campo_escalar.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
/*
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
*/


//Absorbente
double absorbente(double *g,int idx,double dr,double dt,int Nr){
    double dgdt, c;
    c=1.0;
    dgdt=-1.0*c*(3.0*g[idx]-4.0*g[idx-1] + g[idx-2])/(2.0*dr);

    return g[idx] + dgdt*dt;
}

//Kreiss_Oliger
__device__ double Kreiss_Oliger(double *G,int idx,int Nr,double epsilon){

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

__device__ cufftDoubleComplex Kreiss_Oliger_Cx(cufftDoubleComplex *G,int idx,int Nr,double epsilon){
  cufftDoubleComplex Z;
  int id = idx%Nr;
  if(id==0){
    Z.x = epsilon*(G[2].x-2.0*G[1].x+G[0].x);
    Z.y = epsilon*(G[2].y-2.0*G[1].y+G[0].y);

  }
  else if(id==1){
    Z.x = epsilon*(G[3].x-3.0*G[2].x+3.0*G[1].x-G[0].x);
    Z.y = epsilon*(G[3].y-3.0*G[2].y+3.0*G[1].y-G[0].y);
  }
  else if(id==Nr-1){
    Z.x = epsilon*(G[Nr-3].x-2.0*G[Nr-2].x+G[Nr-1].x);
    Z.y = epsilon*(G[Nr-3].y-2.0*G[Nr-2].y+G[Nr-1].y);
  }
  else if(id==Nr-2){
    Z.x = epsilon*(-G[Nr-4].x+3.0*G[Nr-3].x-3.0*G[Nr-2].x+G[Nr-1].x);
    Z.y = epsilon*(-G[Nr-4].y+3.0*G[Nr-3].y-3.0*G[Nr-2].y+G[Nr-1].y);
  }
  else{
    Z.x = epsilon*(G[idx+2].x - 4.0*G[idx+1].x + 6.0*G[idx].x - 4.0*G[idx-1].x+G[idx-2].x);
    Z.y = epsilon*(G[idx+2].y - 4.0*G[idx+1].y + 6.0*G[idx].y - 4.0*G[idx-1].y+G[idx-2].y);
  }
  return Z;
}




//Stress_Energy clasicc
void calculate_rho(double *rho,double *PI,double *psi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){
    rho[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + psi[idx]*psi[idx])/(2.0*A[idx]);}

}
void calculate_ja(double *ja, double *PI, double *psi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){

    ja[idx] = -PI[idx]*psi[idx] / (sqrt(A[idx])*B[idx]);}

}
void calculate_SA(double *SA, double *PI, double *psi, double *A, double *B, int Nr){
  for(int idx=0;idx<Nr;idx++){
    SA[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + psi[idx]*psi[idx])/(2.0*A[idx]);}

}
void calculate_SB(double *SB, double *PI, double *psi, double *A, double *B, int Nr ){
  for(int idx=0;idx<Nr;idx++){

  SB[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) - psi[idx]*psi[idx])/(2.0*A[idx]);}

}
//conservacion de rho, para verificar que no cambie en el espacio
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
__device__ double derivate( double *f, double h, int N, int idx, int symmetric ){
  double temp,sym;

    temp=0.0;

    if (idx < (int)(order-1)/2){
      if (symmetric==0){
        sym=1.0;
      }
      else if (symmetric == 1){
        sym=-1.0;
      }

      for (int m=0; m < half_order_right + idx ; m++){

          temp += coefficient_centrada[m + (int)(order-1)/2 -idx ]*(f[m]);


        }
      for (int m=(int)(order-1)/2 - idx; m > 0 ; m--){

          temp += sym*coefficient_centrada[ (int)(order-1)/2 - idx - m  ]*(f[m]);

        }
      
    }
    else if (idx > N-(int)(order-1)/2-1 && idx<N-1){
        temp = (0.5*f[idx+1] - 0.5*f[idx-1]);
      
    }
    else if (idx =N-1){
      temp = (f[idx] - f[idx-1]);
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*f[idx-(int)(order-1)/2+m];
      }
    }
    temp=temp/h;
    return temp;
}

//Derivada considerando el runge_Kutta, la variable symetric se ocupa para paara considerar si la funcion es par (derivada = 0) o impar (funcion = 0)
__device__ double derivate_RK( double *f,double *Kf, int idx,int s, int id_RK, double dr,double dt, int Nr, int symmetric){
  double temp ;
  double sym;
  temp=0.0;
    if (idx < (int)(order-1)/2){
      if (symmetric==0){
        sym=1.0;
      }
      else if (symmetric == 1){
        sym=-1.0;
      }

      for (int m=0; m < half_order_right + idx ; m++){

          temp += coefficient_centrada[m + (int)(order-1)/2 -idx ]*(f[ m] + dt*a_ij[s]*Kf[ m]);


        }
      for (int m=(int)(order-1)/2 - idx; m > 0 ; m--){

          temp += sym*coefficient_centrada[ (int)(order-1)/2 - idx - m  ]*(f[m] + dt*a_ij[s]*Kf[m]);

        }
      
    }
    else if (idx > Nr-(int)(order-1)/2-1 && idx<Nr-1){
      temp = (0.5*(f[idx+1] + dt*a_ij[s]*Kf[id_RK+1]) - 0.5*(f[idx+1] + dt*a_ij[s]*Kf[id_RK+1]));
    }
    else if (idx =Nr-1){
      temp = ((f[idx] + dt*a_ij[s]*Kf[id_RK]) - (f[idx-1] + dt*a_ij[s]*Kf[id_RK-1]));
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m] + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m]);
      }
    }
    return temp/dr;
}
//Derivada compleja considerando el runge_Kutta

__device__ cufftDoubleComplex derivate_complex_RK( cufftDoubleComplex *f,cufftDoubleComplex *Kf, int idx,int s, int id_RK, int idr, double dr,double dt, int Nr, int l,int symmetric){
  cufftDoubleComplex temp ;
  double sym;
  temp.x = 0.0;
  temp.y = 0.0;

  if (idr < (int)(order-1)/2){
    if (symmetric==0){
      sym=1.0;
    }
    else if (symmetric == 1){
      sym=-1.0;
    }

    for (int m=0; m < half_order_right + idr ; m++){

        temp.x += coefficient_centrada[m + (int)(order-1)/2 -idr ]*(f[ idx + m - idr ].x + dt*a_ij[s]*Kf[ id_RK + m - idr ].x);
        temp.y += coefficient_centrada[m + (int)(order-1)/2 -idr ]*(f[ idx + m - idr ].y + dt*a_ij[s]*Kf[ id_RK + m - idr ].y);

      }
    for (int m=(int)(order-1)/2 - idr; m > 0 ; m--){

        temp.x += sym*coefficient_centrada[ (int)(order-1)/2 - idr - m  ]*(f[ idx + m - idr ].x + dt*a_ij[s]*Kf[ id_RK + m - idr ].x);
        temp.y += sym*coefficient_centrada[ (int)(order-1)/2 - idr - m  ]*(f[ idx + m - idr ].y + dt*a_ij[s]*Kf[ id_RK + m - idr ].y);


      }
    
  }
    /*else if (idx<(int)(order-1)/2 && idx>0){
      for (int m=0;m<order;m++){
        temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }*/

    else if (idr > Nr-(int)(order-1)/2-1 && idr<Nr-1){
      temp.x = (0.5*(f[idx+1].x + dt*a_ij[s]*Kf[id_RK+1].x) - 0.5*(f[idx+1].x + dt*a_ij[s]*Kf[id_RK+1].x));
      temp.y = (0.5*(f[idx+1].y + dt*a_ij[s]*Kf[id_RK+1].y) - 0.5*(f[idx+1].y + dt*a_ij[s]*Kf[id_RK+1].y));

    }
    else if (idr =Nr-1){
      temp.x = ((f[idx].x + dt*a_ij[s]*Kf[id_RK].x) - (f[idx-1].x + dt*a_ij[s]*Kf[id_RK-1].x));
      temp.y = ((f[idx].y + dt*a_ij[s]*Kf[id_RK].y) - (f[idx-1].y + dt*a_ij[s]*Kf[id_RK-1].y));

    }

    else{
      for (int m=0;m<order;m++){
        temp.x += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].x + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m].x);
        temp.y += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m].y + dt*a_ij[s]*Kf[id_RK-(int)(order-1)/2+m].y);

      }
    }
    temp.x =temp.x/dr;
    temp.y /=dr;



    return temp;
}

//Metric fields...
__device__ double initial_phi(double radio){
  return amplitude*exp(- pow(radio/width,2));
}

__device__ double initial_psi(double radio){
  return -2.0* radio/pow(width,2) *amplitude*exp(- pow(radio/width,2));
}
__global__ void metric_initial(double *phi, double *psi, double *PI, double *A, double *B, double *alpha, double *Da,
                                double *Db, double *K, double *Kb, double *lambda, double *U, double *cosmological_constant,
                                int Nr, double dr){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double r;
  if(idx<Nr){
    r = idx*dr;
    //Initial matters fields
    phi[idx] = initial_phi(r);
    psi[idx] = initial_psi(r);
    PI[idx] = 0.0; // Si se escoge otro valor para pi, se tiene que rescalar con *sqrt(A), debido que A no vale 1 en todo el espacio.
    ///Initial metrics fields
    A[idx]=1.0; //Valor provisorio para A, con el fin de tener A=1.0 en el origen y poder calcular la constante cosmologica, usando A_RK calculo nuevamente A
    B[idx]=1.0;
    alpha[idx]=1.0;
    Da[idx]=0.0;
    Db[idx]=0.0;
    K[idx]=0.0;
    Kb[idx]=0.0;

  }
}



__device__ double initial_A(double A, int i, double r, double cosmological_constant, int s, double dr, int Nr){
  double dr_psi, sol;
  double rs;

  if (i < (int)(order-1)/2){
    for (int m=0; m < half_order_right + i ; m++){
      rs= r + dr*m;
      dr_psi += coefficient_centrada[m + (int)(order-1)/2 -i ]*initial_psi(rs);
      }
    for (int m=(int)(order-1)/2 - i; m > 0 ; m--){
      rs= r + dr*m;
      dr_psi += -1.0*coefficient_centrada[ (int)(order-1)/2 - i - m  ]*initial_psi(rs);
      } 
  }
  else if (i > Nr-(int)(order-1)/2-1){
    
    for (int m=-order+1;m<1;m++){
      rs= r + dr*m;
      dr_psi += coefficient_atrasada[-m]*initial_psi(rs);
    }
  }
  else{
    for (int m=0;m<order;m++){
      rs= r + dr*(m-(int)(order-1)/2);
      dr_psi += coefficient_centrada[m]*initial_psi(rs);
    }
  }
  dr_psi /= dr;
  if(i==0){
    sol = A*(  r*pow(dr_psi,2)*0.5 + r*A*cosmological_constant);
  }
  else{
    sol = A*( (1.0-A)/r + r*pow(dr_psi,2)*0.5 + r*A*cosmological_constant);
  }

  return sol;
}
__global__ void A_RK(double *A, double *cosmological_constant, int Nr,double dr){
  
  double K1,K2,K3,K4;
  double radio;

  A[0]=1.0;

  for (int i=0; i<Nr-1 ; i++){
    radio = i*dr;

    K1 = initial_A(A[i] , i, radio, cosmological_constant[0],0,dr, Nr);
    
    K2 = initial_A(A[i] + dr*a_ij[1]*K1, i, radio + c_i[1]*dr, cosmological_constant[0],1,dr, Nr);

    K3 = initial_A(A[i] + dr*a_ij[2]*K2, i, radio + c_i[2]*dr, cosmological_constant[0],2,dr, Nr);

    K4 = initial_A(A[i] + dr*a_ij[3]*K3, i, radio + c_i[3]*dr, cosmological_constant[0],3,dr, Nr);

  A[i+1]=A[i] + dr*(b_i[0]*K1 +b_i[1]*K2 +b_i[2]*K3 +b_i[3]*K4);
    printf("A[%d] = %.10f\n",i,A[i]);
  }
}
__global__ void initial_lambda(double *lambda, double *A, double *B, double dr, int Nr){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;

  if(idx<Nr){
    radio = idx*dr;
    if(idx==0){
      lambda[0]=0.0;
    }
    else{
      lambda[idx] = (1.0 - A[idx]/B[idx])/radio;
    }
  }
}
__global__ void initial_U(double *U,double *lambda, double *A, double dr, int Nr){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;
  double drA;
  if(idx<Nr){
    radio = idx*dr;
    drA = derivate(A,dr,Nr,idx,0);
      U[idx] = (drA - 4.0*lambda[idx])/A[idx];
      printf("U[%d] = %.10f\n",idx,U[idx]);
  }
}
__device__ double Kb_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda,
                         double *U,  double *RK_C , double *Sa, double *rho, double cosmology, double radio, int id, int s, double dt, double dr, int Nr, int t, int error){
  double f1,f2,f3;

  if(id==0){
    f1 = (0.5*derivate_RK(U,RK_C, id, s, 11*Nr + id,dr,dt,Nr,1) + 2.0* (derivate_RK(lambda,RK_C, id, s, 10*Nr + id,dr,dt,Nr,1))* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/
    (A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) - (derivate_RK(Db,RK_C, id, s, 7*Nr + id,dr,dt,Nr,1)) - (derivate_RK(Da,RK_C, id, s, 6*Nr + id,dr,dt,Nr,1)) - (derivate_RK(lambda,RK_C, id, s, 10*Nr + id,dr,dt,Nr,1)));

  }
  else{
    f1 = 1.0/radio*(0.5*(U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 2.0* (lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/
          (A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) - (Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ]) - (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ]) - (lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ]));

  }
    f2 = -0.5*(Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ]) * (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])  - 0.5*derivate_RK(Db,RK_C, id, s, 7*Nr + id,dr,dt,Nr,1) +0.25*(Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])*
          ((U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 4.0*(lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])) +
          (A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);

    f3 = Sa[id] - rho[id] - 2.0*cosmology;
  if(id>=Nr-3  ){

    printf("f1 Kb= %f | %d\n",f1,id);
    printf("U Kb= %f | %d\n",U[id],id);
    printf("lambda Kb= %f | %d\n",lambda[id],id);
    printf("Db Kb= %f | %d\n",Db[id],id);
    printf("Da Kb= %f | %d\n",Da[id],id);
    printf("1.0/radio Kb= %f | %d\n",1.0/radio,id);

    printf("f2 Kb= %f| %d\n",f2,id);

    printf("f3 Kb= %f| %d\n",f3,id);


  }
  if (error==0 && t==0 && id == 50){
    printf("Kb_dot == check\n");
  }



  return  (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) /(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) *(f1+f2) + (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) / 2.0 * f3;
}

__device__ double K_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda, 
                        double *U, double *RK_C , double *Sa, double *rho, double *Sb,  double cosmology, double radio, int id, int s, double dt,double dr, int Nr,  int t, int error){

  double f1,f2,f3;

  f1 = (K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 4.0*(K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ])* (Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) +
        6.0*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ])*(Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);

  if(id==0){
    f2 =  derivate_RK(Da,RK_C, id, s,6*Nr + id,dr,dt,Nr,1) + (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])+ 2.0*derivate_RK(Da,RK_C, id, s,6*Nr + id,dr,dt,Nr,1)
    -0.5*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*((U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 4.0*(lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]));
  }
  else{
    f2 =  derivate_RK(Da,RK_C, id, s,6*Nr + id,dr,dt,Nr,1) + (Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])+ 2.0*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])/radio
    -0.5*(Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*((U[id] + dt * a_ij[s] * RK_C[ 11*Nr + id ]) + 4.0*(lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ])* (B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]));

  }
  f3 =  rho[id] +Sa[id] + 2.0*Sb[id] - 2.0*cosmology;
                      
  if(id==Nr-1  ){


    printf("f3 K= %f\n",f3);


  }


  return  (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * 
          ( f1 - f2/(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) ) + 
          (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) / 2.0 * f3;
}

__device__ double lambda_dot(double *A, double *B, double *alpha, double *Db, double * K, double *Kb,  double *RK_C, double *Ja, int id, int s, double dt,double dr, int Nr,  int t, int error){

  double f1;
  
  f1  =   derivate_RK(Kb,RK_C, id, s, 9*Nr + id ,dr,dt,Nr,0) - 0.5* ( Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])*(( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 3.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) ) +
          Ja[id]/2.0;




          if (error==0 && t==0 && id == 50){
            printf("lambda_dot == check\n");
          }
        
  return 2.0*(alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * ( A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])/( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ]) * f1;

}

__device__ double U_dot(double *A, double *B, double *alpha, double *Da, double *Db, double * K, double *Kb, double *lambda, 
                        double *RK_C, double *Ja, int id, int s, double dt,double dr, int Nr,  int t, int error){
  double f1, f2;

  f1  =   derivate_RK(K,RK_C, id, s, 8*Nr + id ,dr,dt,Nr,0) +  ( Da[id] + dt * a_ij[s] * RK_C[ 6*Nr + id ])*(( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 4.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) );

  f2  =   -2.0*( ( K[id] + dt * a_ij[s] * RK_C[ 8*Nr + id ]) - 3.0*( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]) ) *(  ( Db[id] + dt * a_ij[s] * RK_C[ 7*Nr + id ])  - 
           2.0*  ( lambda[id] + dt * a_ij[s] * RK_C[ 10*Nr + id ]) * ( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])/ ( A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]) );
  


    if (error==0 && t==0 && id == 50){
      printf("U_dot == check\n");
    }
        

  return -2.0*(alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) * ( f1 + f1 + 2.0*Ja[id] );

}

//Funciones para derivada temporal de Db (Db_dot es la derivada que llama la funcion f_Db y f_Db_dot es la funcion alpha * K)
__device__ double f_Db_dot(double *Kb, double *alpha,  double *RK_C, int id, int s, double dt,int Nr,  int t, int error){



  if (error==0 && t==0 && id == 50){
    printf("f_Db_dot == check\n");
  }


  return (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) *( Kb[id] + dt * a_ij[s] * RK_C[ 9*Nr + id ]);
}
__device__ double Db_dot(double *Kb, double *alpha, int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp,sym;
  temp=0.0;

  if (idx < (int)(order-1)/2){
    if (symmetric==0){
      sym=1.0;
    }
    else if (symmetric == 1){
      sym=-1.0;
    }

    for (int m=0; m < half_order_right + idx ; m++){

        temp += coefficient_centrada[m + (int)(order-1)/2 -idx ]*(f_Db_dot(Kb,alpha,RK_C, m ,s,dt ,Nr , t,  error));

      }
    for (int m=(int)(order-1)/2 - idx; m > 0 ; m--){

        temp += sym*coefficient_centrada[ (int)(order-1)/2 - idx - m  ]*(f_Db_dot(Kb,alpha,RK_C, m ,s,dt ,Nr , t,  error));

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

//Matter fields

//Funciones para derivada temporal de PI
__device__ double f_PI_dot(double *psi, double *PI, double *A, double *B,double *alpha, double *RK_C, int id, int s, double dt,double dr,  int Nr,  int t, int error){
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
    

  return (psi[id] + dt * a_ij[s] * RK_C[ 1*Nr + id ]) * (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ]) *( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ])*
          radio*radio/(sqrt(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ]));
}
__device__ double PI_dot(double *psi, double *PI, double *A, double *B,double *alpha, double r2, int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp, sym;
  temp=0.0;
  if (idx < (int)(order-1)/2){
    if (symmetric==0){
      sym=1.0;
    }
    else if (symmetric == 1){
      sym=-1.0;
    }

    for (int m=0; m < half_order_right + idx ; m++){

        temp += coefficient_centrada[m + (int)(order-1)/2 -idx ]*(f_PI_dot(psi, PI,A,B,alpha,RK_C, m ,s,dt,dr ,Nr , t , error));

      }
    for (int m=(int)(order-1)/2 - idx; m > 0 ; m--){

        temp += sym*coefficient_centrada[ (int)(order-1)/2 - idx - m  ]*(f_PI_dot(psi, PI,A,B,alpha,RK_C, m ,s,dt,dr ,Nr , t , error));

      }
    
  }
    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_PI_dot(psi, PI,A,B,alpha,RK_C,idx+m,s,dt,dr ,Nr , t , error));
      }
    }
    else if (idx ==1){
      for (int m=0;m<order;m++){
        temp += coefficient_atrasada[m]*(f_PI_dot(psi, PI,A,B,alpha,RK_C,idx+m,s,dt,dr ,Nr , t , error));
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_PI_dot(psi, PI,A,B,alpha,RK_C,idx-(int)(order-1)/2+m,s,dt,dr ,Nr , t , error));
      }
    }


    if (error==0 && t==0 && idx == 50){
      printf("PI_dot == check\n");
    }
  
    return temp/dr*(1.0/(r2));;
}

//Funciones para derivada temporal de psi
__device__ double f_psi_dot(double *PI, double *A, double *B,double *alpha, double *RK_C, int id, int s, double dt, int Nr,  int t, int error){


  if (error==0 && t==0 && id == 50){
    printf("f_psi_dot == check\n");
  }


  return (PI[id] + dt * a_ij[s] * RK_C[ 2*Nr + id ]) * (alpha[id] + dt * a_ij[s] * RK_C[ 5*Nr + id ])/
          (sqrt(A[id] + dt * a_ij[s] * RK_C[ 3*Nr + id ])*( B[id] + dt * a_ij[s] * RK_C[ 4*Nr + id ]));
}
__device__ double psi_dot(double *PI, double *A, double *B,double *alpha,int idx, int s, double dr, double dt, double *RK_C, int symmetric,int Nr, int t, int error){
  double temp,sym;
  temp=0.0;

  if (idx < (int)(order-1)/2){
    if (symmetric==0){
      sym=1.0;
    }
    else if (symmetric == 1){
      sym=-1.0;
    }

    for (int m=0; m < half_order_right + idx ; m++){

        temp += coefficient_centrada[m + (int)(order-1)/2 - idx ]*(f_psi_dot(PI,A,B,alpha,RK_C, m ,s,dt, Nr , t , error));

      }
    for (int m=(int)(order-1)/2 - idx; m > 0 ; m--){

        temp += sym*coefficient_centrada[ (int)(order-1)/2 - idx - m  ]*(f_psi_dot(PI,A,B,alpha,RK_C, m ,s,dt, Nr , t , error));

      }
    
  }

    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f_psi_dot(PI,A,B,alpha,RK_C,idx+m,s,dt, Nr , t , error));
      }
    }

    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f_psi_dot(PI,A,B,alpha,RK_C,idx-(int)(order-1)/2+m,s,dt,Nr, t , error ));
      }
    }



    if (error==0 && t==0 && idx == 50){
      printf("psi_dot == check\n");
    }
  
    return temp/dr;
}

//Cuantico


//Funciones para derivada temporal de dr_u (derivada espacial de los nodos u)
__device__ cufftDoubleComplex f_dr_u_dot(cufftDoubleComplex *pi, double *A, double *B,double *alpha, double *RK_C, cufftDoubleComplex *RK_Q, int id_nodo, int idr , int space, int s, double dt ,int Nr, int t, int error, int g){

  cufftDoubleComplex Z;

  Z.x= (pi[g*space + id_nodo].x + dt * a_ij[s] * RK_Q[ g*3*space + 2*space + id_nodo ].x) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));
  Z.y= (pi[g*space + id_nodo].y + dt * a_ij[s] * RK_Q[ g*3*space + 2*space + id_nodo ].y) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));


  if (error==0 && t==0 && id_nodo == 50){
    printf("f_dr_u_dot == check\n");
  }


  return Z;
}
__device__ cufftDoubleComplex dr_u_dot(cufftDoubleComplex *pi, double *A, double *B,double *alpha,int id_nodo, int idr,int space, int s, double dr, double dt, double *RK_C, cufftDoubleComplex *RK_Q, 
                                      int symmetric,int Nr,  int t, int error, int g){
  cufftDoubleComplex temp;
  double sym;
  temp.x=0.0;
  temp.y=0.0;


    if (idr < (int)(order-1)/2){
      if (symmetric==0){
        sym=1.0;
      }
      else if (symmetric == 1){
        sym=-1.0;
      }
  
      for (int m=0; m < half_order_right + idr ; m++){
  
        temp.x += coefficient_centrada[m + (int)(order-1)/2 -idr ]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m -idr , m , space, s,dt, Nr , t , error, g)).x;
        temp.y += coefficient_centrada[m + (int)(order-1)/2 -idr ]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m -idr , m , space, s,dt, Nr, t , error, g )).y;


        }
      for (int m=(int)(order-1)/2 - idr; m > 0 ; m--){
  
        temp.x += sym*coefficient_centrada[ - m + (int)(order-1)/2 -idr ]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m -idr , m , space, s,dt, Nr , t , error, g)).x;
        temp.y += sym*coefficient_centrada[ - m + (int)(order-1)/2 -idr ]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m -idr , m , space, s,dt, Nr, t , error, g )).y;  
        }
      
    }

      else if (idr > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp.x += coefficient_atrasada[-m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m , idr + m, space , s,dt, Nr, t , error, g)).x;
        temp.y += coefficient_atrasada[-m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo + m , idr + m, space , s,dt, Nr, t , error, g)).y;

      }
    }
    else{
      for (int m=0;m<order;m++){
        temp.x += coefficient_centrada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo-(int)(order-1)/2+m , idr-(int)(order-1)/2+m , space , s,dt, Nr , t , error, g)).x;
        temp.y += coefficient_centrada[m]*(f_dr_u_dot(pi,A,B,alpha,RK_C,RK_Q, id_nodo-(int)(order-1)/2+m , idr-(int)(order-1)/2+m , space , s,dt, Nr , t , error, g)).y;

      }
    }
 

    temp.x = temp.x/dr;
    temp.y = temp.y/dr;

    return temp;
}


//Funciones para derivada temporal de pi (derivada temporal de los nodos u)

__device__ double f_metric_dot(double *A, double *B,double *alpha, double *RK_C, int idr , int s, double dt, int Nr,  int t,int error){

  return  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ]));
}

__device__ double metric_pi_dot( double *A, double *B,double *alpha, int idr, int s, double dr, double dt, double *RK_C, int symmetric,int Nr,  int t, int error){
  double temp,sym;
  temp=0.0;

  if (idr < (int)(order-1)/2){
    if (symmetric==0){
      sym=1.0;
    }
    else if (symmetric == 1){
      sym=-1.0;
    }

    for (int m=0; m < half_order_right + idr ; m++){

        temp += coefficient_centrada[m + (int)(order-1)/2 - idr ]*(f_metric_dot(A,B,alpha,RK_C, m,s,dt, Nr, t , error));

      }
    for (int m=(int)(order-1)/2 - idr; m > 0 ; m--){

        temp += sym*coefficient_centrada[ (int)(order-1)/2 - idr - m  ]*(f_metric_dot(A,B,alpha,RK_C, m,s,dt, Nr, t , error));

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

//evolucion nodo pi

//Nota: alparecer al disminuir dk la presicion de esta funcion se ve comprometida, puede ser debido a tener bajos nodos, no se aproxima como corresponde la constante cosmologica
//      lo que lleva a una mala aproximacion de A.
__device__ cufftDoubleComplex f_pi_dot(cufftDoubleComplex *u, cufftDoubleComplex *dr_u,double *A, double *B,double *alpha, double *lambda, double *RK_C, cufftDoubleComplex *RK_Q, 
                                      double radio,  int id_nodo, int idr , int space, int s, int k, int l, double dt,double dr, int Nr,  int t, int error, int g, double mass){

  double f1, f2, f3, f4;
  cufftDoubleComplex Z;

  // Aproximacion dr*i -> 0, ~~derivada
  // Nota : en r=dr tambien hay problemas, deberia aproximar tambien ese dr a una derivada?
  if(idr == Origin){
      f1 =   (dr_u[g*space + id_nodo].x + dt*a_ij[s] * RK_Q[ g*3*space + space + id_nodo].x) ;

    f2 = (2.0*l+3.0) * derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo, s , g*3*space + space + id_nodo, idr, dr, dt,Nr, l,1).x;

    f3 = l*(l+1) *derivate_RK(lambda,RK_C,idr , s, 10*Nr + idr, dr,dt,Nr,1)*(u[g*space + id_nodo].x + dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].x);

  }
  else{
      f1 =  l/radio * (u[g*space + id_nodo].x + dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].x) + (dr_u[g*space + id_nodo].x + dt*a_ij[s] * RK_Q[ g*3*space + space + id_nodo].x) ;

    f2 = (2.0*l+2.0)/radio * (dr_u[g*space + id_nodo].x + dt* a_ij[s] * RK_Q[ g*3*space + space + id_nodo].x)  + derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo, s , g*3*space + space + id_nodo, idr, dr, dt,Nr, l,1).x;

    f3 = l*(l+1)/radio *(lambda[idr] + dt * a_ij[s] * RK_C[ 10*Nr + idr ])*(u[g*space + id_nodo].x + dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].x);

  }
  f4 = mass*mass*u[g*space + id_nodo].x*u[g*space + id_nodo].x;

  if(idr==0 && k ==19 && l ==3 && g==0 ){
    printf("g : %d | idr= %d\n", g, idr);
    printf("cte = %f | %f | %f | %f\n",(2.0*l+2.0)/radio,(2.0*l+2.0)/((idr+1)*dr), (2.0*l+2.0)/((idr+2)*dr), (2.0*l+2.0)/((idr+3)*dr));

    printf("u =  %f | %f | %f | %f \n",u[g*space + id_nodo].x, u[g*space + id_nodo + 1].x, u[g*space + id_nodo + 2].x, u[g*space + id_nodo + 3].x );
    printf("dr =  %f | %f | %f | %f\n",dr_u[g*space + id_nodo].x, dr_u[g*space + id_nodo + 1].x, dr_u[g*space + id_nodo + 2].x, dr_u[g*space + id_nodo + 3].x  );

    printf("ure =  %f | %f | %f | %f\n",dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].x, dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo + 1].x, dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo + 2 ].x, dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo + 3].x);
    printf("drure =  %f | %f | %f | %f\n",dt*a_ij[s] * RK_Q[ g*3*space + 1*space + id_nodo].x, dt*a_ij[s] * RK_Q[ g*3*space + 1*space + id_nodo + 1].x, dt*a_ij[s] * RK_Q[ g*3*space + 1*space + id_nodo + 2 ].x, dt*a_ij[s] * RK_Q[ g*3*space + 1*space + id_nodo + 3].x);
    printf(" dr_u_rk:  %f | %f | %f | %f\n", derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo, s , g*3*space + space + id_nodo, idr, dr, dt,Nr, l,1).x ,derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo +1, s , g*3*space + space + id_nodo +1, idr +1, dr, dt,Nr, l,1).x, derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo +2, s , g*3*space + space + id_nodo +2, idr+2, dr, dt,Nr, l,1).x , derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo  +3, s , g*3*space + space + id_nodo +3, idr +3, dr, dt,Nr, l,1).x);
    printf("nan? %f\n",sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ]));
    printf("nan? %f\n",alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]);
    printf("nan? %f\n",B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]);
    printf("nan mul %f\n",(alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])));


  }
if(idr<5 && k ==19 && l ==3 && g ==0){
  printf("f1 =  %f | %d\n",f1,idr);
  printf("metric =  %f | %d\n",metric_pi_dot(A,B,alpha,idr,s,dr,dt, RK_C ,0,Nr, t , error),idr);
  printf("f2 =  %f | %d\n",f2,idr);
  printf("f3 =  %f | %d\n",f3, idr);
  printf("f4 =  %f | %d\n",f4, idr);
}

  Z.x = metric_pi_dot(A,B,alpha,idr,s,dr,dt, RK_C ,0,Nr, t , error) * f1 +  
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f2 +
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f3 - f4;

  if(idr<10 && k ==19 && l ==3 &&g==0){
    printf("Z = %f | %d\n",Z.x,idr);


  }
  if(idr == Origin){
      f1 =   (dr_u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[ g*3*space + space + id_nodo].y) ; //deberia ser cero

    f2 = (2.0*l+3.0) * derivate_complex_RK(dr_u, RK_Q, g*space +  id_nodo, s , g*3*space + space + id_nodo, idr, dr, dt,Nr, l,1).y;

    f3 = l*(l+1) *derivate_RK(lambda,RK_C,idr , s, 10*Nr + idr, dr,dt,Nr,1)*(u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].y);

  }
  else{
    f1 =  l/radio * (u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[g*3*space +  0*space + id_nodo].y) + (dr_u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[ g*3*space + space + id_nodo].y) ;
    f2 = (2.0*l+2.0)/radio * (dr_u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[ g*3*space + space + id_nodo].y)  + derivate_complex_RK(dr_u, RK_Q, g*space + id_nodo, s ,g*3*space + space + id_nodo, idr, dr, dt,Nr,l,1).y;
    f3 = l*(l+1)/radio *(lambda[idr] + dt * a_ij[s] * RK_C[ 10*Nr + idr ])*(u[g*space + id_nodo].y + dt*a_ij[s] * RK_Q[ g*3*space + 0*space + id_nodo].y);
  }
  
  f4 = mass*mass*u[g*space + id_nodo].y*u[g*space + id_nodo].y; // falta agregar A B y alpha 

  Z.y = metric_pi_dot(A,B,alpha,idr,s,dr,dt, RK_C ,0,Nr , t ,error) * f1 +  
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f2 +
  (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ]) * ( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]) /(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])) * f3 - f4;

  if (error==0 && t==0 && id_nodo == 50){
    printf("f_pi_dot == check\n");
  }

  return  Z;
}

__device__ double cosmological(double *A, double *B, double *alpha, double *Da, double *Db, 
  double * K, double *Kb, double *lambda, double *U, double *rho,  double dr,double dt){
    double dr_Db;
    double Gtt;
    dr_Db = (2.0*Db[1] -0.5*Db[2])/dr ;
    Gtt =  dr_Db + (lambda[0] +Db[0] -U[0] -4.0*lambda[0]*B[0]/A[0])/(dr*0.5) - Db[0]*(0.25*Db[0] + 0.5*U[0] + 2.0*lambda[0]*B[0]/A[0]);
    Gtt = Gtt/A[0] - Kb[0] * (2.0*K[0] - 3.0*Kb[0]) ;
    return  Gtt;

  }



//Evolucion...
//( phi , psi , PI, A, B, alpha, Da, Db, K, Kb, lambda, U, u_nodos, pi, dr_u, Nr, Nk, Nl, dr, dk, dt, t);
__global__ void Runge_kutta_classic(double *phi, double *psi, double *PI, double *A, double *B, double *alpha, double *Da, double *Db, 
                              double * K, double *Kb, double *lambda, double *U, 
                              double *SA, double *SB, double *ja, double *rho, double *cosmological_constant, 
                              int Nr,   double dr, double dk, double dt, int t,  double *RK_C,double *RK_C_temp,double *RK_C_reduce, int error,int s){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double radio;
    double epsilon = dr;
                              
    //double M = Nk*dk;
    //if(t==0){
        //rho[Nr] = rho[0];
      //}
      //cosmology = -0.5*1.0*log(1.0/M); 
      //cosmology=0.0;
      //cosmology = cosmological( A, B, alpha, Da, Db, K, Kb, lambda, U,rho ,dr,dt)/(alpha[0]*alpha[0]);
      //if(idx==0 && s==0){
        //cosmology = rho[0];
      //rho[0] = 0.0;
      //}

      // K de Runge kutta temporal


    //inicializar los K de la grilla, RK_Clasico tiene dimensiones 12*Nr , 12 variables y Nr numero de puntos de la grilla

    
    //Dimensiones de RK cuantico es 3*Nk*Nl*Nr, 3 variables, Nk numero de nodos k (radial) , Nl numero de nodos l (angulares) y Nr


    //interacion de Runge kutta
      __syncthreads();

      //                ********* CLASSIC *********

      if (idx < Nr){
        if(s==0){
          for( int i=0 ; i<12 ; i++){
            RK_C[ i*Nr + idx ] = 0.0;
          }
        }
        if ( idx==0 ){
          radio = idx*dr +epsilon;
        }
        else{
          radio = idx*dr ;
        }
       //phi
        RK_C_temp[0*Nr + idx] = (PI[idx] + dt * a_ij[s] * RK_C[ 2*Nr + idx ]) * (alpha[idx] + dt * a_ij[s] * RK_C[ 5*Nr + idx ])/(sqrt(A[idx] + dt * a_ij[s] * RK_C[ 3*Nr + idx ])*( B[idx] + dt * a_ij[s] * RK_C[ 4*Nr + idx ]));
        
        //psi
        RK_C_temp[1*Nr + idx] = psi_dot( PI, A, B, alpha, idx, s, dr, dt, RK_C, 0,Nr, t , error);
        //PI
        RK_C_temp[2*Nr + idx] =  PI_dot( psi, PI, A, B, alpha, radio*radio , idx, s, dr, dt, RK_C, 1, Nr, t , error);
        //A
        RK_C_temp[3*Nr + idx] = -2.0*(alpha[idx] + dt * a_ij[s] * RK_C[5*Nr + idx] ) * (A[idx]+ dt * a_ij[s] * RK_C[3*Nr + idx] )
                                * ((K[idx]+ dt * a_ij[s] * RK_C[8*Nr + idx] ) - 2.0*(Kb[idx]+ dt * a_ij[s] * RK_C[9*Nr + idx] ));
        //B
        RK_C_temp[4*Nr + idx] = -2.0*(alpha[idx] + dt * a_ij[s] * RK_C[5*Nr + idx] ) * (B[idx] + dt * a_ij[s] * RK_C[4*Nr + idx] ) *  (Kb[idx] + dt * a_ij[s] * RK_C[9*Nr + idx] );
        //alpha
        RK_C_temp[5*Nr + idx] = -2.0*(alpha[idx] + dt * a_ij[s] * RK_C[5*Nr + idx] ) * (K[idx] + dt * a_ij[s] * RK_C[8*Nr + idx] ) ;
        //Da
        RK_C_temp[6*Nr + idx] = -2.0*derivate_RK(K,RK_C, idx, s, 8*Nr + idx,dr,dt,Nr,0);
        //Db

        RK_C_temp[7*Nr + idx] = -2.0*Db_dot(Kb, alpha, idx, s, dr, dt, RK_C,0,Nr,t , error);
        //K
        
        RK_C_temp[8*Nr + idx] = K_dot( A, B, alpha, Da, Db, K, Kb, lambda, U, RK_C, SA, rho, SB, cosmological_constant[0], radio, idx,s,dt,dr,Nr, t ,error);
        //Kb
        RK_C_temp[9*Nr + idx] = Kb_dot( A, B, alpha, Da, Db, K, Kb, lambda, U, RK_C, SA, rho, cosmological_constant[0], radio, idx,s,dt,dr,Nr, t ,error);
        //lambda 

        RK_C_temp[10*Nr + idx] = lambda_dot( A, B, alpha, Db, K, Kb, RK_C, ja, idx,s,dt,dr,Nr, t ,error);
        //U
        RK_C_temp[11*Nr + idx] = U_dot( A, B, alpha, Da, Db, K, Kb,  lambda, RK_C, ja, idx,s,dt,dr,Nr, t ,error);



        //Sumar los K de RK y actualizar RK:C y RK_Q (talvez sea mejor solamente añadir RK_C y RK_Q con 4 dimensiones extra para lso k1 k2 k3 k4 y sumar todo al final).
        for( int i=0 ; i<12 ; i++){
          if(s==0){
            RK_C_reduce[i*Nr + idx]=b_i[s]*RK_C_temp[i*Nr + idx];
          }
          else{
            RK_C_reduce[i*Nr +idx] += b_i[s]*RK_C_temp[i*Nr + idx];

          }
        }
        
        if(idx==  Nr-2 && t< 10){
          printf("s: %d | t: %d\n",s,t);
          printf("phi 0,17: %.15f\n",RK_C_temp[0*Nr + idx]);
          printf("PI 0,17: %.15f\n",RK_C_temp[1*Nr + idx]);
          printf("psi 0,17: %.15f\n",RK_C_temp[2*Nr + idx]);
          printf("A 0,17: %.15f\n",RK_C_temp[3*Nr + idx]);
          printf("B 0,17: %.15f\n",RK_C_temp[4*Nr + idx]);
          printf("alpha 0,17: %.15f\n",RK_C_temp[5*Nr + idx]);
          printf("Da 0,17: %.15f\n",RK_C_temp[6*Nr + idx]);
          printf("Db 0,17: %.15f\n",RK_C_temp[7*Nr + idx]);
          printf("K past: %.15f\n",RK_C_temp[8*Nr + idx-1]);
          printf("K 0,17: %.15f\n",RK_C_temp[8*Nr + idx]);
          printf("K last: %.15f\n",RK_C_temp[8*Nr + idx+1]);

          printf("Kb past : %.15f\n",RK_C_temp[9*Nr + idx-1]);
          printf("Kb : %.15f\n",RK_C_temp[9*Nr + idx]);

          printf("Kb last : %.15f\n",RK_C_temp[9*Nr + idx+1]);

          printf("lambda 0,17: %.15f\n",RK_C_temp[10*Nr + idx]);
          printf("U 0,17: %.15f\n",RK_C_temp[11*Nr + idx]);
        }

      }

 
        __syncthreads();
   

    if(idx==Nr-2   && t%1==0 && s==0){
      printf("idx: %d y  t: %d\n",idx,t);

      printf("phi t0 : %.15f\n",phi[t*Nr + idx]);
      printf("phi t1 : %.15f\n",phi[(t+1)*Nr + idx]);

      printf("PI : %.15f\n",PI[idx]);
      printf("psi : %.15f\n",psi[idx]);
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
      printf("ja : %.15f | idx= %d | t=%d\n",ja[idx],idx,t);
      
      printf("SA : %.15f\n",SA[idx]);
      printf("SB : %.15f\n",SB[idx]);
 /*
      printf("Re(u) 0,17: %.15f\n",u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-2].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-2].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-1].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-2].x);
     printf("RKQ 0,17: %.15f\n",u[17*Nk*Nr + 0*Nr + idx].x + dt*RK_sum_Q[0].x);

      printf("Re(dr_u) : %.15f | idx = %d\n",dr_u[17*Nk*Nr + 0*Nr + idx].x,idx);
      
      printf("Re(pi)  : %.15f\n",pi[17*Nk*Nr + 0*Nr + idx].x);
      printf("Im(u) : %.15f\n",u[17*Nk*Nr + 0*Nr + idx].y);
      printf("Im(u_p1) : %.15f\n",u_p1[17*Nk*Nr + 0*Nr + idx].y);

      printf("Im(dr_u) : %.15f\n",dr_u[17*Nk*Nr + 0*Nr + idx].y);
      printf("Im(pi) : %.15f\n",pi[17*Nk*Nr + 0*Nr + idx].y);*/
     
    }/*
    if(idx%Nr==0   && idx< Nk*Nl*Nr && t<13 &&t>6){
      printf("Re(u) k:%d,l:%d = %.15f | t:%d | idx:%d\n",(int)idx/Nr%Nk, (int)idx/Nr/Nl, u_p1[((int)idx/Nr)/Nl*Nk*Nr + ((int)idx/Nr)%Nk*Nr + 0].x, t,idx);
    }*/

}





__global__ void Runge_kutta_nodos(double *A, double *B, double *alpha,double *lambda, 
                            cufftDoubleComplex *u, cufftDoubleComplex *u_p1,  cufftDoubleComplex *pi, cufftDoubleComplex *dr_u,
                            cufftDoubleComplex *u_ghost, cufftDoubleComplex *u_p1_ghost,  cufftDoubleComplex *pi_ghost, cufftDoubleComplex *dr_u_ghost,
                            int Nr,   double dr, double dt, int t,  double *RK_C, 
                            cufftDoubleComplex* RK_Q, cufftDoubleComplex* RK_Q_temp, cufftDoubleComplex* RK_Q_reduce , 
                            cufftDoubleComplex* RK_G ,cufftDoubleComplex* RK_G_temp, cufftDoubleComplex* RK_G_reduce,   int error, int s){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double epsilon = dr*0.5;
    double radio;


    //                ********* QUANTUM *********
      

   
    if (idx < Nr*Nk*Nl){
    if(s==0){
        for( int i=0 ; i<3 ; i++){
        RK_Q[ i*Nr*Nk*Nl + idx ].x = 0.0;
        RK_Q[ i*Nr*Nk*Nl + idx ].y = 0.0;

        }
    }

    int idr = idx%Nr;
    int id_nodo = (int)idx/Nr;
    cufftDoubleComplex temp_Q;
    //u
    RK_Q_temp[0*Nl*Nk*Nr + idx].x = (pi[idx].x + dt * a_ij[s] * RK_Q[ 2*Nk*Nr*Nl + idx ].x) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));
    RK_Q_temp[0*Nl*Nk*Nr + idx].y = (pi[idx].y + dt * a_ij[s] * RK_Q[ 2*Nk*Nr*Nl + idx ].y) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));

    //dr_u
    temp_Q = dr_u_dot( pi, A, B, alpha, idx , idr ,  Nk*Nr*Nl ,s, dr, dt, RK_C, RK_Q, 0,Nr, t ,error, 0);
    RK_Q_temp[1*Nl*Nk*Nr + idx].x =  temp_Q.x/dr;
    RK_Q_temp[1*Nl*Nk*Nr + idx].y =  temp_Q.y/dr;


    // indices de los nodos
    int k,l;
    k = id_nodo%Nk;
    l = (int)id_nodo/Nl;


    if ( idr==0 ){
      radio = idr*dr +epsilon;
      }
      else{
      radio = idr*dr ;
      }
  
    /*
    if(idx==300){
        printf("id_nodo = %d , k = %d , l = %d \n",id_nodo,k,l);
    }
        */

        //pi
    temp_Q = f_pi_dot( u ,dr_u , A, B, alpha, lambda ,  RK_C, RK_Q, radio,  idx, idr, Nk*Nr*Nl,s, k, l, dt, dr, Nr, t , error, 0,0.0);
    RK_Q_temp[2*Nl*Nk*Nr + idx].x =  temp_Q.x;
    RK_Q_temp[2*Nl*Nk*Nr + idx].y =  temp_Q.y;
    /* if(s==0){
        printf("idr = %d, id = %d, Rk=%lf + %lf i \n", idr, idx , RK_temp_Q[0].x,RK_temp_Q[0].y);
        
    }
    */
    //suma y actualizacion de los RK
        for( int i=0 ; i<3 ; i++){
            if(s==0){
                RK_Q_reduce[ i*Nl*Nk*Nr + idx ].x = b_i[s]*RK_Q_temp[i*Nl*Nk*Nr + idx].x;

                RK_Q_reduce[ i*Nl*Nk*Nr + idx ].y = b_i[s]*RK_Q_temp[i*Nl*Nk*Nr + idx].y;

            }
            else{
                RK_Q_reduce[ i*Nl*Nk*Nr + idx ].x += b_i[s]*RK_Q_temp[i*Nl*Nk*Nr + idx].x;

                RK_Q_reduce[ i*Nl*Nk*Nr + idx ].y += b_i[s]*RK_Q_temp[i*Nl*Nk*Nr + idx].y;

            }

        }
        if(idr==3 && l ==3 && k == 19){
          printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx ].x);
          printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -1 ].x);
          printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -2 ].x);
          printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -3 ].x);
          }
/*
        if(idx%Nr==0 && k ==19 && l==17){
        printf("k = %d, l = %d, Rku=%.15lf + %.15lf i \n", k, l , u[idx].x + dt*RK_sum_Q[0].x,RK_sum_Q[0].y);
        printf("k = %d, l = %d, Rkpi=%.15lf + %.15lf i \n", k, l , RK_sum_Q[2].x,RK_sum_Q[2].y);
        printf("k = %d, l = %d, Rkdu=%.15lf + %.15lf i \n", k, l , RK_sum_Q[1].x,RK_sum_Q[1].y);

        
    }
*/
    

    //                ********* GHOSTS *********
    for(int g = 0 ; g < 5 ; g++){
        if(s==0){
            for( int i=0 ; i<3 ; i++){
            RK_G[g*3*Nr*Nk*Nl + i*Nr*Nk*Nl + idx ].x = 0.0;
            RK_G[g*3*Nr*Nk*Nl + i*Nr*Nk*Nl + idx ].y = 0.0;

            }
        
        }
        int idr = idx%Nr;
        int id_nodo = (int)idx/Nr;
        cufftDoubleComplex temp_Q;
        //u
        RK_G_temp[ g*3*Nl*Nk*Nr + 0*Nl*Nk*Nr + idx].x = (pi_ghost[g*Nk*Nl*Nr + idx].x + dt * a_ij[s] * RK_G[ g*3*Nk*Nr*Nl + 2*Nk*Nr*Nl + idx ].x) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));
        RK_G_temp[ g*3*Nl*Nk*Nr + 0*Nl*Nk*Nr + idx].y = (pi_ghost[g*Nk*Nl*Nr + idx].y + dt * a_ij[s] * RK_G[ g*3*Nk*Nr*Nl + 2*Nk*Nr*Nl + idx ].y) * (alpha[idr] + dt * a_ij[s] * RK_C[ 5*Nr + idr ])/(sqrt(A[idr] + dt * a_ij[s] * RK_C[ 3*Nr + idr ])*( B[idr] + dt * a_ij[s] * RK_C[ 4*Nr + idr ]));

        //dr_u
        temp_Q = dr_u_dot( pi_ghost, A, B, alpha, idx , idr ,  Nk*Nr*Nl ,s, dr, dt, RK_C, RK_G, 0,Nr, t ,error, g);
        RK_G_temp[ g*3*Nl*Nk*Nr + 1*Nl*Nk*Nr + idx].x =  temp_Q.x/dr;
        RK_G_temp[ g*3*Nl*Nk*Nr + 1*Nl*Nk*Nr + idx].y =  temp_Q.y/dr;


        // indices de los nodos
        int k,l;
        k = id_nodo%Nk;
        l = (int)id_nodo/Nl;
        double mass;
        
        if (g==2 || g==0){
            mass = 1.0;
        }
        else if (g==4){
            mass = 1.0*sqrt(4.0);
        }
        else{
            mass = 1.0*sqrt(3.0); 
        }
        
        //pi
        temp_Q = f_pi_dot( u_ghost ,dr_u_ghost , A, B, alpha, lambda ,  RK_C, RK_G, radio,  idx, idr, Nk*Nr*Nl,s, k, l, dt, dr, Nr, t , error, g,mass);
        RK_G_temp[ g*3*Nl*Nk*Nr + 2*Nl*Nk*Nr + idx].x =  temp_Q.x;
        RK_G_temp[ g*3*Nl*Nk*Nr + 2*Nl*Nk*Nr + idx].y =  temp_Q.y;


        //suma y actualizacion de los RK
        for( int i=0 ; i<3 ; i++){
            if(s==0){
                RK_G_reduce[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx ].x = b_i[s]*RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].x;

                RK_G_reduce[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx ].y = b_i[s]*RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].y;

            }
            else{
                RK_G_reduce[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx ].x += b_i[s]*RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].x;

                RK_G_reduce[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx ].y += b_i[s]*RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].y;

            }
        }
    }
    //evolucion
    }




}


__global__ void Dissipation(double *phi, double *psi, double *PI, double *A, double *B, double *alpha, double *Da, double *Db, 
    double * K, double *Kb, double *lambda, double *U, 
    cufftDoubleComplex *u, cufftDoubleComplex *u_p1,  cufftDoubleComplex *pi, cufftDoubleComplex *dr_u,
    cufftDoubleComplex *u_ghost, cufftDoubleComplex *u_p1_ghost,  cufftDoubleComplex *pi_ghost, cufftDoubleComplex *dr_u_ghost,
    double dt, int Nr,   int t){
int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx == 0){
        //phi[(t+1)Nr + idx] -= dt*Kreiss_Oliger(phi,idx,Nr,0.1);
        //PI[idx] -= dt*Kreiss_Oliger(PI,idx,Nr,0.1);
        //psi[idx] -= dt*Kreiss_Oliger(psi,idx,Nr,0.1);
            for(int r =0 ; r<Nr ; r++){
/*
                A[r] -= dt*Kreiss_Oliger(A,r,Nr,0.05);
                B[r] -= dt*Kreiss_Oliger(B,r,Nr,0.05);
                alpha[r] -= dt*Kreiss_Oliger(alpha,r,Nr,0.05);
                Da[r] -= dt*Kreiss_Oliger(Da,r,Nr,0.05);
                Db[r] += dt*Kreiss_Oliger(Db,r,Nr,0.05);
                Kb[r] += dt*Kreiss_Oliger(Kb,r,Nr,0.05);
                K[r] += dt*Kreiss_Oliger(K,r,Nr,0.05);
                U[r] -= dt*Kreiss_Oliger(U,r,Nr,0.05);*/  
            }
        //NOTA: kreiss oliger no funciona teniendo cada thread como un punto de la grilla.
        //lambda[idx] -= dt*Kreiss_Oliger(lambda,idx,Nr,0.1);
        //

        /*
        __syncthreads();
        for (int k =0; k<Nk ; k ++){
            for(int l=0 ; l<Nl ;l++){
                for(int r =0 ; r<Nr ; r++){
                  if(l==0){

                    u_p1[l*Nk*Nr + k*Nr + 0].x = (4.0*u_p1[l*Nk*Nr + k*Nr + 1].x -u_p1[l*Nk*Nr + k*Nr + 2].x)/3.0;
                    u_p1[l*Nk*Nr + k*Nr + 0].y = (4.0*u_p1[l*Nk*Nr + k*Nr + 1].y -u_p1[l*Nk*Nr + k*Nr + 2].y)/3.0;
                    pi[l*Nk*Nr + k*Nr + 0].x = (4.0*pi[l*Nk*Nr + k*Nr + 1].x -pi[l*Nk*Nr + k*Nr + 2].x)/3.0;
                    pi[l*Nk*Nr + k*Nr + 0].y = (4.0*pi[l*Nk*Nr + k*Nr + 1].y -pi[l*Nk*Nr + k*Nr + 2].y)/3.0;
                      dr_u[l*Nk*Nr + k*Nr + 0].x = 0.0;
                      dr_u[l*Nk*Nr + k*Nr + 0].y =0.0;
                    }
                    else{
                      u_p1[l*Nk*Nr + k*Nr + 0].x = 0.0;
                      u_p1[l*Nk*Nr + k*Nr + 0].y =0.0;
                      pi[l*Nk*Nr + k*Nr + 0].x = 0.0;
                      pi[l*Nk*Nr + k*Nr + 0].y =0.0;
                      dr_u[l*Nk*Nr + k*Nr + 0].x = 0.0;
                      dr_u[l*Nk*Nr + k*Nr + 0].y =0.0;
                    }*/

                  

/*
                    u_p1[l*Nk*Nr + k*Nr + r].x += dt*Kreiss_Oliger_Cx(u_p1,l*Nk*Nr + k*Nr + r,Nr,0.5).x; 
                    u_p1[l*Nk*Nr + k*Nr + r].y += dt*Kreiss_Oliger_Cx(u_p1,l*Nk*Nr + k*Nr + r,Nr,0.5).y; 
                    pi[l*Nk*Nr + k*Nr + r].x += dt*Kreiss_Oliger_Cx(pi,l*Nk*Nr + k*Nr + r,Nr,0.5).x; 
                    pi[l*Nk*Nr + k*Nr + r].y += dt*Kreiss_Oliger_Cx(pi,l*Nk*Nr + k*Nr + r,Nr,0.5).y;
                    dr_u[l*Nk*Nr + k*Nr + r].x += dt*Kreiss_Oliger_Cx(dr_u,l*Nk*Nr + k*Nr + r,Nr,0.5).x; 
                    dr_u[l*Nk*Nr + k*Nr + r].y += dt*Kreiss_Oliger_Cx(dr_u,l*Nk*Nr + k*Nr + r,Nr,0.5).y;
                   for(int g = 0 ; g<5 ; g++){


                      u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].x -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).x; 
                      u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].y -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).y; 
                      dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].x -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).x; 
                      dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].y -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).y; 
                      pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].x -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).x; 
                      pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r].y -= dt*Kreiss_Oliger_Cx(u_p1_ghost,g*Nk*Nl*Nr + l*Nk*Nr + k*Nr +r,Nr,0.1).y; 
                  }*//*
                }
            }
        }
      

      for( int g=0 ; g<5 ; g++){
        for (int k =0; k<Nk ; k ++){
          for(int l=0 ; l<Nl ;l++){
              for(int r =0 ; r<Nr ; r++){
                if(l==0){

                  u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = (4.0*u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 1].x -u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 2].x)/3.0;
                  u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y = (4.0*u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 1].y -u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 2].y)/3.0;

                  pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = (4.0*pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 1].x -pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 2].x)/3.0;
                  pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y = (4.0*pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 1].y -pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 2].y)/3.0;
                  dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = 0.0;
                  dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y =0.0;
                }
                else{

                  u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = 0.0;
                  u_p1_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y = 0.0;

                  pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = 0.0;
                  pi_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y = 0.0;
                  dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].x = 0.0;
                  dr_u_ghost[g*Nk*Nl*Nr + l*Nk*Nr + k*Nr + 0].y =0.0;
                }
              }
            }
          }
      }*/
    }
        if (idx==2){
            //A[idx-2]=B[idx-2];
            //Kb[idx-1]=K[idx-1]/3.0;
            //simetricas
            phi[(t+1)*Nr + 0]=(4.0*phi[(t+1)*Nr + 1]-phi[(t+1)*Nr + 2])/3.0;
            alpha[0]=(4.0*alpha[1]-alpha[2])/3.0;
            PI[0]=(4.0*PI[1]-PI[2])/3.0;
            A[0]=(4.0*A[1]-A[2])/3.0;
            B[0]=(4.0*B[1]-B[2])/3.0;
            K[0]=(4.0*K[1]-K[2])/3.0;
            Kb[0]=(4.0*Kb[1]-Kb[2])/3.0;

            //antisimetricas
            Db[0]=0.0;
            Da[0]=0.0;
            psi[0]=0.0;
            U[0]=0.0;
            lambda[0]=0.0;

    }
}
__global__ void Evolucion(double *phi, double *psi, double *PI, double *A, double *B, double *alpha, double *Da, double *Db, 
                          double * K, double *Kb, double *lambda, double *U, 
                          cufftDoubleComplex *u, cufftDoubleComplex *u_p1,  cufftDoubleComplex *pi, cufftDoubleComplex *dr_u,
                          cufftDoubleComplex *u_ghost, cufftDoubleComplex *u_p1_ghost,  cufftDoubleComplex *pi_ghost, cufftDoubleComplex *dr_u_ghost,
                          double *RK_C_reduce, cufftDoubleComplex *RK_Q_reduce, cufftDoubleComplex *RK_G_reduce,
                                int Nr,    double dt, int t){


    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx%Nr==3 && t%1==0 && idx/Nr/Nk ==   3 && idx/Nr%Nk==19){
      printf("Re(u) 0,17: %.15f\n",u[idx+1].x);

      printf("Re(u) %d,%d, %d ,%d: %.15f\n",idx%Nr, idx/Nr/Nk,t,idx, u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-2].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-3].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx+1].x);

      printf("Re(u) 0,17: %.15f\n",dr_u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-2].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-3].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx+1].x);

      printf("Re(pi) 0,17: %.15f\n",pi[ idx].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-1].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-2].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-3].x);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx+1].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-1].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-2].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-3].y);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-1].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-2].x);
      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx ].x);
      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx -1 ].x);
      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx -2 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -1 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -2 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -3 ].x);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx ].y);

      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -1 ].y);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -2 ].y);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -3 ].y);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx ].x);

      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -1 ].x);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -2 ].x);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -3 ].x);
    }


    if(idx < Nr){
        phi[(t+1)*Nr +idx] = phi[t*Nr +idx] + dt*RK_C_reduce[0*Nr + idx];
        PI[idx]        += dt*RK_C_reduce[2*Nr + idx];
        psi[idx]       += dt*RK_C_reduce[1*Nr + idx];
        A[idx]         += dt*RK_C_reduce[3*Nr + idx];
        B[idx]         += dt*RK_C_reduce[4*Nr + idx];
        alpha[idx]     += dt*RK_C_reduce[5*Nr + idx];
        Da[idx]        += dt*RK_C_reduce[6*Nr + idx];
        Db[idx]        += dt*RK_C_reduce[7*Nr + idx];
        K[idx]         += dt*RK_C_reduce[8*Nr + idx];
        Kb[idx]        += dt*RK_C_reduce[9*Nr + idx];
        lambda[idx]    += dt*RK_C_reduce[10*Nr + idx];
        U[idx]         += dt*RK_C_reduce[11*Nr + idx];


    }
    __syncthreads();

    //u_p1 es la evolucion de u, es necesario ya que luego tengo que tener la derivada temporal (presicion 1)
    if( idx < Nr*Nk*Nl ){
        u_p1[idx].x    = u[idx].x +  dt*RK_Q_reduce[0*Nl*Nk*Nr + idx].x;
        u_p1[idx].y    = u[idx].y +  dt*RK_Q_reduce[0*Nl*Nk*Nr + idx].y;

        dr_u[idx].x += dt*RK_Q_reduce[1*Nl*Nk*Nr + idx].x;
        dr_u[idx].y += dt*RK_Q_reduce[1*Nl*Nk*Nr + idx].y;

        pi[idx].x   += dt*RK_Q_reduce[2*Nl*Nk*Nr + idx].x;
        pi[idx].y   += dt*RK_Q_reduce[2*Nl*Nk*Nr + idx].y;




    }
    if( idx < Nr*Nk*Nl ){
        for(int g = 0 ; g<5 ; g++){
            u_p1_ghost[ g*Nk*Nl*Nr + idx].x    = u_ghost[ g*Nk*Nl*Nr + idx].x +  dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 0*Nl*Nk*Nr + idx ].x;
            u_p1_ghost[ g*Nk*Nl*Nr + idx].y    = u_ghost[ g*Nk*Nl*Nr + idx].y +  dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 0*Nl*Nk*Nr + idx ].y;
    
            dr_u_ghost[ g*Nk*Nl*Nr + idx].x += dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 1*Nl*Nk*Nr + idx ].x;
            dr_u_ghost[ g*Nk*Nl*Nr + idx].y += dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 1*Nl*Nk*Nr + idx ].y;
    
            pi_ghost[ g*Nk*Nl*Nr + idx].x   += dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 2*Nl*Nk*Nr + idx ].x;
            pi_ghost[ g*Nk*Nl*Nr + idx].y   += dt*RK_G_reduce[ g*3*Nl*Nk*Nr + 2*Nl*Nk*Nr + idx ].y;

        }
    }

    //codiciones de borde 

/*
    if(idx==2 && t%1==0){
      printf("Re(u) 0,17: %.15f\n",u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-2].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-2].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-1].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-2].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-1].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-2].x);

      printf("Re(duce) : %.15f\n",RK_G_reduce[ 0*Nl*Nk*Nr + idx ].x);
      printf("Re(duce) : %.15f\n",RK_G_reduce[ 0*Nl*Nk*Nr + idx -1 ].x);
      printf("Re(duce) : %.15f\n",RK_G_reduce[ 0*Nl*Nk*Nr + idx -2 ].x);
    }*/
    __syncthreads();
    if(idx%Nr==3 && t%1==0 && idx/Nr/Nk ==  3 && idx/Nr%Nk==19){
      printf("Re(u) 0,17: %.15f\n",u[idx+1].x);

      printf("Re(u) %d,%d, %d ,%d: %.15f\n",idx%Nr, idx/Nr/Nk,t,idx, u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-2].x);
      printf("Re(u) 0,17: %.15f\n",u[idx-3].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx+1].x);

      printf("Re(u) 0,17: %.15f\n",dr_u[ idx].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-1].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-2].x);
      printf("Re(u) 0,17: %.15f\n",dr_u[ idx-3].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx+1].x);

      printf("Re(pi) 0,17: %.15f\n",pi[ idx].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-1].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-2].x);
      printf("Re(pi) 0,17: %.15f\n",pi[ idx-3].x);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx+1].y);

      printf("Im(pi) 0,17: %.15f\n",pi[ idx].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-1].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-2].y);
      printf("Im(pi) 0,17: %.15f\n",pi[ idx-3].y);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-1].x);
      printf("Re(u_pi) 0,17: %.15f\n",u_p1[ idx-2].x);

      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx ].x);
      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx -1 ].x);
      printf("Re(duce) : %.15f\n",RK_Q_reduce[ 0*Nl*Nk*Nr + idx -2 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -1 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -2 ].x);
      printf("dr(duce) : %.15f\n",RK_Q_reduce[ 1*Nl*Nk*Nr + idx -3 ].x);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx ].y);

      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -1 ].y);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -2 ].y);
      printf("pi(duce) : %.15f\n",RK_Q_reduce[ 2*Nl*Nk*Nr + idx -3 ].y);

    }


}


// Fluctuaciones se divie en 2 funciones, fluctuacion esta dentro de un ciclo for de los l  por lo que dentro de fluctuacion() para un l voy a integrar todo los k
//una cantidad de Nl veces(ciclo ford), esto porque la integral tiene dentro una sumatoria de los nodos l, por lo que, son Nl integrales. 
__global__ void cambio_RK(  int Nr,   
                            double *RK_C, double *RK_C_temp, 
                            cufftDoubleComplex* RK_Q, cufftDoubleComplex* RK_Q_temp,  
                            cufftDoubleComplex* RK_G ,cufftDoubleComplex* RK_G_temp, int t, int s){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx<Nr){
        for( int i=0 ; i<12 ; i++){
            RK_C[ i*Nr + idx] =RK_C_temp[i*Nr + idx];

        }
    }

    if(idx<Nr*Nl*Nk){

        for( int i=0 ; i<3 ; i++){
                RK_Q[ i*Nr*Nk*Nl + idx].x =RK_Q_temp[i*Nl*Nk*Nr + idx].x;

                RK_Q[ i*Nr*Nk*Nl + idx].y =RK_Q_temp[i*Nl*Nk*Nr + idx].y;

        }

        for(int g=0 ; g<5 ; g++){
            for( int i=0 ; i<3 ; i++){
                    RK_G[ g*3*Nr*Nk*Nl + i*Nr*Nk*Nl + idx].x = RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].x;
        
                    RK_G[ g*3*Nr*Nk*Nl + i*Nr*Nk*Nl + idx].y = RK_G_temp[ g*3*Nl*Nk*Nr + i*Nl*Nk*Nr + idx].y;

            }
        }
    }

}



// Nota**: En este codigo pruebo si es lo mismo que hacer la derivada temporal para las fluctuaciones o solamente remplazarla con pi o dr_u... asi me ahorro crear a u_p1 
// y el problema de 1/alpha

//Nota **: Hay un ligero problema con la derivada radial de los nodos, se tiene la siguiente expresion : psi*r^l + phi r^(l-1)*l, al tener r = 0 y l = 0, se genera 
// una indeterminacion del tipo 0/0, se eligio que solamente se vuelve cero.

__global__ void fluctuation( cufftDoubleComplex *u, cufftDoubleComplex *u_p1, cufftDoubleComplex *dr_u, cufftDoubleComplex *pi, double *temp_array ,  double *temp_fluct, 
                              double dr, double dk, double dt, int Nr, int ghost, int l, int t ){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  int id;
  double radio;
  //temp_fluct (5xNr)
  double cte= 1.0/(4.0*consta_pi);
  double u_snake_x;
  double u_snake_y;

  double d_u_temp_x;
  double d_u_temp_y;
  double d_u_radial_x;
  double d_u_radial_y;

  //radio
  if (idx < Nr){
    if(idx==0){
      radio=0.0;
    }
    else{
      radio=idx*dr;
    }

    //array que guarda la integral de todos los k  en un r de un solo l
    temp_fluct[ 0*Nr + idx] = 0.0;
    temp_fluct[ 1*Nr + idx] = 0.0;
    temp_fluct[ 2*Nr + idx] = 0.0;
    temp_fluct[ 3*Nr + idx] = 0.0;
    temp_fluct[ 4*Nr + idx] = 0.0;

    for (int k=0 ; k < Nk ; k++){
    // indice
    id = ghost*Nk*Nl*Nr + l*Nk*Nr + k*Nr +  idx; 

    //derivada temporal
    d_u_temp_x = pi[ id ].x * pow(radio , l);
    d_u_temp_y = pi[ id ].y * pow(radio , l);

    //derivada radial compleja

    if(l==0){
        d_u_radial_x = (dr_u[id ].x);
        d_u_radial_y = (dr_u[id ].y); 
    }
    else if(l==1){
        d_u_radial_x =  (dr_u[id ].x* pow(radio , l) + u[id ].x);
        d_u_radial_y =  (dr_u[id ].y* pow(radio , l) + u[id ].y);
    }
    else{
        d_u_radial_x = (dr_u[id ].x* pow(radio , l) + u[id ].x* pow(radio , l-1) *l);
        d_u_radial_y = (dr_u[id ].y* pow(radio , l) + u[id ].y* pow(radio , l-1) *l);
    }


    if(idx==0){
        u_snake_x = u[ id +1].x * pow(radio +dr, l) ;
        u_snake_y = u[ id +1].y * pow(radio +dr, l) ;
    }
    else{
        u_snake_x = u[ id ].x * pow(radio , l) ;
        u_snake_y = u[ id ].y * pow(radio , l) ;

    }


    temp_fluct[ 0*Nr + idx] += ( pow(d_u_temp_x,2) + pow(d_u_temp_y,2));
    temp_fluct[ 1*Nr + idx] += ( pow(d_u_radial_x,2) + pow(d_u_radial_y,2));
    temp_fluct[ 2*Nr + idx] += d_u_temp_x*d_u_radial_x + d_u_temp_y*d_u_radial_y;
    temp_fluct[ 3*Nr + idx] += ( pow(u_snake_x,2) + pow(u_snake_y,2) );
    temp_fluct[ 4*Nr + idx] += ( pow(u_snake_x,2) + pow(u_snake_y,2) );

  }
    //fluctuacion tiempos
    temp_fluct[ 0*Nr + idx] = temp_fluct[ 0*Nr + idx] * (2*l + 1) * cte * dk;
    //fluctuacion radial
    temp_fluct[ 1*Nr + idx] = temp_fluct[ 1*Nr + idx] * (2*l + 1) * cte * dk;
    //fluctuacion t r
    temp_fluct[ 2*Nr + idx] = temp_fluct[ 2*Nr + idx] * (2*l + 1) * cte* dk;
    //fluctuacion angulares
    temp_fluct[ 3*Nr + idx] = temp_fluct[ 3*Nr + idx] * 0.5 * l*(l+1)*(2*l + 1) * cte* dk;
    //fluctuacion phi
    temp_fluct[ 4*Nr + idx] = temp_fluct[ 4*Nr + idx] * (2*l + 1) * cte* dk;





    //guardar las fluctuacions en un array Nl*Nr*5, (5 fluctuaciones )
    temp_array[ 0*Nl*Nr + l*Nr + idx ]=temp_fluct[ 0*Nr + idx];
    temp_array[ 1*Nl*Nr + l*Nr + idx ]=temp_fluct[ 1*Nr + idx];
    temp_array[ 2*Nl*Nr + l*Nr + idx ]=temp_fluct[ 2*Nr + idx];
    temp_array[ 3*Nl*Nr + l*Nr + idx ]=temp_fluct[ 3*Nr + idx];
    temp_array[ 4*Nl*Nr + l*Nr + idx ]=temp_fluct[ 4*Nr + idx];
  }
}

//segunda parte, sumar las  Nl  fluctuaciones de cada r
__global__  void suma_fluct(double *temp_array ,  int Nr){
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

//Actualizar el tensro Stress_Energy
__global__ void stress_energy(double *SA, double *SB, double *rho, double *ja, double *phi, double *psi, double *PI, double *A, double *B, double *alpha, double *temp_array , double dr, double dt, int Nr,   int t){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < Nr){


    //calcular las distintas derivadas del campos clasico ( o phi )




    double  radio;
    if(idx==0){
      radio=dr;
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

    //valor de espectacion de las derivadas
    //PELIGRO cuando alpha = 0
    espectation_dt_phi = PI[idx]*PI[idx] +  temp_array[ idx ];

    espectation_dr_phi = psi[idx]*psi[idx] + temp_array[ 1*Nl*Nr + idx ];


    espectation_dr_dt_phi = PI[idx]*psi[idx] +  temp_array[ 2*Nl*Nr + idx ];

    espectation_dtheta_phi = temp_array[ 3*Nl*Nr + idx ];
    // espectation_dphi_phi = temp_array[ 4*Nl*Nr + idx ];


    double xPI2x , xpsi2x, xpipsix;
    xPI2x =  espectation_dt_phi;
    xpsi2x = espectation_dr_phi;
    xpipsix =  espectation_dr_dt_phi;
    //Stress-Energy

    //masa (para el campo phi mu = 0.0 y para los fantasmas 1 o sqrt(3)), mu = m*c/h, masa, velocidad de la luz y constante de plank reducida


    //actualizacion

     rho[idx] =  1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) + 1.0/(B[idx]*radio*radio)*espectation_dtheta_phi ;

    

    ja[idx] = - xpipsix/(sqrt(A[idx])*B[idx]);

    SA[idx] = 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) - 1.0/(B[idx]*radio*radio)*espectation_dtheta_phi ;

    SB[idx] = 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) - xpsi2x) ;
    
    if(idx==0 &&  t<50){
      printf("Stress_Energy en t= %d | idx : %d:\n",t,idx); 

        printf(" <x|PI|x>^2 = %1.5f\n",xPI2x);
        printf(" <x|psi|x>^2 = %1.5f\n",xpsi2x);
        printf(" <x|PI psi|x>^2 = %1.5f\n",temp_array[ 2*Nl*Nr + idx ]);
        printf(" <x|theta|x>^2 = %1.5f\n",temp_array[ 3*Nl*Nr + idx ]/(B[idx]*radio*radio));
        printf(" <x|phi|x>^2 = %1.5f\n",temp_array[ 4*Nl*Nr + idx ]);
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

//Stress_Energy Ghosts...

__global__ void stress_energy_ghost(double *SA, double *SB, double *rho, double *ja,double *A, double *B, double *alpha, double *temp_array , double dr, double dt, int Nr,   int t, double mu, int g){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double dt_phi, dr_phi;
    if(idx < Nr){
  
      double  radio;
      if(idx==0){
        radio=dr;
      }
      else{
        radio=idx*dr;
      }
  
      //variables
  

  
      //valor de espectacion de las derivadas
      //espectation_dt_phi = temp_array[ idx ];
      //espectation_dr_phi =  temp_array[ 1*Nl*Nr + idx ];
      //espectation_dr_dt_phi =  temp_array[ 2*Nl*Nr + idx ];
      //espectation_dtheta_phi = temp_array[ 3*Nl*Nr + idx ];
      //espectation_phi_phi = temp_array[ 4*Nl*Nr + idx ];
  
  
      double xPI2x , xpsi2x, xpipsix;
      xPI2x =  temp_array[ idx ];
      xpsi2x =  temp_array[ 1*Nl*Nr + idx ];
      xpipsix =   temp_array[ 2*Nl*Nr + idx ];
      //Stress-Energy
  
      //masa (para el campo phi mu = 0.0 y para los fantasmas 1 o sqrt(3)), mu = m*c/h, masa, velocidad de la luz y constante de plank reducida
  
      //actualizacion
      if (g == 0 || g == 2 || g == 4){

        rho[idx] -=  1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) + 1.0/(B[idx]*radio*radio)*temp_array[ 3*Nl*Nr + idx ] - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
        
        ja[idx] -= - xpipsix/(sqrt(A[idx])*B[idx]);
    
        SA[idx] -= 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) - 1.0/(B[idx]*radio*radio)*temp_array[ 3*Nl*Nr + idx ] - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
    
        SB[idx] -= 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) - xpsi2x)  - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
      }
      else{

        rho[idx] +=  1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) + 1.0/(B[idx]*radio*radio)*temp_array[ 3*Nl*Nr + idx ] - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
        
        ja[idx] += - xpipsix/(sqrt(A[idx])*B[idx]);
    
        SA[idx] += 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) + xpsi2x) - 1.0/(B[idx]*radio*radio)*temp_array[ 3*Nl*Nr + idx ] - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
    
        SB[idx] += 1.0/(2.0*A[idx]) * (xPI2x/(B[idx]*B[idx]) - xpsi2x)  - 0.5*mu*mu*temp_array[ 4*Nl*Nr + idx ];
      }


      if(idx== 0){
      printf("Stress_Energy en t= %d:\n",t); 
          printf(" <x|PI|x>^2 = %1.5f\n",xPI2x);
          printf(" temp = %1.5f\n",temp_array[idx]);

          printf(" <x|psi|x>^2 = %1.5f\n",xpsi2x);
          printf(" <x|PI psi|x>^2 = %1.5f\n",temp_array[ 2*Nl*Nr + idx ]);
          printf(" <x|theta|x>^2 = %1.5f\n",temp_array[ 3*Nl*Nr + idx ]/(B[idx]*radio*radio));
          printf(" <x|phi|x>^2 = %1.5f\n",temp_array[ 4*Nl*Nr + idx ]);

          printf("rho : %.15f\n",rho[idx]);
          printf("ja : %.15f\n",ja[idx]);
          printf("SA : %.15f\n",SA[idx]);
          printf("SB : %.15f\n",SB[idx]);
      
      
        }
    }

  }

__global__ void cost_cosm_value(double *rho,double *cosmological_constant){
  //cosmological_constant[0] = 1.0/pow(2.0*consta_pi,2)*log(pow(3.0,9)/pow(2.0,16))/8.0;
  cosmological_constant[0] = -rho[0];
    printf("Constante cosmologica : %.15f \n", cosmological_constant[0]);
  
}

__global__ void Tensor_tt(double * T_tt,double * T_rr,double * T_tr,double * T_00, double *rho,double *SA,double *ja,double *SB, int t, int Nr){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx<Nr){
    T_tt[ t*Nr + idx] = rho[idx];
    T_rr[ t*Nr + idx] = SA[idx];
    T_tr[ t*Nr + idx] = ja[idx];
    T_00[ t*Nr + idx] = SB[idx];

}
  

  
}

//remplazo el array de u por los valores de u_p1 ( su paso de tiempo)
__global__ void cambio_u(cufftDoubleComplex *u, cufftDoubleComplex *u_p1, int Nr,   int g){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  if(idx < Nr*Nk*Nl){
    u[ g*Nr*Nk*Nl + idx].x = u_p1[  g*Nr*Nk*Nl + idx].x;
    u[ g*Nr*Nk*Nl + idx].y = u_p1[ g*Nr*Nk*Nl + idx].y;

  }  
  __syncthreads();

}


//condiciones iniciales



//k_wavenumber/sqrt(PI*omega_phi)*pow(k_wavenumber,l_value)/gsl_sf_doublefact(2*l_value+1)


//C.I
void u_initial(cufftDoubleComplex *u,int k, int l, int field, int Nr,double dr, double dk , double mass){
    int id_u= field + l*Nk*Nr + k*Nr ;
    double omega = sqrt( dk*(k+1)*dk*(k+1) + mass*mass ) ;


    for (int r=0 ; r<Nr ; r++){

        //Hay un problema en r=0, debido a que u = 0/0, por lo que indetermina, por lo que, se eligio escoger una aproximacion asintotica.

        //Aproximacion a kr << 1, considerando dr=0.025 y dk=pi/15 ~ 0.2, entonces para que recien k*r tenga un valor igual a 1 
        // considerando dk*dr*i (k=1), entonces i debria ser 200 aproximadamente, ahora considerando k=Nk (por ahora Nk =20),
        // dk*21 ~ 4.2, por lo tanto para que dr*i baje este valor y usar una condicion asintotica  podemos considerar  un margen
        // de que kr<0.1 para que se cumpla, entonces dr*i < 0.1/4.2 ~ 0.025, entonces para que sea aceptable i debe ser 1
        // podemos elegir ahora nuestro rango de r para que cumpla la condicion asintotica, la cual se eligira  entre [0,10].

        // Nota: la aproximacion dependera de k, para frecuencias altas la condicion inicial en r pequeños no sera del todo precisa
        // lo cual dejará abierta la posibilidad de modificar esta seccion para que dependa de un rango apropiado de k tambien.

        if (r == Origin ){
          if(2*l+1 < GSL_SF_DOUBLEFACT_NMAX){
        //if(l==0){
            u[id_u + r].x = (dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1) ;
            u[id_u + r].y = 0.0;
          }
          else{
            u[id_u + r].x = 0.0;
            u[id_u + r].y = 0.0;
          }
            //u[id_u + r].x=0.0;


        }
        else{
            //u[id_u + r].x=0.0;

            u[id_u + r].x = (dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l);
            u[id_u + r].y=0.0;

        }
    }
}

//C.I
void pi_initial(cufftDoubleComplex* pi,int k, int l,int field, int Nr,double dr, double dk, double mass){
    int id_u = field + l*Nk*Nr + k*Nr ;
    double omega = sqrt( dk*(k+1)*dk*(k+1) + mass*mass ) ;

    for (int r=0 ; r<Nr ; r++){
        if (r == Origin){
            if(2*l+1 < GSL_SF_DOUBLEFACT_NMAX){
            //if(l==0){
                pi[id_u + r].x = 0.0 ;
                pi[id_u + r].y = -omega*(dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1);
              }
              else{
                pi[id_u + r].x = 0.0;
                pi[id_u + r].y = 0.0;
              }

        }
        else{
            pi[id_u + r].x= 0.0;
            //pi[id_u + r].y= 0.0;

            pi[id_u + r].y = -omega*(dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l);

        }
    }
}
//C.I
void dr_u_initial(cufftDoubleComplex* dr_u,int k, int l,int field, int Nr,double dr, double dk, double mass){
    int id_u = field + l*Nk*Nr + k*Nr ;

    double omega = sqrt( dk*(k+1)*dk*(k+1) + mass*mass ) ;

    double dr_j;

    //La derivada de los esfericos de bessel es k j_l(kr)' =k( (l/(kr)) * j_l(kr) - j_{l+1}(kr))

    for (int r=0 ; r<Nr ; r++){
        if (r == Origin){
            if(2*l+1 < GSL_SF_DOUBLEFACT_NMAX){

            //dr_u[id_u + r].x=0.0;

              dr_u[id_u + r].x = (pow(dk*(k+1),l+3)/sqrt(consta_pi * omega)) * (-1)*(dr*r) /gsl_sf_doublefact(2*l+3);
              dr_u[id_u + r].y = 0.0;
            }
            else{
              dr_u[id_u + r].x = 0.0;
              dr_u[id_u + r].y = 0.0;
            }

        }
        else {
            //dr_u[id_u + r].x=0.0;
            dr_u[id_u + r].x = ( pow(dk*(k+1),2) /sqrt(consta_pi * omega)) * (-1) * gsl_sf_bessel_jl( l +1, dk*(k+1)*dr*(r)  )/ (pow(dr*r,l));
            dr_u[id_u + r].y = 0.0;

        }

    }
}
//C.I
//iniciar phi
void inicial_phi(double *phi, double dr,int Nr){
  double a=0.0;
  double std=1.5;
  for (int i=0;i<Nr;i++){
    //phi[i]=a;
    phi[i]=a*(i*dr-20.0)/std*exp(-( (i*dr-20.0) /std)*((i*dr-20.0) /std));

    //phi[i]=a*exp(-( (i*dr-20.0) /std)*((i*dr-20.0) /std));
  }
}
//C.I

//iniciar A, dfr es la misma forma de phi, se ocupa para encontrar el punto medio de la derivada de phi cuando ocupo el RK4
double dfr(double r,double a,double std){
  //return 0.0;
  //return a*exp(-( (r-0.0) /std)*((r-0.0) /std));
  return a*(r-20.0)/std*exp(-( (r-20.0) /std)*((r-20.0) /std));
 }
 //C.I


//rellenar un array
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
  int Nr=200;
  int Nt=2000;
  // Defino los array del host
  double *A,*B,*alpha,*phi,*psi,*PI,*lambda,*K,*Kb,*U, *Da, *Db, *temp_phi, *temp_Kb,*temp_alpha;
  double *rho,*ja,*SA,*SB;

  // Defino los array del cuda

  double *cuda_A,*cuda_B,*cuda_alpha,*cuda_phi,*cuda_psi,*cuda_PI,*cuda_lambda,*cuda_K,*cuda_Kb,*cuda_U, *cuda_Da, *cuda_Db, *cuda_temp_phi, *cuda_temp_Kb,*cuda_temp_alpha;
  double *cuda_rho,*cuda_ja,*cuda_SA,*cuda_SB, *cosmological_constant;

  double *T_tt, *cuda_T_tt;
  double *T_rr, *cuda_T_rr;
  double *T_tr, *cuda_T_tr;
  double *T_00, *cuda_T_00;

  //double *K_s,*b_i, *a_ij, *c_i;

  //deltas
  double dr=0.025;
  //double dt=5.0/10000;
  double dt=dr/4.0; //el bueno
  //double dt=4.0/20000;
  printf("dr=%f , dt=%f",dr,dt);
// mallocs
phi=(double *)malloc(Nr*Nt*sizeof(double));
//Tensor stress-energia host
T_tt=(double *)malloc(Nr*Nt*sizeof(double));
T_rr=(double *)malloc(Nr*Nt*sizeof(double));
T_tr=(double *)malloc(Nr*Nt*sizeof(double));
T_00=(double *)malloc(Nr*Nt*sizeof(double));

temp_Kb=(double *)malloc(Nt*Nr*sizeof(double));
temp_phi=(double *)malloc(Nr*sizeof(double));
temp_alpha=(double *)malloc(Nt*Nr*sizeof(double));

  PI=(double *)malloc(Nr*sizeof(double));
  psi=(double *)malloc(Nr*sizeof(double));

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
  rho=(double *)malloc((Nr+1)*sizeof(double));
  SA=(double *)malloc(Nr*sizeof(double));
  SB=(double *)malloc(Nr*sizeof(double));









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

//Ghost Fields

cufftDoubleComplex *u_ghost;
cufftDoubleComplex *u_ghost_p1;

cufftDoubleComplex *pi_ghost;
cufftDoubleComplex *dr_u_ghost;

cufftDoubleComplex *cuda_u_ghost;
cufftDoubleComplex *cuda_u_ghost_p1;

cufftDoubleComplex *cuda_pi_ghost;
cufftDoubleComplex *cuda_dr_u_ghost;

//Nodos Quantum

  u_nodos=(cufftDoubleComplex*)malloc(Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  u_nodos_p1=(cufftDoubleComplex*)malloc(Nk*Nl*Nr*sizeof(cufftDoubleComplex));

  pi=(cufftDoubleComplex*)malloc(       Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  dr_u=(cufftDoubleComplex*)malloc(     Nk*Nl*Nr*sizeof(cufftDoubleComplex));

// cuda mallocs
    time1 = (double) clock();            
    time1 = time1 / CLOCKS_PER_SEC; 
    int idk;
    double dk = consta_pi/15.0;

    //Inicializar las variables de los nodos
    #pragma omp parallel for 
    for (idk=0;idk<Nk;idk++){
      for (int idl=0;idl<Nl;idl++){
          u_initial(u_nodos,    idk,idl, 0, Nr,dr, dk, 0.0);
          pi_initial(pi,        idk,idl, 0, Nr,dr, dk, 0.0);
          dr_u_initial(dr_u,    idk,idl, 0, Nr,dr, dk, 0.0);
      }
    }

//Nodos

  u_ghost=(cufftDoubleComplex*)malloc(       5*Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  u_ghost_p1=(cufftDoubleComplex*)malloc(     5*Nk*Nl*Nr*sizeof(cufftDoubleComplex));

  pi_ghost=(cufftDoubleComplex*)malloc(       5*Nk*Nl*Nr*sizeof(cufftDoubleComplex));
  dr_u_ghost=(cufftDoubleComplex*)malloc(     5*Nk*Nl*Nr*sizeof(cufftDoubleComplex));

// cuda Ghost
    time1 = (double) clock();            
    time1 = time1 / CLOCKS_PER_SEC; 
    double mass;
    //Inicializar las variables de los nodos
    for(int g=0 ; g<5 ;g++){
      #pragma omp parallel for 
      for (idk=0;idk<Nk;idk++){
        for (int idl=0;idl<Nl;idl++){
            if (g==2 || g==0){
                mass = 1.0;
            }
            else if (g==4){
                mass = 1.0*sqrt(4.0);
            }
            else{
                mass = 1.0*sqrt(3.0);
            }
            u_initial(u_ghost,        idk,idl, g*Nk*Nl*Nr,  Nr,dr, dk, mass);
            pi_initial(pi_ghost,      idk,idl, g*Nk*Nl*Nr,  Nr,dr, dk, mass);
            dr_u_initial(dr_u_ghost,  idk,idl, g*Nk*Nl*Nr,  Nr,dr, dk, mass);
        }
      }
    }

    timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
    printf("The elapsed time is %lf seconds, opm\n", timedif);
    //printf("Redr_u_%d,%d = %lf \n",19,17,dr_u[ 718000].x);

 


/*
for (int idk=0;idk<Nk;idk++){
  for (int idl=0;idl<Nl;idl++){
    printf("Reu_%d,%d = %lf \n",idk,idl,u_nodos[ idl*Nk*Nr + idk*Nr + 0 ].x);
    printf("Repi_%d,%d = %lf \n",idk,idl,pi[ idl*Nk*Nr + idk*Nr + 0 ].x);
    printf("Redr_u_%d,%d = %lf \n",idk,idl,dr_u[ idl*Nk*Nr + idk*Nr + 0 ].x);
    printf("Imu_%d,%d = %lf \n",idk,idl,u_nodos[ idl*Nk*Nr + idk*Nr + 0 ].y);
    printf("Impi_%d,%d = %lf \n",idk,idl,pi[ idl*Nk*Nr + idk*Nr + 0 ].y);
    printf("Imdr_u_%d,%d = %lf \n",idk,idl,dr_u[ idl*Nk*Nr + idk*Nr + 0 ].y);
  }
}*/


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
//CUDA...


cudaMalloc((void **)&cuda_phi, Nt*Nr*sizeof(double));

//Tensor stress-energia cuda
cudaMalloc((void **)&cuda_T_tt, Nt*Nr*sizeof(double));
cudaMalloc((void **)&cuda_T_rr, Nt*Nr*sizeof(double));
cudaMalloc((void **)&cuda_T_tr, Nt*Nr*sizeof(double));
cudaMalloc((void **)&cuda_T_00, Nt*Nr*sizeof(double));

cudaMalloc((void **)&cuda_psi, Nr*sizeof(double));
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

cudaMalloc((void **)&cuda_u_ghost,    5*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&cuda_u_ghost_p1, 5*Nr*Nk*Nl*sizeof(cufftDoubleComplex));

cudaMalloc((void **)&cuda_dr_u_ghost, 5*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&cuda_pi_ghost,   5*Nr*Nk*Nl*sizeof(cufftDoubleComplex));




cudaMalloc((void **)&cuda_rho, Nr*sizeof(double));
cudaMalloc((void **)&cuda_SA, Nr*sizeof(double));
cudaMalloc((void **)&cuda_SB, Nr*sizeof(double));
cudaMalloc((void **)&cuda_ja, Nr*sizeof(double));
cudaMalloc((void **)&cosmological_constant, 1*sizeof(double));


//cuda memcpy

//CLASSIC
/*
cudaMemcpy(cuda_phi, phi, Nr*Nt*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_psi, psi, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_PI, PI, Nr*sizeof(double), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_A, A, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_B, B, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_alpha, alpha, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_Da, Da, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_Db, Db, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_K, K, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_Kb, Kb, Nr*sizeof(double), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_lambda, lambda, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_U, U, Nr*sizeof(double), cudaMemcpyHostToDevice);*/

//QUAMTUN

cudaMemcpy(cuda_u_nodos,  u_nodos,          Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_u_nodos_p1,  u_nodos_p1,    Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_dr_u,     dr_u,             Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_pi,       pi,               Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

//GHOST

cudaMemcpy(cuda_u_ghost,  u_ghost,            5*Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_u_ghost_p1,  u_ghost_p1,      5*Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

cudaMemcpy(cuda_dr_u_ghost,     dr_u_ghost,         5*Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_pi_ghost,       pi_ghost,           5*Nr*Nk*Nl*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);


//SEMICLASSIC

cudaMemcpy(cuda_SA, SA, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_SB, SB, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_ja, ja, Nr*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(cuda_rho, rho, (Nr+1)*sizeof(double), cudaMemcpyHostToDevice);



//RK
double *RK_C , *RK_C_temp, *RK_C_reduce;
cufftDoubleComplex *RK_Q, *RK_Q_temp, *RK_Q_reduce;
cufftDoubleComplex *RK_G, *RK_G_temp, *RK_G_reduce;
// ****Classic ****
cudaMalloc((void **)&RK_C, consta_var * Nr*sizeof(double ));
cudaMalloc((void **)&RK_C_temp, consta_var * Nr*sizeof(double ));
cudaMalloc((void **)&RK_C_reduce, consta_var * Nr*sizeof(double ));

// ****Quantum ****
cudaMalloc((void **)&RK_Q, 3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&RK_Q_temp, 3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&RK_Q_reduce, 3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));

// ****Ghosts ****
cudaMalloc((void **)&RK_G, 5*3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&RK_G_temp, 5*3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));
cudaMalloc((void **)&RK_G_reduce, 5*3*Nr*Nk*Nl*sizeof(cufftDoubleComplex));


double *temp_array, *temp_fluct;
cudaMalloc((void **)&temp_array, 5 * Nl*Nr*sizeof(double));
cudaMalloc((void **)&temp_fluct, 5 * Nr*sizeof(double));

//Evolution...
time1 = (double) clock();            /* get initial time */
time1 = time1 / CLOCKS_PER_SEC; 



    int thread=64;
    dim3 bloque(thread);
    dim3 grid((int)ceil((float)(Nr*Nk*Nl)/thread));
    dim3 grid_radial((int)ceil((float)(Nr)/thread));


    ////Simulacion////
printf("thread = %d , block = %d , dk = %lf ", thread , (int)ceil((float)(Nr*Nk*Nl)/thread) , dk);


/////Initial Condition/////
metric_initial<<<grid_radial,bloque>>>(cuda_phi , cuda_psi , cuda_PI, cuda_A, cuda_B, cuda_alpha, cuda_Da, cuda_Db, cuda_K, cuda_Kb, cuda_lambda,
  cuda_U,cosmological_constant,Nr,dr);
  
cudaDeviceSynchronize();
for(int g = 0 ; g<5 ;g++){

/////Quamtum field////
  if(g==0){
    for (int l = 0 ; l < Nl; l++ ){
      fluctuation<<< grid_radial  , bloque>>>( cuda_u_nodos, cuda_u_nodos_p1, cuda_dr_u, cuda_pi, temp_array, temp_fluct, dr, dk,dt, Nr, 0 ,l,0);
    cudaDeviceSynchronize();
    }  
      suma_fluct<<< grid_radial , bloque>>>(temp_array,Nr);
    cudaDeviceSynchronize();
      stress_energy<<< grid_radial , bloque>>>(cuda_SA, cuda_SB, cuda_rho, cuda_ja, cuda_phi, cuda_psi,cuda_PI, cuda_A, cuda_B, cuda_alpha, temp_array,dr,dt,Nr,0);
    cudaDeviceSynchronize();
  }

/////Ghosts fields////
  for (int l = 0 ; l < Nl; l++ ){
    fluctuation<<< grid_radial , bloque >>>(cuda_u_ghost, cuda_u_ghost_p1, cuda_dr_u_ghost, cuda_pi_ghost, temp_array, temp_fluct, dr, dk, dt, Nr, g , l, 0);
  }
  cudaDeviceSynchronize();
    suma_fluct<<< grid_radial , bloque>>>(temp_array,Nr);
  cudaDeviceSynchronize();
    stress_energy_ghost<<< grid_radial , bloque>>>(cuda_SA, cuda_SB, cuda_rho, cuda_ja, cuda_A, cuda_B, cuda_alpha, temp_array,dr,dt,Nr,0, 0.0, g  );
  cudaDeviceSynchronize();
}

  cost_cosm_value<<<1,1>>>(cuda_rho,cosmological_constant);
cudaDeviceSynchronize();



  A_RK<<<1,1>>>( cuda_A, cosmological_constant, Nr,dr);

cudaDeviceSynchronize();

  initial_lambda <<<grid_radial,bloque>>> (cuda_lambda, cuda_A, cuda_B, dr, Nr);

cudaDeviceSynchronize();

  initial_U <<<grid_radial,bloque>>> (cuda_U, cuda_lambda, cuda_A, dr, Nr);

cudaDeviceSynchronize();

for (int t = 0 ; t < 10   ; t++ ){


/////Quamtum field////
    for (int l = 0 ; l < Nl; l++ ){

      fluctuation<<< grid_radial  , bloque>>>( cuda_u_nodos, cuda_u_nodos_p1, cuda_dr_u, cuda_pi, temp_array, temp_fluct, dr, dk,dt, Nr,  0 ,l,t);
      cudaDeviceSynchronize();

    }

    cudaDeviceSynchronize();
    suma_fluct<<< grid_radial , bloque>>>(temp_array,Nr);
    cudaDeviceSynchronize();
    stress_energy<<< grid_radial , bloque>>>(cuda_SA, cuda_SB, cuda_rho, cuda_ja, cuda_phi, cuda_psi,cuda_PI, cuda_A, cuda_B, cuda_alpha, temp_array,dr,dt,Nr,t);
    cudaDeviceSynchronize();

/////Ghosts fields////

    for (int g = 0 ; g<5 ; g++){
        for (int l = 0 ; l < Nl; l++ ){
            fluctuation<<< grid_radial , bloque >>>(cuda_u_ghost, cuda_u_ghost_p1, cuda_dr_u_ghost, cuda_pi_ghost, temp_array, temp_fluct, dr, dk, dt, Nr, g , l, t);
        }
        if (g==2 || g==0){
            mass = 0.0;
        }
        else if (g==4){
            mass = 0.0*sqrt(4.0);
        }
        else{
            mass = 0.0*sqrt(3.0); 
        }
        cudaDeviceSynchronize();
        suma_fluct<<< grid_radial , bloque>>>(temp_array,Nr);
        cudaDeviceSynchronize();
        stress_energy_ghost<<< grid_radial , bloque>>>(cuda_SA, cuda_SB, cuda_rho, cuda_ja, cuda_A, cuda_B, cuda_alpha, temp_array,dr,dt,Nr,t, mass, g  );
        cudaDeviceSynchronize();

    }


    ///Evolucion///
    for (int s=0; s<4; s++){

      Runge_kutta_classic<<< grid_radial , bloque >>>(cuda_phi , cuda_psi , cuda_PI, cuda_A, cuda_B, cuda_alpha, cuda_Da, cuda_Db, cuda_K, cuda_Kb, cuda_lambda,
                                                      cuda_U, cuda_SA, cuda_SB, cuda_ja, cuda_rho,cosmological_constant, Nr, dr, dk, dt, t, RK_C, RK_C_temp, RK_C_reduce, 1,s);
      cudaDeviceSynchronize();

      Runge_kutta_nodos<<< grid , bloque >>>( cuda_A, cuda_B, cuda_alpha, cuda_lambda,
                                              cuda_u_nodos, cuda_u_nodos_p1, cuda_pi, cuda_dr_u, 
                                              cuda_u_ghost,  cuda_u_ghost_p1, cuda_pi_ghost, cuda_dr_u_ghost,
                                              Nr, dr, dt, t, RK_C, 
                                              RK_Q, RK_Q_temp, RK_Q_reduce, RK_G, RK_G_temp, RK_G_reduce, 1,s);
      cudaDeviceSynchronize();

      cambio_RK <<< grid , bloque >>>(Nr, RK_C, RK_C_temp, RK_Q, RK_Q_temp, RK_G, RK_G_temp, t, s);

      if(s==3){
          Evolucion<<< grid , bloque >>>( cuda_phi , cuda_psi , cuda_PI, cuda_A, cuda_B, cuda_alpha, cuda_Da, cuda_Db, cuda_K, cuda_Kb, cuda_lambda,
                                          cuda_U,  cuda_u_nodos, cuda_u_nodos_p1, cuda_pi, cuda_dr_u, 
                                          cuda_u_ghost,  cuda_u_ghost_p1, cuda_pi_ghost, cuda_dr_u_ghost,
                                          RK_C_reduce, RK_Q_reduce, RK_G_reduce, Nr,  dt, t);
      }
      cudaDeviceSynchronize();

    }
    cudaDeviceSynchronize();
    Dissipation<<<1,4>>>(   cuda_phi , cuda_psi , cuda_PI, cuda_A, cuda_B, cuda_alpha, cuda_Da, cuda_Db, cuda_K, cuda_Kb, cuda_lambda,
                        cuda_U, cuda_u_nodos, cuda_u_nodos_p1, cuda_pi, cuda_dr_u, 
                        cuda_u_ghost,  cuda_u_ghost_p1, cuda_pi_ghost, cuda_dr_u_ghost,dt,Nr,t);

    cudaDeviceSynchronize();



    Tensor_tt<<< grid_radial,bloque >>>(cuda_T_tt,cuda_T_rr,cuda_T_tr,cuda_T_00, cuda_rho,cuda_SA,cuda_ja,cuda_SB,  t,  Nr);
    cudaDeviceSynchronize();

  }
timedif = ( ((double) clock()) / CLOCKS_PER_SEC) - time1;
printf("The elapsed time is %lf seconds, GPU\n", timedif);


//cudamemcpy devuelta.

cudaMemcpy(rho, cuda_rho, Nr*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(SA, cuda_SA, Nr*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(ja, cuda_ja, Nr*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(SB, cuda_SB, Nr*sizeof(double), cudaMemcpyDeviceToHost);

/*
  for(int r=0; r<Nr; r++){
    printf("phi_%d : %.15f\n",r,phi[5*Nr + r]);
  }
*/

//error cuda
cudaError_t err = cudaGetLastError();
printf("Error: %s\n",cudaGetErrorString(err));


//guardar salida Tensor energia momentum
guardar_salida_rho(rho,Nr,1);
guardar_salida_SA(SA,Nr,1);
guardar_salida_ja(ja,Nr,1);
guardar_salida_SB(SB,Nr,1);

size_t free_memory, total_memory;
cudaMemGetInfo( &free_memory, &total_memory );

//Memoria
printf("Memoria libre: %zu bytes\n", free_memory);
printf("Memoria total: %zu bytes\n", total_memory);

//  MB para mayor claridad
printf("Memoria libre: %.2f MB\n", free_memory / (1024.0 * 1024.0));
printf("Memoria total: %.2f MB\n", total_memory / (1024.0 * 1024.0));



  free(phi);free(psi);free(PI);free(K);free(Kb);free(U);free(A);free(B);free(alpha);free(lambda);

}

// Compilar con : nvcc quan.cu -o quan.cux -Xcompiler /openmp


//nota: sacar el puntero en las funcinoes, si quiero optimiazr, solo dejar el phi + dt a_ij K en la funciones  por ejemplo en vez de punteros en todas las funciones.

//notas :  hacer mas free para los nodos y para cuda
