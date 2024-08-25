#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
void guardar_salida_phi(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_2.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_chi(double *data,int Nr, int T) {
  FILE *fp = fopen("dr_campo_escalar.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void cargar_coeff_centradas_df(float *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_centrada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(float) , N , arch );
  fclose(arch);

}
void cargar_coeff_atrasada_df(float *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_atrasada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(float) , N , arch );
  fclose(arch);

}
void cargar_coeff_adelantada_df(float *data, int N){

  FILE *arch;
  arch=fopen("coeff_det_adelantada.npy","rb");
  if (arch==NULL)
      exit(1);
  fread( data , sizeof(float) , N , arch );
  fclose(arch);

}

__global__ void difference_tenth(double *df,double *f,float *coef_centrada,float *coef_atrasada, float *coef_adelantada, int idx, float h, int N){
  double temp;
  temp=0;

  if (idx<11){
    for (int m=0;m<11;m++){
      temp += coef_adelantada[m]*f[idx+m];
    }
    df[idx]=temp/h; 
  }
  else if (idx > N-10){
    for (int m=-10;m<1;m++){
      temp += coef_atrasada[-m]*f[idx+m];
    }
    df[idx]=temp/h;
  }
  else{
    for (int m=0;m<11;m++){      temp += coef_centrada[m]*f[idx-5+m];
    }
    df[idx]=temp/h;
  }

}
/*__global__ double Runge_Kutta4_temporal( double func,double func_past, double yn, double h, int Nr){
  int id =threadIdx.x + blockDim.x*blockIdx.x;

  double k1 = h * func_past;
  double k2_3 = h*(func_past + 0.5* (func_past-func));  //interpolacion lineal
  double k4 = h*(func);  
  double func_plus,yn_plus;
  yn_plus = yn + 1.0/6.0 * (k1+4.0*k2_3+k4);
  return yn_plus;
}
__global__ void evolution_phi( double *phi, double t, double pi, double *alpha, double *A, double *B, double dx, double dt, int Nr){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    phi[ (t+1) * Nr + idx] =( alpha[idx] /( sqrtd(A[idx]) * B[idx] )) * dt + phi[ t * Nr + idx];
}
__global__ void evolution_phi( double *chi, double t, double *pi, double *alpha, double *A, double *B, double dx, double dt, int Nr){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func=( alpha[idx] /( sqrtd(A[idx]) * B[idx] )) * dt + phi[ t * Nr + idx];
    dr_func=diff_centrada(func,coef_a,idx,dr);
    chi[ (t+1) * Nr + idx] =Runge_Kutta4();
}*/
__global__ void Kb_dot( double *Kb, double *K, double *A,double *B, double *alpha, double *Db, double *Da,
  double *lambda, double *U, double *rho, double *S_A, double dr, double dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      Kb[idx]=0.0;
    }
    else{
      double func,func_1,func_2,func_3, dr_Db;

      func_1 = 0.5 * U[idx] + 2.0 * lambda[idx] * B[idx] / A[idx] - Db[idx] - lambda[idx] - Da[idx];
      dr_Db = 0;
      func_2 = -0.5 * Da[idx] * Db[idx] - 0.5*dr_Db + 0.25*Db[idx]*(U[idx] + 4.0*lambda[idx]*B[idx]/A[idx]) +A[idx]*K[idx]*Kb[idx];
      func_3 = S_A[idx] - rho[idx];

      func= alpha[idx]/(A[idx] * (idx*dr) )*func_1 - alpha[idx]/A[idx]*func_2 + alpha[idx]/(2.0)*func_3;
    }
  }
    //Kb=Runge_Kutta4(func);
}

__global__ void K_dot( double *K, double *Kb, double *A,double *B, double *alpha, double *Db, double *Da,
  double *lambda, double *U, double *rho, double *S_A, double *S_B, double dr, double dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      K[idx]=0.0;
    }
    else{
      double func,func_1,func_2,func_3, dr_Da;

      func_1 = K[idx]*K[idx] - 4.0*K[idx]*Kb[idx] + 6*Kb[idx]*Kb[idx];
      dr_Da = 0;
      func_2 = Da[idx]*Da[idx] + dr_Da + 2.0*Da[idx]/( idx*dr ) + 0.5*Da[idx]*(U[idx] + 4.0*lambda[idx]*B[idx]/A[idx]);
      func_3 = rho[idx] + S_A[idx] + 2.0*S_B[idx];

      func= alpha[idx]*func_1 - alpha[idx]/A[idx]*func_2 + alpha[idx]/(2.0)*func_3;
    }
  }
    //Kb=Runge_Kutta4(func);
}

__global__ void lambda_dot( double *lambda, double *Kb, double *K,double *A, double *B, double *alpha, double *Db,
  double *ja, float dr, float dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if(t==0){
   lambda[idx]=(1.0-A[idx]/B[idx])/(idx*dr);
  }
  else{
    double func, dr_Kb;
    dr_Kb = 0;

    func= 2.0*alpha[idx]*A[idx]/B[idx]*(dr_Kb - 0.5*Db[idx]*( K[idx] - 3.0*Kb[idx] ) + 0.5*ja[idx]);


    //Kb=Runge_Kutta4(func);
  }
}

__global__ void U_dot( double *U, double *Kb, double *K, double *A, double *B, double *alpha, double *Db,
  double *Da, double *lambda, double *ja, float dr, float dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if(t==0){
   U[idx]=(1.0-4.0*lambda[idx]/A[idx]);
  }
  else{
    double func,func_1,func_2,func_3, dr_K;
    dr_K = 0;
    func_1 = dr_K + Da[idx]*(K[idx] - 4.0*Kb[idx]);
    func_2 = 2.0*(K[idx] -3.0*Kb[idx])*(Db[idx] - 2.0*lambda[idx]*B[idx]/A[idx]);
    func_3 = 4.0*alpha[idx]*ja[idx];

    func= -2.0*alpha[idx] * (func_1 - func_2) - func_3;


    //Kb=Runge_Kutta4(func);
  }
}

__global__ void A_dot( double *A, double *K, double *Kb, double *alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func, dt_A,sum_A;
    dt_A = 0;
    sum_A = 0;
    func= -2.0*alpha[idx]*A[idx]*(K[idx] - 2.0*Kb[idx]);
    float b_i[4]={1.0,2.0,2.0,1.0};
    float c_i[4]={1.0,0.5,0.5,1.0};
    float a_ij[4][4]={{0,0,0,0},{0.5,0,0,0},{0,0.5,0,0},{0,0,1.0,0}};
    float k_i[4];
    double temp,temp2,temp_sum;
    temp2=0;
    temp_sum=temp;
    temp = -2.0*alpha[idx]*(K[idx] - 2.0*Kb[idx]);
    
    for (int i=0;i<4;i++){
     temp2=A[idx];
     for(int j=0;j<i;j++){
      temp2 += a_ij[i][j]*k_i[j];}
     k_i[i]=temp*temp2;
     temp_sum += b_i[i]*k_i[i]*dt*1.0/6.0;}
    A[idx] = A[idx] + temp_sum;//pasó temporal
}

__global__ void B_dot( double *B,double *Kb, double *alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      B[idx]=1.0;
    }
    else{
      double func;


      func= -2.0*alpha[idx]*B[idx]*Kb[idx];
      //B[idx]=Runge_Kutta4(func);
    }
  }
}

__global__ void Db_dot( double *Db,double *Kb, double *alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      Db[idx]=0.0;
    }
    else{
      double dr_func,func;
      func=alpha[idx]*Kb[idx];
      dr_func= 0;
    }
  }
    //B[idx]=Runge_Kutta4(-2.0*dr_func);
}

__global__ void alpha_dot( double *alpha, double *B,double *K,  double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      alpha[idx]=1.0;
    }
    else{
      double func,f_alpha;
      f_alpha=0;
      func=alpha[idx]*alpha[idx]*f_alpha*K[idx];
    }
  }
    //B[idx]=Runge_Kutta4(-func);
}

__global__ void Da_dot( double *Da,double *K, double *alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (t==0){
      Da[idx]=0.0;
    }
    else{
      double func,f_alpha, dr_func;
      f_alpha=0;
      func=alpha[idx]*f_alpha*K[idx];
      dr_func=0;
      //B[idx]=Runge_Kutta4(-dr_func);
    }
  }
}
__global__ void calculate_rho(double *rho,double *PI,double *chi, double *A, double *B, int Nr, int t ){
   int idx;
   rho[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);
}
__global__ void calculate_ja(double *ja,double *PI,double *chi, double *A, double *B, int Nr, int t ){
    int idx =threadIdx.x + blockDim.x*blockIdx.x;
    ja[idx] = -PI[idx]*chi[idx] / (sqrt(A[idx])*B[idx]);
}
__global__ void calculate_SA(double *SA,double *PI,double *chi, double *A, double *B, int Nr, int t ){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  SA[idx] = (PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);
}
__global__ void calculate_SB(double *SB,double *PI,double *chi, double *A, double *B, int Nr, int t ){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  SB[idx]=(PI[idx]*PI[idx] / (B[idx]*B[idx]) - chi[idx]*chi[idx])/(2.0*A[idx]);
}
  //Resolver el albertiano(phi) = 0
  //phi_tplus=
  //Nota: Deja todas las variables sin [], despues lo remplazas
__global__ void phi_evolution(double *phi, double *phi_t_plus, double *pi, double *A, double *B, double *alpha, float dt, int Nr, int t){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx <Nr){
    phi[ (t+1) * Nr + idx ] =  phi[ t * Nr + idx ] + pi[idx] * alpha[idx]/(sqrt(A[idx])*B[idx]) * dt ;
  }
}
/*
__global__ void chi_evolution( double *chi, double *A , double *B, double *alpha, float dt, float dr, int Nr){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<Nr){
    if (idx=Nr-1){
    }
    else if (idx=0){
    }
    phi_dot_plus=alpha[ idx+1 ]*pi[ idx+1 ]/(sqrtd(A[ idx+1 ])*B[ idx+1 ]);
    phi_dot_minus=alpha[ idx-1 ]*pi[ idx-1 ]/(sqrtd(A[ idx-1 ])*B[ idx-1 ]);
    chi[idx]=( phi_dot_plus - phi_dot_minus )/(2.0*dr) *dt +chi[idx]; 
}
}*/
 void inicial_phi(double *phi, float dr,int Nr){
  float a=0.2;
  float std=0.015;
  for (int i=0;i<Nr;i++){
   phi[i]=a*exp(-( (i*dr-0.5) /std)*((i*dr-0.5) /std));
  }
 }

void rellenar(double *f, int Nr,double num){
  for (int i=0;i<Nr;i++){
    f[i]=num;
  }
}

int main(){
  int Nr=1000;
  int Nt=10000;
  // Defino los array del host
  double *A,*B,*alpha,*phi,*chi,*PI,*lambda,*K,*Kb,*U;
  //Defino los array de device
  double *cuda_A,*cuda_B,*cuda_alpha,*cuda_phi,*cuda_chi,*cuda_PI,*cuda_K,*cuda_Kb,*cuda_U,*cuda_lambda;
  double *cuda_Db, *cuda_Da;
  float *coef_centrada, *coef_adelantada, *coef_atrasada;
  float *cuda_coef_centrada, *cuda_coef_adelantada, *cuda_coef_atrasada;

  //deltas
  float dr=1.0/Nr;
  float dt=1.0/Nt;

// mallocs
  A=(double *)malloc(Nr*sizeof(double));
  B=(double *)malloc(Nr*sizeof(double));
  alpha=(double *)malloc(Nr*sizeof(double));
  phi=(double *)malloc(Nr*sizeof(double));
  PI=(double *)malloc(Nr*sizeof(double));
  chi=(double *)malloc(Nr*sizeof(double));
  lambda=(double *)malloc(Nr*sizeof(double));
  K=(double *)malloc(Nr*sizeof(double));
  Kb=(double *)malloc(Nr*sizeof(double));
  U=(double *)malloc(Nr*sizeof(double));

  coef_centrada=(float *)malloc(11*sizeof(float));
  coef_atrasada=(float *)malloc(11*sizeof(float));
  coef_adelantada=(float *)malloc(11*sizeof(float));

cargar_coeff_centradas_df(coef_centrada,11);
cargar_coeff_atrasada_df(coef_atrasada,11);
cargar_coeff_adelantada_df(coef_adelantada,11);
for (int i=0;i<11;i++){
  printf("coef : %f\n",coef_atrasada[i]);
}
//condiciones iniciales
inicial_phi(phi,dr,Nr);
//inicial_chi(chi,phi,coef_centrada,coef_atrasada,coef_adelantada,dr,Nr);
rellenar(PI, Nr, 0);
rellenar(K,Nr,0.0);
rellenar(Kb,Nr,0.0);
//rellenar(Da,Nr,0.0);
//rellenar(Db,Nr,0.0);
rellenar(alpha, Nr, 1.0);
rellenar(B,Nr,1.0);

//pendiente inicial del A...
// cuda mallocs
guardar_salida_phi(phi,Nr,1);
guardar_salida_chi(chi,Nr,1);



cudaMalloc ((void**)cuda_phi, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_alpha, Nr*sizeof(double) );


cudaMalloc ((void**)cuda_A, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_B, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_PI, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_chi, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_lambda, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_K, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_Kb, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_U, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_Da, Nr*sizeof(double) );
cudaMalloc ((void**)cuda_Db, Nr*sizeof(double) );

int thread=1024;
dim3 bloque(thread);
dim3 grid((int)ceil((float)(Nr)/thread));


B_dot<<<grid,bloque>>>( cuda_B, cuda_Kb, cuda_alpha, dt, Nr, t);
Db_dot<<<grid,bloque>>>( cuda_Db, cuda_Kb, cuda_alpha, dt, Nr, t);
alpha_dot<<<grid,bloque>>>( cuda_alpha, cuda_B, cuda_K, dt, Nr, t);
Da_dot<<<grid,bloque>>>( cuda_Da, cuda_K, cuda_alpha, dt, Nr, t);


rellenar(A,Nr,1.0);
rellenar(PI, Nr, 0.0);
inicial_phi(phi,dr,Nr);


cudaMemcpy( cuda_phi, phi, Nr*sizeof(double), cudaMemcpyHostToDevice );
cudaMemcpy( cuda_PI, PI, Nr*sizeof(double), cudaMemcpyHostToDevice );
cudaMemcpy( cuda_A, A, Nr*sizeof(double), cudaMemcpyHostToDevice );

difference_tenth<<<grid,bloque>>>(chi,phi,dr,Nr); 


Kb_dot<<<grid,bloque>>>( cuda_K, cuda_Kb,cuda_A,cuda_B, cuda_alpha, 
  cuda_Db, cuda_Da, cuda_lambda,cuda_U,cuda_rho, cuda_SA, dr, dt, Nr, t);

K_dot<<<grid,bloque>>>( cuda_Kb, cuda_K,cuda_A,cuda_B, cuda_alpha, 
  cuda_Db, cuda_Da, cuda_lambda,cuda_U,cuda_rho, cuda_SA, cuda_SB, dr, dt, Nr, t);

lambda_dot<<<grid,bloque>>>( cuda_lambda, cuda_Kb,cuda_K,cuda_A,cuda_B, cuda_alpha, 
  cuda_Db, cuda_ja, dr, dt, Nr, t);

U_dot<<<grid,bloque>>>( cuda_lambda, cuda_Kb,cuda_K,cuda_A,cuda_B, cuda_alpha, 
  cuda_Db, cuda_lamda, cuda_ja, dr, dt, Nr, t);

cudaMemcpy( phi, cuda_phi, Nr*sizeof(double), cudaMemcpyDeviceToHost );
/*
//mmcopy
 cudaMemcpy( cuda_phi, phi, Nr*sizeof(double), cudaMemcpyHostToDevice );



 for(int t=0; t<Nt;t++){


 }

 cudaMemcpy( phi, cuda_phi, Nr*sizeof(double), cudaMemcpyDeviceToHost );*/




  free(phi);free(chi);free(PI);free(K);free(Kb);free(U);free(A);free(B);free(alpha);free(lambda);
  /*cudaFree(cuda_phi);cudaFree(cuda_chi);cudaFree(cuda_A);cudaFree(cuda_B);cudaFree(cuda_alpha);
  cudaFree(cuda_K);cudaFree(cuda_Kb);cudaFree(cuda_lambda);cudaFree(cuda_U);*/

}
