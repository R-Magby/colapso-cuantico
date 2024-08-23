#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>

__global__ double diff_centrada(double f, float coef_a,int idx, float h){
  double df;
  for (int m=0;m<11;m++){
    df += coef_a[m]*f[idx-5+m];
  }
  return df/h;
}
__global__ double diff_atrasada(double f, float coef_a,int idx, float h){
  double df;
  for (int m=-10;m<1;m++){
    df += coef_a[m+10]*f[idx+m];
  }
  return df/h;
}
__global__ double diff_adelantada(double f, float coef_a,int idx, float h,){
  double df;
  for (int m=-idx;m<11;m++){
    df += coef_a[m+idx]*f[idx+m];
  }
  return df/h;
}
__global__ void diff()
__global__ double Runge_Kutta4_temporal( double func,double func_past, double yn, double h, int Nr){
  int id =threadIdx.x + blockDim.x*blockIdx.x;

  double k1 = h * func_past;
  double k2_3 = h*(func_past + 0.5* (func_past-func));  //interpolacion lineal
  double k4 = h*(func);  
  double func_plus;
  yn_plus = y_n + 1.0/6.0 * (k1+4.0*k2_3+k4);
  return yn_plus
}
__global__ void evolution_phi( double phi, double t, double pi, double alpha, double A, double B, double dx, double dt, int Nr){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    phi[ (t+1) * Nr + idx] =( alpha[idx] /( sqrtd(A[idx]) * B[idx] )) * dt + phi[ t * Nr + idx];
}
__global__ void evolution_phi( double chi, double t, double pi, double alpha, double A, double B, double dx, double dt, int Nr){
  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func=( alpha[idx] /( sqrtd(A[idx]) * B[idx] )) * dt + phi[ t * Nr + idx];
    dr_func=diff_centrada(func,coef_a,idx,dr);
    chi[ (t+1) * Nr + idx] =Runge_Kutta4();
}
__global__ void Kb_dot( double Kb, double K, double A,double B, double alpha, double Db, double Da,
  double lamba, double U, double rho, double S_A, double dx, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func,func_1,func_2,func_3, dr_Db;

    func_1 = 0.5 * U + 2.0 * lambda * B / A - Db - lambda - Da;
    dr_Db = 0;
    func_2 = -0.5 * Da * Db - 0.5*dr_Db + 0.25*Db*(U + 4.0*lambda*B/A) +A*K*Kb;
    func_3 = S_A - rho;

    func= alpha/(A* (idx*dr) )*func_1 - alpha/A*func_2 + alpha/(2.0)*func_3;


    Kb=Runge_Kutta4(func);
}

__global__ void K_dot( double K, double Kb, double A,double B, double alpha, double Db, double Da,
  double lamba, double U, double rho, double S_A, double dx, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func,func_1,func_2,func_3, dr_Da;

    func_1 = K*K - 4.0*K*Kb + 6*Kb*Kb;
    dr_Da = 0;
    func_2 = Da*Da + dr_Da + 2.0*Da/( idx*dr ) + 0.5*Da*(U + 4.0*lambda*B/A);
    func_3 = rho + S_A + 2.0*S_B;

    func= alpha*func_1 - alpha/A*func_2 + alpha/(2.0)*func_3;


    Kb=Runge_Kutta4(func);
}

__global__ void lambda_dot( double lamba, double Kb, double K,double A, double B, double alpha, double Db,
  double ja, float dr, float dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if(t==0){
   lambda[idx]=(1.0-A[idx]/B[idx])/(idx*dr)
  }
  else{
    double func, dr_Kb;
    dr_Kb = 0;

    func= 2.0*alpha*A/B*(dr_Kb - 0.5*Db*( K - 3.0*Kb ) + 0.5*ja);


    Kb=Runge_Kutta4(func);
  }
}

__global__ void U_dot( double U, double Kb, double K, double A, double B, double alpha, double Db,
  double Da, double lambda, double ja, float dx, float dt, int Nr, int t){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
  if(t==0){
   U[idx]=(1.0-4.0*lambda[idx]/A[idx]);
  }
  else{
    double func,func_1,func_2,func_3, dr_Kb;
    dr_K = 0;
    func_1 = dr_K + Da*(K - 4.0*Kb);
    func_2 = 2.0*(K -3.0*Kb)*(Db - 2.0*lambda*B/A);
    func_3 = 4.0*alpha*ja;

    func= -2.0*alpha * (func_1 - func_2) - func_3;


    Kb=Runge_Kutta4(func);
  }
}

__global__ void A_dot( double A, double K, double Kb, double alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func, dt_A,sum_A;
    dt_A = 0;
    sum_A = 0;
    func= -2.0*alpha*A*(K - 2.0*Kb);
    float b_i[4]={1.0,2.0,2.0,1.0};
    flaot c_i[4]={1.0,0.5,0.5,1.0};
    float a_ij[4][4]={{0,0,0,0},{0.5,0,0,0},{0,0.5,0,0},{0,0,1.0,0}};
    float k_i[4];
    double temp,temp2,k_i,temp_sum;
    temp2=0;
    temp_sum=temp;
    temp = -2.0*alpha[idx]*(K[idx] - 2.0*Kb[idx]);
    
    for (int i=0;i<4;i++){
     temp2=A[idx];
     for(int j=0;j<i;j++){
      temp2 += a_ij[i][j]*k_i[j];}
     k_i[i]=temp*temp2;
     temp_sum += b[i]*k_i[i]*h*1.0/6.0;}
    A[idx] = A[idx] + temp_sum;//pasó temporal
}

__global__ void B_dot( double B,double Kb, double alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func;
    dr_K = 0;


    func= -2.0*alpha*B*Kb;
    B=Runge_Kutta4(func);
}

__global__ void Db_dot( double B,double Kb, double alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double dr_func;
    func=alpha*Kb;
    dr_func= 0;
    B=Runge_Kutta4(-2.0*dr_func);
}

__global__ void alpha_dot( double B,double Kb, double alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func,f_alpha;
    f_alpha=0;
    func=alpha*alpha*f_alpha*K;
    
    B=Runge_Kutta4(-func);
}

__global__ void Da_dot( double B,double Kb, double alpha, double dt, int Nr){

  int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double func,f_alpha, dr_func;
    f_alpha=0;
    func=alpha*f_alpha*K;
    dr_func=0
    B=Runge_Kutta4(-dr_func);
}
__global__ void calculate_rho(double rho,double PI,double chi, double A, double B, int Nr, int t ){
   int idx;
   rho=(PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);
}
__global__ void calculate_ja(double ja,double PI,double chi, double A, double B, int Nr, int t ){
   int idx;
   ja = -PI[idx]*chi[idx] / (sqrt(A[idx])*B[idx]);
}
__global__ void calculate_SA(double SA,double PI,double chi, double A, double B, int Nr, int t ){
   int idx;
   SA=(PI[idx]*PI[idx] / (B[idx]*B[idx]) + chi[idx]*chi[idx])/(2.0*A[idx]);
}
__global__ void calculate_SB(double SB,double PI,double chi, double A, double B, int Nr, int t ){
   int idx;
   SB=(PI[idx]*PI[idx] / (B[idx]*B[idx]) - chi[idx]*chi[idx])/(2.0*A[idx]);
}
  //Resolver el albertiano(phi) = 0
  //phi_tplus=
  //Nota: Deja todas las variables sin [], despues lo remplazas
__global__ void phi_evolution(double phi, double phi_t_plus, double pi, double A, double B, double alpha, float dt, int Nr, int t){
idx= threadlock.x +
If (idx <Nr){
phi[ t * Nr + idx ] = pi[idx] * alpha[idx]/(sqrtf(A[idx])*B[idx]) * dt + phi[ t * Nr + idx ];}
}

__global__ void chi_evolution( double chi, double A , double B, double alpha, float dt , int Nr){
idx=
If (idx<Nr){
If (idx=Nr-1){
}
else if (idx=0){
}
phi_dot_plus=alpha[ idx+1 ]*pi[ idx+1 ]/(sqrtd(A[ idx+1 ])*B[ idx+1 ]);
phi_dot_minus=alpha[ idx-1 ]*pi[ idx-1 ]/(sqrtd(A[ idx-1 ])*B[ idx-1 ]);
chi[idx]=( phi_dot_plus - phi_dot_minus )/(2.0*dr) *dt +chi[idx]; }
}
 void incial_phi(double phi, float dr,int Nr){
  float a=0.02;
  float std=0.15;
  for (int i=0;i<Nr;i++){
   phi[i]=a*exp(-(i*dr/std)*(i*dr/std))
  }
 }
void incial_chi(double chi, double phi, float dr,int Nr){
 for(int i=0;i<Nr;i++){
  chi[i]=difference_tenth(phi,dr,i);
 }
}
void rellenar(double f, int Nr,double num){
 for (int i=0;i<Nr;i++){
  f[i]=num;
 }
}
int main(){
 int Nr=1000;
 int Nt=10000;
// Defino los array del host
 double A,B,alpha,phi,chi,PI,lambda,K,Kb,U;
//Defino los array de device
 double cuda_A,cuda_B,cuda_alpha,cuda_phi,cuda_chi,cuda_PI,cuda_K,cuda_Kb,cuda_U;

//deltas
 float dr=1.0/Nr;
 float dt=1.0/Nt;

// mallocs
  A=(double)malloc(Nr*sizeof(double));


//condiciones iniciales
inicial_phi(phi,dr,Nr);
inicial_chi(chi,phi,dr,Nr);
rellenar(PI, Nr, 0);
rellenar(K,Nr,0.0);
rellenar(Kb,Nr,0.0);
rellenar(Da,Nr,0.0);
rellenar(Db,Nr,0.0);
rellenar(alpha, Nr, 1.0);
rellenar(B,Nr,1.0);
//pendiente inicial del A...
// cuda mallocs

  cudaMalloc ((void**)cuda_A, Nr*sizeof(double) );

//mmcopy
 cudaMemcpy( cuda_phi, phi, Nr*sizeof(double), cudaMemcpyHostToDevice );
 for(int t=0; t<Nt;t++){


 }

 cudaMemcpy( phi, cuda_phi, Nr*sizeof(double), cudaMemcpyDeviceToHost );




  free(phi);free(chi);free(PI);free(K)free(Kb);free(U),free(A);free(B);free(alpha);free(lambda);
  cudaFree(cuda_phi);cudaFree(cuda_chi);cudaFree(cuda_A);cudaFree(cuda_B);cudaFree(cuda_alpha);
  cudaFree(cuda_K);cuadFree(cuda_Kb);cudaFree(cuda_lambda);cudaFree(cuda_U);

}
