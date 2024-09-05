#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>
#define order 3
double coefficient_adelantada[order];
double coefficient_centrada[order];
double coefficient_atrasada[order];
double b_i[4];
double c_i[4];
double a_ij[4];

void guardar_salida_phi(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_2.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_alpha(double *data,int Nr, int T) {
  FILE *fp = fopen("campo_escalar_lapso.dat", "wb");
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
        temp += coefficient_adelantada[m]*f[idx+m];
      }
      df[idx]=temp/h; 
    }
    else if (idx > N-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*f[idx+m];
      }
      df[idx]=temp/h;
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*f[idx-(int)(order-1)/2+m];
      }
      df[idx]=temp/h;
    }
  }
}
double difference_tenth_RK( double *f,double *Kf, double *a_ij, int idx,int s, double h,double dt, int Nr, int symmetric){
  double temp ;
  temp=0.0;
    if (idx==0){
      if (symmetric==0){
        temp=0.0;
      }
      else if (symmetric == 1){
        for (int m=1;m<order;m++){
          temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
        }

      }
    }
    else if (idx<(int)(order-1)/2 && idx>0){
      for (int m=0;m<order;m++){
        temp += coefficient_adelantada[m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }
    else if (idx > Nr-(int)(order-1)/2-1){
      for (int m=-order+1;m<1;m++){
        temp += coefficient_atrasada[-m]*(f[idx+m] + dt*a_ij[s]*Kf[(s)*Nr+idx+m]);
      }
    }
    else{
      for (int m=0;m<order;m++){
        temp += coefficient_centrada[m]*(f[idx-(int)(order-1)/2+m] + dt*a_ij[s]*Kf[(s)*Nr+idx-(int)(order-1)/2+m]);
      }
    }
    return temp/h;
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

double Kb_dot( double Kb, double K, double A,double B, double alpha, double Db, double Da,
  double lambda, double U, double rho, double S_A,double dr_Db, double dr, int idx, int Nr, int t){
    double epsilon=dr*0.5;
    double rs;
        double Mp=1.0/sqrt(8.0*3.1415);

    if (t==0){
      return 0.0;
    }
    else{
      double func,func_1,func_2,func_3;
      if (idx==0){
      rs=idx*dr+epsilon;}
      else{
      rs=idx*dr;
      }
      func_1 = 0.5 * U + 2.0 * lambda * B / A - Db - lambda - Da;
      func_2 = -0.5 * Da * Db - 0.5*dr_Db + 0.25*Db*(U + 4.0*lambda*B/A) +A*K*Kb;
      func_3 = S_A - rho;

      func= alpha/(A * rs)*func_1 + alpha/A*func_2 + alpha/(2.0*Mp*Mp)*func_3;
      return func;
    }
 }

double K_dot( double K, double Kb, double A,double B, double alpha, double Db, double Da,
  double lambda, double U, double rho, double S_A, double S_B, double dr_Da, double dr, int idx, int Nr, int t){
    double epsilon=dr*0.5;
    double rs;
    double Mp=1.0/sqrt(8.0*3.1415);
    if (t==0){
      return 0.0;
    }
    else{
      double func,func_1,func_2,func_3;
      if (idx==0){
        rs=idx*dr+epsilon;}
      else{
        rs=idx*dr;
      }
      func_1 = K*K - 4.0*K*Kb + 6.0*Kb*Kb;
      func_2 = Da*Da + dr_Da + 2.0*Da/rs - 0.5*Da*(U + 4.0*lambda*B/A);
      func_3 = rho + S_A + 2.0*S_B;

      func= alpha*func_1 - alpha/A*func_2 + alpha/(2.0*Mp*Mp)*func_3;
      return func;
    }
}

double lambda_dot( double lambda, double Kb, double K,double A, double B, double alpha, double Db,
  double ja,double dr_Kb, double dr, int idx, int Nr, int t){
  double epsilo=dr*0.5;
      double Mp=1.0/sqrt(8.0*3.1415);
    double rs;
  if(t==0){
      if (idx==0){
      rs=idx*dr+epsilo;}
      else{
      rs=idx*dr;
      }
    return (1.0-A/B)/(rs);
  }
  else{
    double func;
      if (idx==0){
      rs=idx*dr+epsilo;}
      else{
      rs=idx*dr;
      }

    func= 2.0*alpha*A/B*(dr_Kb - 0.5*Db*( K - 3.0*Kb ) + 0.5*ja/(Mp*Mp));

    //func= 2.0*alpha*A/B*(( K - 3.0*Kb )/rs );   
    return func;  
    }
  }

double U_dot( double U, double Kb, double K,double A, double B, double alpha, double Db,
  double Da, double lambda, double ja, double dr_K, double dr_A,  double dr, double dt, int Nr, int t){
    double Mp=1.0/sqrt(8.0*3.1415);

  if(t==0){
    return (dr_A-4.0*lambda)/A;
  }
  else{
    double func,func_1,func_2,func_3;
    func_1 = dr_K + Da*(K - 4.0*Kb);
    func_2 = 2.0*(K -3.0*Kb)*(Db - 2.0*lambda*B/A);
    func_3 = 4.0*alpha*ja/(Mp*Mp);

    func= -2.0*alpha * (func_1 - func_2) - func_3;
    return func; 
  }
}

double A_dot( double A, double K, double Kb, double alpha, double dt, int Nr){

    double func;

    func= -2.0*alpha*A*(K - 2.0*Kb);

    return func; 
}

double B_dot( double B,double Kb, double alpha, double dt, int Nr){

  return -2.0*alpha*B*Kb; 
}

double f_Db(double *alpha, double *Kb,double *K_a,double *K_Kb, double *a_ij,double dt, int idx, int s,int Nr,int t){
  return (alpha[ idx]+dt*a_ij[s]*K_a[(s)*Nr + idx])*(Kb[idx]+dt*a_ij[s]*K_Kb[(s)*Nr + idx]); ;
  }
double Db_dot(double *Kb, double *alpha,double *K_a,double *K_Kb,  double *a_ij,double dt, double dr, int idx, int s,int Nr,int t){
  double temp;
  temp=0.0;

    if (idx==0){

      temp=0.0;
    }
    else if (idx<(int)(order-1)/2 && idx>0){
        for (int m=0;m<order;m++){
      
        temp += coefficient_adelantada[m]*f_Db(alpha,Kb,K_a,K_Kb,a_ij,dt,idx+m,s,Nr,t);
        }
    
    }
    else if (idx > Nr-(int)(order-1)/2-1){
        for (int m=-order+1;m<1;m++){
          temp += coefficient_atrasada[-m]*f_Db(alpha,Kb,K_a,K_Kb,a_ij,dt,idx+m,s,Nr,t );
        }
    }
    else{
        for (int m=0;m<order;m++){
          temp += coefficient_centrada[m]*f_Db(alpha,Kb,K_a,K_Kb,a_ij,dt,idx-(int)(order-1)/2+m,s,Nr,t);
        }
    }
    return -2.0*temp/dr; ;
  }

double alpha_dot( double alpha,double K,  double dt, int Nr){
    return -2.0*alpha*K;
}

double Da_dot(double dr_K, double dt, int Nr,int t){
      return -2.0*dr_K;
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
double phi_evolution(double pi, double A, double B, double alpha, double dt, int Nr, int t){
    return pi * alpha/(sqrt(A)*B);
}
double f_chi(double *A, double *B, double *alpha, double *PI,double *K_A,
  double *K_a,double *K_B,double *K_PI, double *a_ij, double dt, int idx, int s,int Nr,int t){
  return (alpha[idx]+dt*a_ij[s]*K_a[(s)*Nr + idx])*(PI[idx]+dt*a_ij[s]*K_PI[(s)*Nr + idx])
  /(sqrt(A[idx]+dt*a_ij[s]*K_A[(s)*Nr + idx])*(B[idx]+dt*a_ij[s]*K_B[(s)*Nr + idx]));
  }

double chi_evolution( double *chi, double *A, double *B, double *alpha, double *PI,double *K_A,
  double *K_a,double *K_B,double *K_PI, double *a_ij, int idx, int s, double dt, double dr, int Nr,int t){
  double temp;
  temp=0.0;

    if (idx==0){

      temp= 0.0;
    }
    else if (idx<(int)(order-1)/2 && idx>0){
    for (int m=0;m<order;m++){
      
      temp += coefficient_adelantada[m]*f_chi(A,B,alpha,PI,K_A,K_a,K_B,K_PI,a_ij,dt,idx+m,s,Nr,t);
    }
    
  }
  else if (idx > Nr-(int)(order-1)/2-1){
    for (int m=-order+1;m<1;m++){
      temp += coefficient_atrasada[-m]*f_chi(A,B,alpha,PI,K_A,K_a,K_B,K_PI,a_ij,dt,idx+m,s,Nr,t );
    }
  }
  else{
    for (int m=0;m<order;m++){
      temp += coefficient_centrada[m]*f_chi(A,B,alpha,PI,K_A,K_a,K_B,K_PI,a_ij,dt,idx-(int)(order-1)/2+m,s,Nr,t);
    }
  }
      return temp/dr; ;
  }

double f_PI(double *A, double *B, double *alpha, double *chi,double *K_A,
    double *K_a,double *K_B,double *K_chi, double *a_ij, int idx, int s, double dt,double dr,int Nr,int t){
    double r2;
    double epsilon=dr*0.5;
    if(idx==0){
     r2=epsilon; }
     else{
      r2=idx*dr*idx*dr;
     }

    return  (alpha[idx]+dt*a_ij[s]*K_a[(s)*Nr + idx])*(B[idx]+dt*a_ij[s]*K_B[(s)*Nr + idx])*(chi[idx]+dt*a_ij[s]*K_chi[(s)*Nr + idx])*r2
            /(sqrt(A[idx]+dt*a_ij[s]*K_A[(s)*Nr + idx]));
    }
double PI_evolution( double *PI, double *A, double *B, double *alpha, double *chi,double *K_A,
  double *K_a,double *K_B,double *K_chi, double *a_ij, int idx, int s, double dt, double dr, int Nr,int t){
  double temp;
  double epsilon=dr*0.5;

  temp=0.0;
    if (idx==0){
        for (int m=1;m<order;m++){
        
            temp += coefficient_adelantada[m]*f_PI(A,B,alpha,chi,K_A,K_a,K_B,K_chi,a_ij,idx+m,s,dt,dr,Nr,t);
        } 
      return temp/dr*(1.0/(idx*dr*idx*dr+epsilon));
    }
    else if (idx<(int)(order-1)/2 && idx>0){
        for (int m=0;m<order;m++){
        
            temp += coefficient_adelantada[m]*f_PI(A,B,alpha,chi,K_A,K_a,K_B,K_chi,a_ij,idx+m,s,dt,dr,Nr,t);
        } 
        return temp/dr*(1.0/(idx*dr*idx*dr));

    }
    else if (idx > Nr-(int)(order-1)/2-1){
        for (int m=-order+1;m<1;m++){

            temp += coefficient_atrasada[-m]*f_PI(A,B,alpha,chi,K_A,K_a,K_B,K_chi,a_ij,idx+m,s,dt,dr,Nr,t);
        }
        return temp/dr*(1.0/(idx*dr*idx*dr));

    }
    else{
        for (int m=0;m<order;m++){

            temp += coefficient_centrada[m]*f_PI(A,B,alpha,chi,K_A,K_a,K_B,K_chi,a_ij,idx-(int)(order-1)/2+m,s,dt,dr,Nr,t);
        }
        return temp/dr*(1.0/(idx*dr*idx*dr));

    }
}
void inicial_phi(double *phi, double dr,int Nr){
  double a=0.002;
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
  double epsilon=dr*0.5;
  double Mp=1.0/sqrt(8.0*3.1415);
    double a=0.002;
  double std=1.5;
  A[0]=1.0;
  ks[0]=0.0;


  for (int idx=1;idx<Nr;idx++){

    sumas=0.0;
    for (int s=0 ; s<4 ;s++){
      A0=(A[idx-1]+dr*a_ij[s]*ks[s]);

      //ks[s+1]= A0*((1.0/rs)*(1.0-A0)+rs*chi[idx-1]*chi[idx-1]*0.5 + 0.0);
      //chi0=0.01*(rs-5.0)*(-2.0/(0.2*0.2))*exp(-1.0*((rs-5.0)/0.2)*((rs-5.0)/0.2));
        double temp;
        temp=0.0;

            if (idx==1){
              for (int m=0;m<order;m++){
                  rs= dr*((idx-1)+m)+dr*c_i[s];

                temp += coefficient_adelantada[m]*dfr(rs,a,std);
              } 

            }
            else if (idx > Nr-(int)(order-1)/2-1){
              for (int m=-order+1;m<1;m++){
              rs= dr*((idx-1)+m)+dr*c_i[s];

              temp +=  coefficient_atrasada[-m]*dfr(rs,a,std);
              }
            }
            else{
            for (int m=0;m<order;m++){
              rs= dr*((idx-1)-(int)(order-1)/2+m)+dr*c_i[s];

              temp += coefficient_centrada[m]*dfr(rs,a,std);
            }
            
          }
        if (idx-1==0){
        rs= dr*(idx-1)+dr*c_i[s] + epsilon;
      }
      else{
        rs= dr*(idx-1)+dr*c_i[s];
      }
            chi0= temp/dr; 
      ks[s+1]= A0*((1.0/rs)*(1.0-A0)+rs*chi0*chi0*0.5 + 0.0);

      sumas += b_i[s]*ks[s+1];
    }
   A[idx]=A[idx-1]+dr*sumas;
  }
}
void rellenar(double *f, int Nr,double num){
  for (int i=0;i<Nr;i++){
    f[i]=num;
  }
}


/*notas: primera sol buena fue con Nr=500 ,dr=5.0/Nr y dt=5/20.000, con amplitud 0.0002 y std=1.5
Cuando aumento la amplitud por ejemplo a a=0.02 las simulaciones divergen con Nr=500 dr=50/r y dt=5.0/80.000
divergen en el rebote...*/
int main(){
  int Nr=500;
  int Nt=400000;
  int minit= Nt/100;
  // Defino los array del host
  double *A,*B,*alpha,*phi,*chi,*PI,*lambda,*K,*Kb,*U, *Da, *Db, *temp_phi, *temp_Kb;
  double *rho,*ja,*SA,*SB;
  //double *K_s,*b_i, *a_ij, *c_i;
  double *coef_atrasada,*coef_centrada,*coef_adelantada;

  //deltas
  double dr=50.0/Nr;
  //double dt=5.0/10000;
  double dt=5.0/80000; //el bueno
  //double dt=4.0/20000;
  printf("dr=%f , dt=%f",dr,dt);
// mallocs
phi=(double *)malloc(Nr*minit*sizeof(double));
temp_Kb=(double *)malloc(15000*Nr*sizeof(double));
temp_phi=(double *)malloc(Nr*sizeof(double));

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



  b_i[0]=1.0/6.0;b_i[1]=1.0/3.0;b_i[2]=1.0/3.0;b_i[3]=1.0/6.0;
  a_ij[0]=0.0;a_ij[1]=0.5;a_ij[2]=0.5;a_ij[3]=1.0;
  c_i[0]=0.0;c_i[1]=0.5;c_i[2]=0.5;c_i[3]=1.0;


cargar_coeff_centradas_df(coef_centrada,order);
cargar_coeff_atrasada_df(coef_atrasada,order);
cargar_coeff_adelantada_df(coef_adelantada,order);
for (int i=0; i < order ;i++){
  coefficient_centrada[i]=coef_centrada[i];
  coefficient_adelantada[i]=coef_adelantada[i];
  coefficient_atrasada[i]=coef_atrasada[i];
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
  lambda[idx]=lambda_dot( lambda[idx], Kb[idx] , K[idx], A[idx], B[idx], alpha[idx], Db[idx], ja[idx],0.0, dr, idx, Nr, 0);
  U[idx]=U_dot( U[idx], Kb[idx] , K[idx] , A[idx] , B[idx] ,  alpha[idx] , Db[idx] ,Da[idx],  lambda[idx] , ja[idx],0.0,dr_A0[idx], dr, dt, Nr, 0);
  //printf("r : %d\n",idx);
  //printf("U : %.15f\n",U[idx]);
  //printf("lambda : %.15f\n",lambda[idx]);
}
guardar_salida_chi(A,Nr,1.0);




//pendiente inicial del A...
// cuda mallocs

printf("Condiciones iniciales: check\n");

double *K1,*K2,*K3,*K4,*K5,*K6,*K7,*K8,*K9,*K10,*K11,*K12;
int stage=4,num_ecu=9;
double dr_Db, dr_Da, dr_K,dr_Kb,dr_A;
//dr_Db = (double *)malloc(Nr*sizeof(double));
//dr_Da = (double *)malloc(Nr*sizeof(double));
//dr_K = (double *)malloc(Nr*sizeof(double));
//dr_Kb = (double *)malloc(Nr*sizeof(double));
//dr_temp = (double *)malloc(Nr*sizeof(double));


//El U y lambda son los unicos que no aparecen en derivadas
//es necesario que cada K tenga dimensiones  [ stage x Nr ]
//para hacer las diferencias finitas.

K1=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K2=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K3=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K4=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K5=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K6=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K7=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K8=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K9=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K10=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K11=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);
K12=(double*)malloc(sizeof(double)*( stage + 1 )*Nr);



float beta=1.0/100000;

printf("Rho antes de la simulacion rho = %.15f\n",conservacion(rho,dr,Nr));



    
for(int t=1;t< Nt ;t++){


  rellenar(K10,(stage+1)*Nr,0.0);
  rellenar(K11,(stage+1)*Nr,0.0);
  rellenar(K12,(stage+1)*Nr,0.0);
  rellenar(K1,(stage+1)*Nr,0.0);
  rellenar(K2,(stage+1)*Nr,0.0);
  rellenar(K3,(stage+1)*Nr,0.0);  
  rellenar(K4,(stage+1)*Nr,0.0);
  rellenar(K5,(stage+1)*Nr,0.0);
  rellenar(K6,(stage+1)*Nr,0.0);  
  rellenar(K7,(stage+1)*Nr,0.0);
  rellenar(K8,(stage+1)*Nr,0.0);
  rellenar(K9,(stage+1)*Nr,0.0);  

  calculate_rho(rho,PI,chi,A,B,Nr);
  calculate_SA(SA,PI,chi,A,B,Nr);
  calculate_SB(SB,PI,chi,A,B,Nr);
  calculate_ja(ja,PI,chi,A,B,Nr);

  for(int s=0 ; s < stage ; s++){
    for(int idx=0;idx<Nr;idx++){
      if (s==0){
        K1[idx]=0.0;K2[idx]=0.0;K3[idx]=0.0;K4[idx]=0.0;
        K5[idx]=0.0;K6[idx]=0.0;K7[idx]=0.0;K8[idx]=0.0;
        K9[idx]=0.0;K10[idx]=0.0;K11[idx]=0.0;K12[idx]=0.0;
      }

      //if(t%1==0 && idx==Nr-1){
      //printf("drK:%f",dr_K);}

 
      K12[ (s+1)*Nr + idx ]=phi_evolution(PI[idx] + dt*a_ij[s]*K10[s*Nr+idx],A[idx] + dt*a_ij[s]*K9[s*Nr+idx],B[idx] + dt*a_ij[s]*K1[s*Nr+idx],alpha[idx] + dt*a_ij[s]*K3[s*Nr+idx],dt, dr, Nr);
      K10[ (s+1)*Nr + idx ]=PI_evolution(PI,A,B,alpha,chi,K9,K3,K1,K11,a_ij, idx, s, dt, dr, Nr,t);
      K11[ (s+1)*Nr + idx ]=chi_evolution(chi,A,B,alpha,PI,K9,K3,K1,K10,a_ij, idx, s, dt, dr, Nr,t);

      K9[ (s+1)*Nr + idx ]=A_dot( A[idx] + dt*a_ij[s]*K9[s*Nr +idx],K[idx] + dt*a_ij[s]*K6[s*Nr +idx],Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], alpha[idx] +dt*a_ij[s]*K3[s*Nr +idx], dt, Nr);
      K1[ (s+1)*Nr + idx ]=B_dot( B[idx] + dt*a_ij[s]*K1[s*Nr +idx], Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], dt, Nr);
      K2[ (s+1)*Nr + idx ]=Db_dot(Kb , alpha,K3,K5,a_ij, dt,dr,idx,s, Nr, t);

      K3[ (s+1)*Nr + idx ]=alpha_dot( alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], K[idx] + dt*a_ij[s]*K6[s*Nr +idx], dt, Nr);
      dr_K= difference_tenth_RK(K, K6,a_ij, idx,s, dr,dt,Nr,0); 

      K4[ (s+1)*Nr + idx ]=Da_dot( dr_K, dt, Nr,t);

      
      dr_Da=difference_tenth_RK( Da, K4,a_ij, idx,s, dr,dt,Nr,1); 
      K6[ (s+1)*Nr + idx ]=K_dot( K[idx] + dt*a_ij[s]*K6[s*Nr +idx], Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], A[idx] + dt*a_ij[s]*K9[s*Nr +idx], B[idx] + dt*a_ij[s]*K1[s*Nr +idx], alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], Db[idx] + dt*a_ij[s]*K2[s*Nr +idx], Da[idx] + dt*a_ij[s]*K4[s*Nr +idx], lambda[idx] + dt*a_ij[s]*K7[s*Nr +idx], U[idx] + dt*a_ij[s]*K8[s*Nr +idx] , rho[idx], SA[idx], SB[idx],dr_Da, dr, idx, Nr, t);
      
      dr_Db=difference_tenth_RK( Db, K2,a_ij, idx,s, dr,dt,Nr,1); 
      
      K5[ (s+1)*Nr + idx ]=Kb_dot( Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], K[idx] + dt*a_ij[s]*K6[s*Nr +idx], A[idx] + dt*a_ij[s]*K9[s*Nr +idx], B[idx] + dt*a_ij[s]*K1[s*Nr +idx], alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], Db[idx] + dt*a_ij[s]*K2[s*Nr +idx], Da[idx] + dt*a_ij[s]*K4[s*Nr +idx], lambda[idx] + dt*a_ij[s]*K7[s*Nr +idx], U[idx] + dt*a_ij[s]*K8[s*Nr +idx],rho[idx], SA[idx],dr_Db, dr, idx, Nr, t);



      dr_Kb=difference_tenth_RK(Kb, K5,a_ij, idx,s, dr,dt,Nr,0);      
      K7[ (s+1)*Nr + idx ]=lambda_dot( lambda[idx] + dt*a_ij[s]*K7[s*Nr +idx], Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], K[idx] + dt*a_ij[s]*K6[s*Nr +idx], A[idx] + dt*a_ij[s]*K9[s*Nr +idx], B[idx] + dt*a_ij[s]*K1[s*Nr +idx], alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], Db[idx] + dt*a_ij[s]*K2[s*Nr +idx], ja[idx], dr_Kb, dr, idx, Nr, t);
      dr_K= difference_tenth_RK(K, K6,a_ij, idx,s, dr,dt,Nr,0); 
      K8[ (s+1)*Nr + idx ]=U_dot( U[idx] + dt*a_ij[s]*K8[s*Nr +idx], Kb[idx] + dt*a_ij[s]*K5[s*Nr +idx], K[idx] + dt*a_ij[s]*K6[s*Nr +idx], A[idx] + dt*a_ij[s]*K9[s*Nr +idx], B[idx] + dt*a_ij[s]*K1[s*Nr +idx],  alpha[idx] + dt*a_ij[s]*K3[s*Nr +idx], Db[idx] + dt*a_ij[s]*K2[s*Nr +idx], Da[idx] + dt*a_ij[s]*K4[s*Nr +idx],  lambda[idx] + dt*a_ij[s]*K7[s*Nr +idx],  ja[idx],dr_K,0.0, dr, dt, Nr, t);

      K1[idx] += dt *( b_i[s] * K1[(s+1)*Nr+idx]);
      K2[idx] += dt *( b_i[s] * K2[(s+1)*Nr+idx]);
      K3[idx] += dt *( b_i[s] * K3[(s+1)*Nr+idx]);
      K4[idx] += dt *( b_i[s] * K4[(s+1)*Nr+idx]);
      K5[idx] += dt *( b_i[s] * K5[(s+1)*Nr+idx]);
      K6[idx] += dt *( b_i[s] * K6[(s+1)*Nr+idx]);
      K7[idx] += dt *( b_i[s] * K7[(s+1)*Nr+idx]);
      K8[idx] += dt *( b_i[s] * K8[(s+1)*Nr+idx]);
      K9[idx] += dt *( b_i[s] * K9[(s+1)*Nr+idx]);

      K10[idx] += dt *( b_i[s] * K10[(s+1)*Nr+idx]);
      K11[idx] += dt *( b_i[s] * K11[(s+1)*Nr+idx]);
      K12[idx] += dt *( b_i[s] * K12[(s+1)*Nr+idx]);
      //if(s==0 && t==1){
      //printf("kpi=%f\n",A[idx]);};
    }
  }

    for(int idx=0;idx<Nr;idx++){ 
      B[idx]          +=K1[idx];
      Db[idx]         +=K2[idx];
      alpha[idx]      +=K3[idx];
      Da[idx]         +=K4[idx];
      Kb[idx]         +=K5[idx];
      K[idx]          +=K6[idx];
      lambda[idx]     +=K7[idx];
      U[idx]          +=K8[idx];
      A[idx]          +=K9[idx];

      PI[idx]         = PI[idx] + K10[idx];
      chi[idx]        = chi[idx] + K11[idx];
      temp_phi[idx]  += K12[idx];

      //reinicio los K CEROS

        K1[idx]=0.0;K2[idx]=0.0;K3[idx]=0.0;K4[idx]=0.0;
        K5[idx]=0.0;K6[idx]=0.0;K7[idx]=0.0;K8[idx]=0.0;
        K9[idx]=0.0;K10[idx]=0.0;K11[idx]=0.0;K12[idx]=0.0;
        //K[idx] =K[idx] + Kreiss_Oliger(K,idx,Nr,0.05);
        //Kb[idx] =Kb[idx] + Kreiss_Oliger(Kb,idx,Nr,0.05);
        //Db[idx] =Db[idx] + Kreiss_Oliger(Db,idx,Nr,0.05);
        //Da[idx] =Da[idx] + Kreiss_Oliger(Da,idx,Nr,0.05);
        //PI[idx] =PI[idx] + Kreiss_Oliger(PI,idx,Nr,0.05);
        //chi[idx] =chi[idx] + Kreiss_Oliger(chi,idx,Nr,0.05);

      //K[idx]=K[idx]/sqrt(1+K[idx]*K[idx]);
        //Kb[idx]=Kb[idx]/sqrt(1+K[idx]*K[idx]);
        //if (alpha[idx]<=0.000001){alpha[idx]=0.000001;}

        if (idx==2){
            //A[idx-1]=B[idx-1];
            //Kb[idx-1]=K[idx-1]/3.0;
            temp_phi[0]=(4.0*temp_phi[1]-temp_phi[2])/3.0;
            alpha[0]=(4.0*alpha[1]-alpha[2])/3.0;
            PI[0]=(4.0*PI[1]-PI[2])/3.0;
            A[0]=(4.0*A[1]-A[2])/3.0;
            B[0]=(4.0*B[1]-B[2])/3.0;
            K[0]=(4.0*K[1]-K[2])/3.0;
            Kb[0]=(4.0*Kb[1]-Kb[2])/3.0;

            //antisimetricas

            //simetricas

        }

       /*if(t%1==0){
        temp_Kb[(int)t*Nr+idx] = Kb[idx] + K5[idx];
       } */ 
        if (idx==Nr-1){
        Kb[idx]=absorbente(Kb,idx,dr,dt,Nr);
        K[idx]=absorbente(K,idx,dr,dt,Nr);
        PI[idx]=absorbente(PI,idx,dr,dt,Nr);
        chi[idx]=absorbente(chi,idx,dr,dt,Nr);

        }
      if(t%100==0){
        phi[(int)t*Nr/100 + idx] = temp_phi[idx] ;

      }

    /*if (idx==0 && t<62000 &&t>61000 ){
        printf("r : %d\n",idx);

    printf("PI : %.15f\n",K10[Nr+idx]);
        printf("PI : %.15f\n",K10[2*Nr+idx]);

    printf("PI : %.15f\n",K10[3*Nr+idx]);

    printf("PI : %.15f\n",K10[4*Nr+idx]);

    printf("chi : %.15f\n",K11[Nr+idx]);
        printf("chi : %.15f\n",K11[2*Nr+idx]);

    printf("chi : %.15f\n",K11[3*Nr+idx]);

    printf("chi : %.15f\n",K11[4*Nr+idx]);
    printf("A : %.15f\n",K9[Nr+idx]);
        printf("A : %.15f\n",K9[2*Nr+idx]);

    printf("A : %.15f\n",K9[3*Nr+idx]);

    printf("A : %.15f\n",K9[4*Nr+idx]);
    printf("B : %.15f\n",K1[Nr+idx]);
        printf("B : %.15f\n",K1[2*Nr+idx]);

    printf("B : %.15f\n",K1[3*Nr+idx]);

    printf("B : %.15f\n",K1[4*Nr+idx]);

    printf("alpha : %.15f\n",K3[Nr+idx]);
        printf("alpha : %.15f\n",K3[2*Nr+idx]);

    printf("alpha : %.15f\n",K3[3*Nr+idx]);

    printf("alpha : %.15f\n",K3[4*Nr+idx]);
    printf("Kb : %.15f\n",K5[Nr+idx]);
    printf("Kb : %.15f\n",K5[2*Nr+idx]);

    printf("Kb : %.15f\n",K5[3*Nr+idx]);

    printf("Kb : %.15f\n",K5[4*Nr+idx]);
    printf("K : %.15f\n",K6[Nr+idx]);
        printf("K : %.15f\n",K6[2*Nr+idx]);

    printf("K : %.15f\n",K6[3*Nr+idx]);

    printf("K : %.15f\n",K6[4*Nr+idx]);
    printf("lambda : %.15f\n",K7[Nr+idx]);
        printf("lambda : %.15f\n",K7[2*Nr+idx]);

    printf("lambda : %.15f\n",K7[3*Nr+idx]);

    printf("lambda : %.15f\n",K7[4*Nr+idx]);    
    printf("U : %.15f\n",K8[Nr+idx]);
        printf("U : %.15f\n",K8[2*Nr+idx]);

    printf("U : %.15f\n",K8[3*Nr+idx]);

    printf("U : %.15f\n",K8[4*Nr+idx]);
        printf("Da : %.15f\n",K4[Nr+idx]);
        printf("Da : %.15f\n",K4[2*Nr+idx]);

    printf("Da : %.15f\n",K4[3*Nr+idx]);

    printf("Da : %.15f\n",K4[4*Nr+idx]);
        printf("Db : %.15f\n",K2[Nr+idx]);
        printf("Db : %.15f\n",K2[2*Nr+idx]);

    printf("DB : %.15f\n",K2[3*Nr+idx]);

    printf("DB : %.15f\n",K2[4*Nr+idx]);
        printf("rho : %.15f\n",rho[idx]);
    printf("ja : %.15f\n",ja[idx]);
    printf("SA : %.15f\n",SA[idx]);
    printf("SB : %.15f\n",SB[idx]);
        printf("U : %.15f\n",U[idx]);
    printf("lambda : %.15f\n",lambda[idx]);}*/
    
    if (idx==0 &&t%1000==0){
    printf("t : %d\n",t);
    printf("r : %d\n",idx);


    printf("PI : %.15f\n",phi[(int)t*Nr/100 + idx]);

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
    printf("conservacion de rho = %.15f\n",conservacion(rho,dr,Nr));

    }

    }
      
}
printf("Runge-Kutta: check\n");


guardar_salida_phi(phi,Nr,minit);

guardar_salida_alpha(temp_Kb,Nr,15000);








  free(phi);free(chi);free(PI);free(K);free(Kb);free(U);free(A);free(B);free(alpha);free(lambda);

}