
//Nota segun el paper M. Alcubierre, J.A. Gonz´alez, Regularization of spherically symmetric evolution codes in
// numerical relativity, Computer Physics Communications 167, 2 (2005)
// lambda soluciona el problema de 1/r  cuanro r es 0 definiendolo como impar, y cuando 1/r*F donde F es cero en el origen, se puede ocupar l'hopital
//entonces los casos son l*u/r, será cero por l'hopital cuando l sea distinto de cero, sino tambien es cero pq l=0
//Da/r y Db/r 
//U/r tambien
// basicamente todo el primer f1 de Kb_dot
// en estos casos se puede aplicar l'hopital

//para Pi_dot tenemos que dejarlo como = dr(alphaBchi/sqrt(A)) + 2*alphaBchi/sqrt(A)/r, esto para no tener el 1/r^2 (probar si da mejores resultados) 
// en el caso de r=0 ocupamos =  dr(alphaBchi/sqrt(A)) + 2*alphaBchi/sqrt(A)/r, por l'hopital el segundo ternimo se deriva y alfinal queda 
// =  3*dr(alphaBchi/sqrt(A)), o solamente  3.0*alphaB/sqrt(A) dr(chi), pq A, B y alpha son pares. 


//para el caso cuantico l*u/r ya se solucionó arriba, el psi/r se puede usar l'hopital y el l(l+1)*alpha B/sqrt(A) *lambda*u/r, aqui en r=0 lambda simpre es cero, 
// por lo que se aplica l'hopital en l distinto de cero, cuando l = 0, siempre seria cero..

// para el caso de |u_tilda |^2/r^2, u_tilda, que al final sonesfericos de bessel, deberia ser en l distinto de cero, deberia ser cero por el * r^l,
// en l = 0 deberia ser no nulo, pero por ejemplo en la integral de phi_theta viene multiplicado por l, por lo que esa contribucion se anula.

//Nota: ultimo problema : el ja, tratar de que sea nulo en el espacio, es cero en r=0
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


// Constant
#define consta_pi 3.14159

// Derivate

#define order 11
__device__ double coefficient_centrada[11] ={-7.93650794e-04,  9.92063492e-03, -5.95238095e-02,  2.38095238e-01,
    -8.33333333e-01,  0.00000000e+00,  8.33333333e-01, -2.38095238e-01,
     5.95238095e-02, -9.92063492e-03,  7.93650794e-04,
  };
#define half_order_right (int)(order+1)/2
#define half_order_left (int)(order-1)/2
// Parameter
#define Nr 500
#define dr 0.025
#define Nt 100
#define dt dr/4.0

//Parameter classic
#define amplitude 0.0
#define width 1.5
// Parameter quantum
#define Nk 5
#define Nl 5
#define dk consta_pi/15.0

#define  dim_q Nk*Nl*Nr

#define ct 0.0
//Runge-Kutta tenth order implicit
#define step 5

__constant__  double a_ij[5][5];
__constant__  double b_i[5];
__constant__  double c_i[5];


struct Relativity_G{
    double *A, *B, *alpha;
    double *Da, *Db;
    double *K, *Kb;
    double *U, *lambda;
};
struct field_Classic{
    double *phi, *psi, *pi;
};
struct field_imag{
    double *u, *psi, *pi;
};
struct field_real{
    double *u, *psi, *pi;
};
struct field_Quantum{
    struct field_real real;
    struct field_imag imag;
};


//guardado de diferentes salidas.
void guardar_salida_rho(double *data,int T) {
  FILE *fp = fopen("rho.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_SA(double *data, int T) {
    FILE *fp = fopen("SA.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_ja(double *data, int T) {
    FILE *fp = fopen("ja.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_SB(double *data,int T) {
    FILE *fp = fopen("SB.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }

__device__ double psi_dot(Relativity_G metrics, double *field_pi, int id_nodo, int idx){
    double temp,sym;
    double f,fm1,fp1;
    temp = 0.0;
    if (idx < half_order_left){
        sym=1.0;
        for (int m=0; m < half_order_right + idx ; m++){
            f= metrics.alpha[m]/(sqrt(metrics.A[m])*metrics.B[m]) * field_pi[id_nodo + m];
            temp += coefficient_centrada[m + half_order_left -idx ]*(f);
        }
        for (int m=half_order_left - idx; m > 0 ; m--){
            f= metrics.alpha[m]/(sqrt(metrics.A[m])*metrics.B[m]) * field_pi[id_nodo + m];
            temp += sym*coefficient_centrada[ half_order_left - idx - m  ]*(f);
        }
    }
    else if (idx > Nr-half_order_left-1 && idx < Nr-1){
        fm1=metrics.alpha[idx-1]/(sqrt(metrics.A[idx-1])*metrics.B[idx-1]) * field_pi[id_nodo + idx-1];
        fp1=metrics.alpha[idx+1]/(sqrt(metrics.A[idx+1])*metrics.B[idx+1]) * field_pi[id_nodo + idx+1];

        temp = (0.5*fp1 - 0.5*fm1); 
    }
    else if (idx == Nr-1){
        fm1=metrics.alpha[idx-1]/(sqrt(metrics.A[idx-1])*metrics.B[idx-1]) * field_pi[id_nodo + idx-1];
        f=metrics.alpha[idx]/(sqrt(metrics.A[idx])*metrics.B[idx]) * field_pi[id_nodo + idx];

        temp = (f - fm1);
    }
    else{
        for (int m=0;m<order;m++){
            f=metrics.alpha[idx-half_order_left+m]/(sqrt(metrics.A[idx-half_order_left+m])*metrics.B[idx-half_order_left+m]) * field_pi[id_nodo + idx-half_order_left+m];
            temp += coefficient_centrada[m]*f;
        }
    }
    temp=temp/dr;
    return temp;
}
__device__ double f_pi_dot(field_Classic field_C, Relativity_G metrics, int idx){
  double temp,sym;
  double f,fm1,fp1;
  temp=0.0;

  if (idx < half_order_left){
      sym=1.0;
      for (int m=0; m < half_order_right + idx ; m++){
            f=metrics.alpha[m]*metrics.B[m] * field_C.psi[m]/sqrt(metrics.A[m]);
            temp += coefficient_centrada[m + half_order_left -idx ]*(f);
      }
      for (int m=half_order_left - idx; m > 0 ; m--){

            f=metrics.alpha[m]*metrics.B[m] * field_C.psi[m]/sqrt(metrics.A[m]);
            temp += sym*coefficient_centrada[ half_order_left - idx - m  ]*(f);
      }
    
  }
  else if (idx > Nr-half_order_left-1 && idx < Nr-1){

    
      fm1=metrics.alpha[idx-1]*metrics.B[idx-1] * field_C.psi[idx-1]/sqrt(metrics.A[idx-1]);
      fp1=metrics.alpha[idx+1]*metrics.B[idx+1] * field_C.psi[idx+1]/sqrt(metrics.A[idx+1]);

      temp = (0.5*fp1 - 0.5*fm1); 
  }
  else if (idx == Nr-1){
    fm1=metrics.alpha[idx-1]*metrics.B[idx-1] * field_C.psi[idx-1]/sqrt(metrics.A[idx-1]);
    f=metrics.alpha[idx]*metrics.B[idx] * field_C.psi[idx]/sqrt(metrics.A[idx]);

      temp = (f - fm1);
  }
  else{
      for (int m=0;m<order;m++){
          f=metrics.alpha[idx-half_order_left+m]*metrics.B[idx-half_order_left+m] * field_C.psi[idx-half_order_left+m]/sqrt(metrics.A[idx-half_order_left+m]);
          temp += coefficient_centrada[m]*f;
      }
  }
  temp=temp/dr;
  return temp;
}
__device__ double derivate_metric(Relativity_G metrics, int idx){
    double temp,sym;
    double f,fm1,fp1;
    temp=0.0;


    if (idx < half_order_left){
        sym=1.0;
        for (int m=0; m < half_order_right + idx ; m++){
            f=metrics.alpha[m]*metrics.B[m]/sqrt(metrics.A[m]);
            temp += coefficient_centrada[m + half_order_left -idx ]*(f);
        }
        for (int m=half_order_left - idx; m > 0 ; m--){
            f=metrics.alpha[m]*metrics.B[m]/sqrt(metrics.A[m]);
            temp += sym*coefficient_centrada[ half_order_left - idx - m  ]*(f);
        }
      
    }
    else if (idx > Nr-half_order_left-1 && idx < Nr-1){
        fm1=metrics.alpha[idx-1]*metrics.B[idx-1]/sqrt(metrics.A[idx-1]);
        fp1=metrics.alpha[idx+1]*metrics.B[idx+1]/sqrt(metrics.A[idx+1]);

        temp = (0.5*fp1 - 0.5*fm1); 
    }
    else if (idx == Nr-1){
        fm1=metrics.alpha[idx-1]*metrics.B[idx-1]/sqrt(metrics.A[idx-1]);
        f=metrics.alpha[idx]*metrics.B[idx]/sqrt(metrics.A[idx]);

        temp = (f - fm1);
    }
    else{
        for (int m=0;m<order;m++){
            f=metrics.alpha[idx-half_order_left+m]*metrics.B[idx-half_order_left+m]/sqrt(metrics.A[idx-half_order_left+m]);
            temp += coefficient_centrada[m]*f;
        }
    }
    temp=temp/dr;
    return temp;

}
__device__ double derivate( double *f, int id_nodo, int idx, int symmetric ){
    double temp,sym;
  
      temp=0.0;
  
      if (idx < half_order_left){
        if (symmetric==0){
          sym=1.0;
        }
        else if (symmetric == 1){
          sym=-1.0;
        }
  
        for (int m=0; m < half_order_right + idx ; m++){
  
            temp += coefficient_centrada[m + half_order_left -idx ]*(f[id_nodo + m]);
  
  
          }
        for (int m=half_order_left - idx; m > 0 ; m--){
  
            temp += sym*coefficient_centrada[ half_order_left - idx - m  ]*(f[id_nodo + m]);
  
          }
        
      }
      else if (idx > Nr-half_order_left-1 && idx < Nr-1){
        temp = (0.5*f[id_nodo + idx+1] - 0.5*f[id_nodo + idx-1]); 
      }
      else if (idx == Nr-1){
        temp = (f[id_nodo + idx] - f[id_nodo + idx-1]);
      }
      else{
        for (int m=0;m<order;m++){
          temp += coefficient_centrada[m]*f[id_nodo + idx-half_order_left+m];
        }
      }
      temp=temp/dr;
      return temp;
}
__global__ void cost_cosm_value(double *rho,double *cosmological_constant){
  //cosmological_constant[0] = pow(2.0,4)/pow(2.0*consta_pi,2)*log(pow(3.0,9)/pow(2.0,16))/8.0;
  cosmological_constant[0] = -rho[0];

    printf("Constante cosmologica : %.15f \n", cosmological_constant[0]);
  
}

__global__ void Tensor_tt(double * T_tt,double * T_rr,double * T_tr,double * T_00, double *rho,double *SA,double *ja,double *SB, int t){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx<Nr){
    T_tt[ t*Nr + idx] = rho[idx];
    T_rr[ t*Nr + idx] = SA[idx];
    T_tr[ t*Nr + idx] = ja[idx];
    T_00[ t*Nr + idx] = SB[idx];
      if(idx==0){
        printf("rho[0]=%.15f\n",rho[0]);
      }
    
  }
}

__global__ void Tensor_energy_momentum(double *rho, double *ja, double *SA, double *SB, Relativity_G metrics, field_Classic field_C, field_Quantum field_Q, int g, int siono){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int id;
    int cinetica;
    double radio;
    //temp_fluct (5xNr)
    double cte= 1.0/(4.0*consta_pi);
    double u_snake_x;
    double u_snake_y;
    
    double d_u_temp_x;
    double d_u_temp_y;
    double d_u_radial_x;
    double d_u_radial_y;
    
    double inte_pi, inte_psi, inte_pi_psi, inte_theta;
    double xpi2x, xpsi2x, xpi_psix, xtheta2x;

    //radio
    if (idx < Nr){

        xpi2x=0.0;
        xpsi2x=0.0;
        xpi_psix=0.0;
        xtheta2x=0.0;    

        if(idx==0){
        radio=0.0;
        }
        else{
        radio=idx*dr;
        }
        for(int l=0 ; l<Nl ; l++){

            inte_pi=0.0;
            inte_psi=0.0;
            inte_pi_psi=0.0;
            inte_theta=0.0;

            for (int k=0 ; k < Nk ; k++){
                // indice
                id = g*dim_q + l*Nk*Nr + k*Nr +  idx; 
                
                //derivada temporal
                d_u_temp_x = field_Q.real.pi[ id ] * pow(radio , l);
                d_u_temp_y = field_Q.imag.pi[ id ] * pow(radio , l);
                
                //derivada radial compleja
                
                if(l==0){
                d_u_radial_x = (field_Q.real.psi[id ]);
                d_u_radial_y = (field_Q.imag.psi[id ]); 
                }
                else if(l==1){
                d_u_radial_x =  (field_Q.real.psi[id ]* pow(radio , l) + field_Q.real.u[id ]);
                d_u_radial_y =  (field_Q.imag.psi[id ]* pow(radio , l) + field_Q.imag.u[id ]);
                }
                else{
                d_u_radial_x = (field_Q.real.psi[id ]* pow(radio , l) + field_Q.real.u[id ]* pow(radio , l-1) *l);
                d_u_radial_y = (field_Q.imag.psi[id ]* pow(radio , l) + field_Q.imag.u[id ]* pow(radio , l-1) *l);
                }
                
                /*
                if(idx==0){
                u_snake_x = field_Q.real.u[ id +1] * pow(radio +dr, l) ;
                u_snake_y = field_Q.iamg.u[ id +1] * pow(radio +dr, l) ;
                }*/
                if(idx==0){
                  u_snake_x = field_Q.real.u[ id +1 ] * pow(radio +dr, l) ;
                  u_snake_y = field_Q.imag.u[ id +1 ] * pow(radio +dr, l) ;
                }
                else{
                  u_snake_x = field_Q.real.u[ id ] * pow(radio , l) ;
                  u_snake_y = field_Q.imag.u[ id ] * pow(radio , l) ;
                }
                
                
                inte_pi += ( pow(d_u_temp_x,2) + pow(d_u_temp_y,2));
                inte_psi += ( pow(d_u_radial_x,2) + pow(d_u_radial_y,2));
                inte_pi_psi += d_u_temp_x*d_u_radial_x + d_u_temp_y*d_u_radial_y;
                inte_theta += ( pow(u_snake_x,2) + pow(u_snake_y,2) );
                //temp_fluct[ 4*Nr + idx] += ( pow(u_snake_x,2) + pow(u_snake_y,2) );

            }
            //fluctuacion tiempos
            xpi2x += inte_pi * (2*l + 1) * cte * dk;
            //fluctuacion radial
            xpsi2x += inte_psi * (2*l + 1) * cte * dk;
            //fluctuacion t r
            xpi_psix += inte_pi_psi * (2*l + 1) * cte* dk;
            //fluctuacion angulares
            xtheta2x += inte_theta * 0.5 * l*(l+1)*(2*l + 1) * cte* dk;
            //fluctuacion phi
            //temp_fluct[ 4*Nr + idx] = temp_fluct[ 4*Nr + idx] * (2*l + 1) * cte* dk;

        }
    
        //printf("r=%d, xtheta2x=%.20f\n", idx,xtheta2x);
        if(g==0){
            cinetica=1;
            xpi2x   = field_C.pi[idx]*field_C.pi[idx] + xpi2x;
            xpsi2x  = field_C.psi[idx]*field_C.psi[idx] + xpsi2x;
            xpi_psix = field_C.pi[idx]*field_C.psi[idx] + xpi_psix;
        }
        else if(g%2==1){
            cinetica=-1;
        }
        else{
            cinetica=1;
        }
        if(idx==0){
          radio=dr;
          }
          else{
          radio=idx*dr;
          }
        rho[idx] +=     cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) + 1.0/(metrics.B[idx]*radio*radio)*xtheta2x );
  
        ja[idx] += -    cinetica*xpi_psix/(sqrt(metrics.A[idx])*metrics.B[idx]);
    
        SA[idx] +=      cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) - 1.0/(metrics.B[idx]*radio*radio)*xtheta2x );
    
        SB[idx] +=      cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) - xpsi2x) );
        if(idx==0 && siono==0 && g==5){
          printf("Stress_Energy en  idx : %d of g : %d:\n",idx,g); 
    
            printf(" <x|PI|x>^2 = %1.5f\n",xpi2x);
            printf(" <x|psi|x>^2 = %1.5f\n",xpsi2x);
            printf(" <x|PI psi|x>^2 = %1.5f\n",xpi_psix);
            printf(" <x|theta|x>^2 = %1.5f\n",xtheta2x);
            //printf(" <x|phi|x>^2 = %1.5f\n",temp_array[ 4*Nl*Nr + idx ]);
            printf("alpha : %.15f\n",metrics.alpha[idx]);
        
            printf("A : %.15f\n",metrics.A[idx]);
            printf("B : %.15f\n",metrics.B[idx]);
        
            printf("rho : %.15f\n",rho[idx]);
            printf("ja : %.15f\n",ja[idx]);
            printf("SA : %.15f\n",SA[idx]);
            printf("SB : %.15f\n",SB[idx]);
        
        
          }
    }
}
  
__global__ void T_zeros(double *rho, double *ja, double *SA, double *SB){
    int idx=threadIdx.x + blockDim.x*blockIdx.x;
    if (idx<Nr){
        rho[idx]=0.0;
        SA[idx]=0.0;
        SB[idx]=0.0;
        ja[idx]=0.0;
    }
}
__global__ void evo_metrics(Relativity_G metrics,Relativity_G metrics_RK, double *rho, double *ja, double *SA, double*SB, double *Cosm){

  int id = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;
  double f1,f2,f3;
  if(id < Nr){

    if(id==0){radio=dr;}
    else{radio=id*dr;}

      //                                                ********metrica********

      metrics_RK.A[id] = -2.0*metrics.alpha[id]*metrics.A[id]*(metrics.K[id] - 2.0*metrics.Kb[id]);
      metrics_RK.B[id] = -2.0*metrics.alpha[id]*metrics.B[id]*metrics.Kb[id];
      metrics_RK.alpha[id] = -2.0*metrics.alpha[id]*metrics.K[id];
      //                                                ********metrica_prima********
      metrics_RK.Da[id] = -2.0*derivate(metrics.K, 0, id, 0);
      metrics_RK.Db[id] = -2.0*(derivate(metrics.alpha, 0, id, 0)*metrics.Kb[id] + metrics.alpha[id]*derivate(metrics.Kb,0, id,0) ); 

      //                                                ********Curvatura extrinseca********
      //Determinamos un valor para una expresion que se repite bastante
      double temp;
      temp = (metrics.U[id] + 4.0*metrics.lambda[id]*metrics.B[id]/metrics.A[id] );

      //K
      f1 =(pow(metrics.K[id],2) - 4.0*metrics.K[id]*metrics.Kb[id] + 6.0*pow(metrics.Kb[id],2));
      f3= (rho[id] + SA[id] + 2.0*SB[id] - 2.0*Cosm[0]);
      if(id==0){
          f2=(3.0*derivate(metrics.Da,0,id,1) + pow(metrics.Da[id],2) - 0.5*metrics.Da[id]* temp );
      }
      else{
          f2=(derivate(metrics.Da,0,id,1) + pow(metrics.Da[id],2) + 2.0*metrics.Da[id]/radio- 0.5*metrics.Da[id]* temp );
      }

      metrics_RK.K[id] =  metrics.alpha[id]* f1 - metrics.alpha[id]/metrics.A[id] * f2 + 0.5*metrics.alpha[id] * f3;
      if(id==0){
        //printf("K : %.15f | f1 : %.15f | f2 : %.15f | f3 : %.15f\n", metrics_RK.K[id],f1,f2,f3);
      }
      //Kb
      //Solamente derivaremos las funciones que son impares, ya que las demas actuan como constantes en r = 0,
      if(id==0){
          f1 = (0.5*(derivate( metrics.U,0,id,1) + 4*metrics.B[id]/metrics.A[id]*derivate(metrics.lambda,0,id,1)) - derivate(metrics.Db,0,id,1) - derivate(metrics.lambda,0,id,1) - derivate(metrics.Da,0,id,1) );
      }
      else{
          f1 = 1.0/radio * (0.5*temp - metrics.Db[id] - metrics.lambda[id] - metrics.Da[id] );
      }
      f2=(-0.5*metrics.Da[id]*metrics.Db[id] - 0.5*derivate(metrics.Db,0,id,1) + 0.25*metrics.Db[id]* temp + metrics.A[id]*metrics.K[id]*metrics.Kb[id]);
      f3=(SA[id] - rho[id] - 2.0*Cosm[0]);

      metrics_RK.Kb[id] = metrics.alpha[id]/metrics.A[id] * (  f1 + f2  )+ 0.5*metrics.alpha[id] * f3;
      if(id==0){
        //printf("Kb : %.15f | f1 : %.15f | f2 : %.15f | f3 : %.15f\n", metrics_RK.Kb[id],f1,f2,f3);
        //printf("dU : %.15f | dlambda : %.15f | dDb : %.15f | dDa : %.15f\n", derivate( metrics.U,0,id,1),derivate(metrics.lambda,0,id,1),derivate(metrics.Db,0,id,1),derivate(metrics.Da,0,id,1));

      }
      //                                                ********Regularacion********

      metrics_RK.U[id] = -2.0*metrics.alpha[id] * (derivate( metrics.K,0,id,0) + metrics.Da[id]* ( metrics.K[id] - 4.0*metrics.Kb[id] ) 
                                                      - 2.0* ( metrics.K[id] - 3.0*metrics.Kb[id] ) * ( metrics.Db[id] - 2.0*metrics.lambda[id]*metrics.B[id]/metrics.A[id] ))
                          -4.0*metrics.alpha[id] * ja[id];

      temp = 2.0*metrics.alpha[id]*metrics.A[id]/metrics.B[id];
      metrics_RK.lambda[id] = temp * ( derivate(metrics.Kb,0, id, 0) - 0.5*metrics.Db[id] * ( metrics.K[id] - 3.0*metrics.Kb[id] ) + 0.5*ja[id] );
      /*if(id==  0 ){  
        printf("A : %.15f\n",metrics.A[id]);
        printf("B : %.15f\n",metrics.B[id]);
        printf("alpha : %.15f\n",metrics.alpha[id]);
        printf("Da : %.15f\n", metrics.Da[id]);
        printf("K past: %.15f\n",metrics.K[id-1]);
        printf("K %.15f\n",metrics.K[id]);
        printf("K last: %.15f\n",metrics.K[id+1]);

        printf("Kb past : %.15f\n",metrics.Kb[id-1]);
        printf("Kb : %.15f\n",metrics.Kb[id]);
        printf("Kb last : %.15f\n",metrics.Kb[id+1]);

        printf("K-3kb : %.15f\n",metrics.K[id] - 3.0*metrics.Kb[id]);
        printf("K-3kb +1: %.15f\n",metrics.K[id+1] - 3.0*metrics.Kb[id+1]);
        printf("K-3kb +2: %.15f\n",metrics.K[id+2] - 3.0*metrics.Kb[id+2]);

        printf("Db : %.15f\n", metrics.Db[id]);
        printf("Db ´1: %.15f\n", metrics.Db[id+1]);
        printf("Db ´+2: %.15f\n", metrics.Db[id+2]);

        printf("dKb : %.15f\n",derivate(metrics.Kb,0, id, 0));
        printf("dKb +1: %.15f\n",derivate(metrics.Kb,0, id+1, 0));
        printf("dKb +2: %.15f\n",derivate(metrics.Kb,0, id+2, 0));

        printf("lambda : %.15f\n",metrics.lambda[id]);
        printf("U : %.15f\n",metrics_RK.U[id]);
        printf("lambda +1 : %.15f\n",metrics.lambda[id+1]);
        printf("U +1 : %.15f\n",metrics_RK.U[id+1]);
        printf("lambda +2 : %.15f\n",metrics.lambda[id+2]);
        printf("U +2: %.15f\n",metrics_RK.U[id+2]);
        printf("ja: %.15f\n",ja[id]);
        printf("ja +1: %.15f\n",ja[id+1]);
        printf("ja +2: %.15f\n",ja[id+2]);

      }*/
    }
}
__global__ void evo_fields_classics(field_Classic field_C, field_Classic field_C_RK, Relativity_G metrics){
    int id = threadIdx.x + blockDim.x*blockIdx.x;;

  double radio,temp;
  if(id < Nr){

    if(id==0){radio=dr;}
    else{radio=id*dr;}
    
    temp = metrics.alpha[id]/(sqrt(metrics.A[id])*metrics.B[id]);
    
    field_C_RK.phi[id] = temp*field_C.pi[id];
    field_C_RK.psi[id] = psi_dot(metrics,field_C.pi,0,id);
      if(id==0){
          field_C_RK.pi[id] =  3.0*f_pi_dot(field_C,metrics,id) ;  
      }
      else{
          field_C_RK.pi[id] =  f_pi_dot(field_C,metrics,id) + 2.0*metrics.alpha[id]*metrics.B[id] * field_C.psi[id]/sqrt(metrics.A[id])/radio;  
      }
  }
}
__global__ void evo_fields_quantums(  field_Quantum field_Q, field_Quantum field_Q_RK, Relativity_G metrics,  int g, int s){
    int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
    int id_nodo = threadIdx.x + blockDim.x*blockIdx.x;;
    int r = id_nodo%Nr;
    int k = (int)id_nodo/Nr%Nk;
    int l = (int)id_nodo/Nr/Nl;
    int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
    double temp[2];
    double A_B_alpha_prime;
    double mass;
    if(g==0){
      mass=0.0;
    }
    else if(g==1 || g==3){
      mass=1.0;
    }
    else if(g==2 || g==4){
      mass=1*sqrt(3.0);
    }
    else if(g==5){
      mass=1*sqrt(4.0);
    }
    if(id < 6*dim_q){

      field_Q_RK.real.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.real.pi[id];
      field_Q_RK.imag.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.imag.pi[id];

      temp[0] = psi_dot(metrics, field_Q.real.pi, id_nodo_ghost ,r);
      temp[1] = psi_dot(metrics, field_Q.imag.pi, id_nodo_ghost ,r);

      field_Q_RK.real.psi[id] = temp[0];
      field_Q_RK.imag.psi[id] = temp[1];

      double f1, f2, f3, f4, radio;
      if (r==0){
          radio = 0.0;
      }
      else{
          radio = dr*r;
      }
      A_B_alpha_prime = derivate_metric(metrics, r);
      //pi real
      if(r==0){
          f1 = (l+1) * ( field_Q.real.psi[id]); // cero debido a A_B_alpha_prime
          f2 = (2*l+3) * derivate(field_Q.real.psi,id_nodo_ghost,r,1);
          f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.real.u[id];
      }
      else{
          f1 = l/radio * (field_Q.real.u[id]) + (field_Q.real.psi[id]);
          f2 = (2*l+2)/radio * field_Q.real.psi[id] + derivate(field_Q.real.psi,id_nodo_ghost,r,1);
          f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.real.u[id];
      }
      f4 = mass * mass* field_Q.real.u[id];//* field_Q.real.u[id];
      if (id==0*dim_q+3){
        //printf("id : %d | r: %d | k: %d | l: %d | id_nodo : %d | id_nodo_g : %d\n",id,r,k,l,id_nodo,id_nodo_ghost);
        //printf("s: %d | g: %d | u: %.15f | psi: %.15f | pi: %.15f | f1:%.15f | f2 : %.15f | f3: %.15f\n",s,g,field_Q.real.u[id],field_Q.real.psi[id] , (field_Q.real.pi[id]),f1,f2,f3);
        //printf("s: %d | g: %d | arg: %.15f | u: %.15f | lambda: %.15f |\n",s,g,l*(l+1)/radio,field_Q.real.u[id] , (metrics.lambda[r]));

      }
      //Nota : averiguar si f3 tiene un menos o no.
      field_Q_RK.real.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 - f3) - metrics.alpha[r]*metrics.B[r]*(sqrt(metrics.A[r]))*f4;
      if (id==0*dim_q+3){
        //printf("s: %d | g: %d |  pi_dot: %.15f | psi_ddr : %.15f  |\n",s,g,field_Q_RK.real.pi[id],derivate(field_Q.real.psi,id_nodo_ghost,r,1));
      }
      //pi imaginario
      if(r==0){
          f1 = (l+1) * ( field_Q.imag.psi[id]); // cero
          f2 = (2*l+3) *  derivate(field_Q.imag.psi,id_nodo_ghost,r,1);
          f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.imag.u[id];
      }
      else{

          f1 = l/radio * (field_Q.imag.u[id]) + (field_Q.imag.psi[id]);
          f2 = (2*l+2)/radio * field_Q.imag.psi[id] + derivate(field_Q.imag.psi,id_nodo_ghost,r,1);
          f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.imag.u[id];
      }
      f4 = mass * mass* field_Q.imag.u[id];//* field_Q.imag.u[id];

      field_Q_RK.imag.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 - f3) - metrics.alpha[r]*metrics.B[r]*(sqrt(metrics.A[r]))*f4;
      if (id==0*dim_q+3){
        //printf("id : %d | r: %d | k: %d | l: %d | id_nodo : %d | id_nodo_g : %d\n",id,r,k,l,id_nodo,id_nodo_ghost);
        //printf("s: %d | g: %d | u: %.15f | psi: %.15f | pi: %.15f | f1:%.15f | f2 : %.15f | f3: %.15f\n",s,g,field_Q.imag.u[id],field_Q.imag.psi[id] , (field_Q.imag.pi[id]),f1,f2,f3);
        //printf("s: %d | g: %d | arg: %.15f | u: %.15f | lambda: %.15f |\n",s,g,l*(l+1)/radio,field_Q.imag.u[id] , (metrics.lambda[r]));

        //printf("s: %d | g: %d |  pi_dot: %.15f | psi_ddr : %.15f  |\n",s,g,field_Q_RK.imag.pi[id],derivate(field_Q.imag.psi,id_nodo_ghost,r,1));
      }
  }
}
__global__ void y_tilde_metrics( Relativity_G metrics , Relativity_G RG_K1, Relativity_G RG_K2,  Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5, Relativity_G y_tilde_M, int s){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < Nr){
      y_tilde_M.A[id]      = metrics.A[id]        + dt*( a_ij[s][0]*RG_K1.A[id]      + a_ij[s][1]*RG_K2.A[id]      + a_ij[s][2]*RG_K3.A[id]      + a_ij[s][3]*RG_K4.A[id]      + a_ij[s][4]*RG_K5.A[id] );
      y_tilde_M.B[id]      = metrics.B[id]        + dt*( a_ij[s][0]*RG_K1.B[id]      + a_ij[s][1]*RG_K2.B[id]      + a_ij[s][2]*RG_K3.B[id]      + a_ij[s][3]*RG_K4.B[id]      + a_ij[s][4]*RG_K5.B[id] );
      y_tilde_M.alpha[id]  = metrics.alpha[id]    + dt*( a_ij[s][0]*RG_K1.alpha[id]  + a_ij[s][1]*RG_K2.alpha[id]  + a_ij[s][2]*RG_K3.alpha[id]  + a_ij[s][3]*RG_K4.alpha[id]  + a_ij[s][4]*RG_K5.alpha[id] );
      if(y_tilde_M.alpha[id] > 1.0){
        y_tilde_M.alpha[id] = 1.0;
      }
      y_tilde_M.Da[id]     = metrics.Da[id]       + dt*( a_ij[s][0]*RG_K1.Da[id]     + a_ij[s][1]*RG_K2.Da[id]     + a_ij[s][2]*RG_K3.Da[id]     + a_ij[s][3]*RG_K4.Da[id]     + a_ij[s][4]*RG_K5.Da[id] );
      y_tilde_M.Db[id]     = metrics.Db[id]       + dt*( a_ij[s][0]*RG_K1.Db[id]     + a_ij[s][1]*RG_K2.Db[id]     + a_ij[s][2]*RG_K3.Db[id]     + a_ij[s][3]*RG_K4.Db[id]     + a_ij[s][4]*RG_K5.Db[id] );
      y_tilde_M.Kb[id]     = metrics.Kb[id]       + dt*( a_ij[s][0]*RG_K1.Kb[id]     + a_ij[s][1]*RG_K2.Kb[id]     + a_ij[s][2]*RG_K3.Kb[id]     + a_ij[s][3]*RG_K4.Kb[id]     + a_ij[s][4]*RG_K5.Kb[id] );
      y_tilde_M.K[id]      = metrics.K[id]        + dt*( a_ij[s][0]*RG_K1.K[id]      + a_ij[s][1]*RG_K2.K[id]      + a_ij[s][2]*RG_K3.K[id]      + a_ij[s][3]*RG_K4.K[id]      + a_ij[s][4]*RG_K5.K[id] );
      y_tilde_M.lambda[id] = metrics.lambda[id]   + dt*( a_ij[s][0]*RG_K1.lambda[id] + a_ij[s][1]*RG_K2.lambda[id] + a_ij[s][2]*RG_K3.lambda[id] + a_ij[s][3]*RG_K4.lambda[id] + a_ij[s][4]*RG_K5.lambda[id] );
      y_tilde_M.U[id]      = metrics.U[id]        + dt*( a_ij[s][0]*RG_K1.U[id]      + a_ij[s][1]*RG_K2.U[id]      + a_ij[s][2]*RG_K3.U[id]      + a_ij[s][3]*RG_K4.U[id]      + a_ij[s][4]*RG_K5.U[id] );
      
      //if(id==20){
      //  printf("A_tilda=%.15f\n",y_tilde_M.A[id]);
      //  }
      if(id==0){
        y_tilde_M.Da[id]=0.0;
        y_tilde_M.Db[id]=0.0;
        y_tilde_M.lambda[id]=0.0;
        y_tilde_M.U[id]=0.0;
      }
    }
}
    
__global__ void y_tilde_field_classic(field_Classic field_C, field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5, field_Classic y_tilde_C, int s){

    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < Nr){
      y_tilde_C.phi[id] = field_C.phi[id] + dt*( a_ij[s][0]* FC_K1.phi[id]  + a_ij[s][1]* FC_K2.phi[id]   + a_ij[s][2]* FC_K3.phi[id]   + a_ij[s][3]* FC_K4.phi[id]   + a_ij[s][4]* FC_K5.phi[id]);
      y_tilde_C.pi[id]  = field_C.pi[id]  + dt*( a_ij[s][0]* FC_K1.pi[id]   + a_ij[s][1]* FC_K2.pi[id]    + a_ij[s][2]* FC_K3.pi[id]    + a_ij[s][3]* FC_K4.pi[id]    + a_ij[s][4]* FC_K5.pi[id]);
      y_tilde_C.psi[id] = field_C.psi[id] + dt*( a_ij[s][0]* FC_K1.psi[id]  + a_ij[s][1]* FC_K2.psi[id]   + a_ij[s][2]* FC_K3.psi[id]   + a_ij[s][3]* FC_K4.psi[id]   + a_ij[s][4]* FC_K5.psi[id]);
      if(id==0){
        y_tilde_C.psi[id]=0.0;
      }
    }
  
}
__global__ void y_tilde_field_quantum(field_Quantum field_Q, field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5, field_Quantum y_tilde_Q, int s, int g){

    int id =  g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
    if(id < 6*dim_q){
      y_tilde_Q.real.u[id] = field_Q.real.u[id]     + dt*( a_ij[s][0]* FQ_K1.real.u[id]     + a_ij[s][1]* FQ_K2.real.u[id]    + a_ij[s][2]* FQ_K3.real.u[id]    + a_ij[s][3]* FQ_K4.real.u[id]    + a_ij[s][4]* FQ_K5.real.u[id]);
      y_tilde_Q.real.pi[id]  = field_Q.real.pi[id]  + dt*( a_ij[s][0]* FQ_K1.real.pi[id]    + a_ij[s][1]* FQ_K2.real.pi[id]   + a_ij[s][2]* FQ_K3.real.pi[id]   + a_ij[s][3]* FQ_K4.real.pi[id]   + a_ij[s][4]* FQ_K5.real.pi[id]);
      y_tilde_Q.real.psi[id] = field_Q.real.psi[id] + dt*( a_ij[s][0]* FQ_K1.real.psi[id]   + a_ij[s][1]* FQ_K2.real.psi[id]  + a_ij[s][2]* FQ_K3.real.psi[id]  + a_ij[s][3]* FQ_K4.real.psi[id]  + a_ij[s][4]* FQ_K5.real.psi[id]);

      y_tilde_Q.imag.u[id] = field_Q.imag.u[id]     + dt*( a_ij[s][0]* FQ_K1.imag.u[id]     + a_ij[s][1]* FQ_K2.imag.u[id]    + a_ij[s][2]* FQ_K3.imag.u[id]    + a_ij[s][3]* FQ_K4.imag.u[id]    + a_ij[s][4]* FQ_K5.imag.u[id]);
      y_tilde_Q.imag.pi[id]  = field_Q.imag.pi[id]  + dt*( a_ij[s][0]* FQ_K1.imag.pi[id]    + a_ij[s][1]* FQ_K2.imag.pi[id]   + a_ij[s][2]* FQ_K3.imag.pi[id]   + a_ij[s][3]* FQ_K4.imag.pi[id]   + a_ij[s][4]* FQ_K5.imag.pi[id]);
      y_tilde_Q.imag.psi[id] = field_Q.imag.psi[id] + dt*( a_ij[s][0]* FQ_K1.imag.psi[id]   + a_ij[s][1]* FQ_K2.imag.psi[id]  + a_ij[s][2]* FQ_K3.imag.psi[id]  + a_ij[s][3]* FQ_K4.imag.psi[id]  + a_ij[s][4]* FQ_K5.imag.psi[id]);
    }
      if(id==dim_q+3){
        //printf("pi_tilda=%.15f | K1: %.15f | | K2: %.15f | | K3: %.15f | | K4: %.15f | | K5: %.15f |\n",y_tilde_Q.real.pi[id],FQ_K1.real.pi[id],FQ_K2.real.pi[id],FQ_K3.real.pi[id],FQ_K4.real.pi[id],FQ_K5.real.pi[id]);
        //printf("a_i,1: %.15f | a_i,2: %.15f | a_i,3: %.15f | a_i,4: %.15f | a_i,5: %.15f |\n",a_ij[s][0],a_ij[s][1],a_ij[s][2],a_ij[s][3],a_ij[s][4])  ;
      }
}


//Metodo runge kutta de decimo orden implicito, ocupando el metodo de gauss-legendre. 
//Orden del metodo es p = S*2, siendo S el numero de pasos ("step")
void RK_implicit_tenth(Relativity_G metrics, Relativity_G RG_K1,Relativity_G RG_K2, Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5, Relativity_G y_tilde_M,
    field_Classic field_C, field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5, field_Classic y_tilde_C,
    field_Quantum field_Q, field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5, field_Quantum y_tilde_Q,
    double *rho, double *ja, double *SA, double *SB, double *Cosm ){
      int thread = 64;
      ////Simulacion////
      dim3  bloque(thread);
      dim3  grid_radial ((int)ceil((float)Nr/thread));
      dim3  grid_quantum ((int)ceil((float)dim_q/thread));
    //ciclo for de la cantidad de pasos
    for(int s=0 ; s<step  ; s++){

        cudaDeviceSynchronize();
        //Calculo de y_(n+1)
        y_tilde_metrics         <<< grid_radial, bloque >>>(metrics,RG_K1,RG_K2,RG_K3,RG_K4,RG_K5, y_tilde_M, s);
        y_tilde_field_classic   <<< grid_radial, bloque >>>(field_C,FC_K1,FC_K2,FC_K3,FC_K4,FC_K5, y_tilde_C, s);
        for (int g=0; g < 6; g++){
            y_tilde_field_quantum   <<< grid_quantum, bloque >>>(field_Q,FQ_K1,FQ_K2,FQ_K3,FQ_K4,FQ_K5, y_tilde_Q, s, g);
        }
        //Calculo del tensor energia momentum en el tiempo n+1 
        T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
        for (int g=0; g < 6; g++){
            Tensor_energy_momentum <<< grid_quantum, bloque >>>(rho, ja, SA, SB, y_tilde_M, y_tilde_C, y_tilde_Q, g,1);
        }    
        cudaDeviceSynchronize();

        //Actualizacion de los K
        if(s==0){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K1, rho, ja, SA, SB, Cosm);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K1,  y_tilde_M);
            for (int g=0; g < 6; g++){
              evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K1,  y_tilde_M,g,s);
            }
        }
        else if(s==1){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K2, rho, ja, SA, SB, Cosm);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K2,  y_tilde_M);
            for (int g=0; g < 6; g++){
              evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K2,  y_tilde_M, g,s);
            }
        }
        else if(s==2){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K3, rho, ja, SA, SB, Cosm);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K3,  y_tilde_M);
            for (int g=0; g < 6; g++){
              evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K3,  y_tilde_M, g,s);
            }
        }
        else if(s==3){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K4, rho, ja, SA, SB, Cosm);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K4,  y_tilde_M);
            for (int g=0; g < 6; g++){
              evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K4,  y_tilde_M, g,s);
            }
        }
        else if(s==4){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K5, rho, ja, SA, SB, Cosm);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K5,  y_tilde_M);
            for (int g=0; g < 6; g++){
              evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K5,  y_tilde_M, g,s);
            }
        }
    }
}

__global__ void next_step_metric(   Relativity_G metrics,  Relativity_G RG_K1,Relativity_G RG_K2, Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5){
  int id =  threadIdx.x + blockDim.x*blockIdx.x;
  if(id<Nr){

    metrics.A[id]      = metrics.A[id]        + dt*( b_i[0]*RG_K1.A[id]      + b_i[1]*RG_K2.A[id]      + b_i[2]*RG_K3.A[id]      + b_i[3]*RG_K4.A[id]      + b_i[4]*RG_K5.A[id] );
    metrics.B[id]      = metrics.B[id]        + dt*( b_i[0]*RG_K1.B[id]      + b_i[1]*RG_K2.B[id]      + b_i[2]*RG_K3.B[id]      + b_i[3]*RG_K4.B[id]      + b_i[4]*RG_K5.B[id] );
    metrics.alpha[id]  = metrics.alpha[id]    + dt*( b_i[0]*RG_K1.alpha[id]  + b_i[1]*RG_K2.alpha[id]  + b_i[2]*RG_K3.alpha[id]  + b_i[3]*RG_K4.alpha[id]  + b_i[4]*RG_K5.alpha[id] );
    if(id==0){
      printf("alpha=%.15f\n",metrics.alpha[id]);
    }
    if(metrics.alpha[id] > 1.0){
      metrics.alpha[id] = 1.0;
    }
    metrics.Da[id]     = metrics.Da[id]       + dt*( b_i[0]*RG_K1.Da[id]     + b_i[1]*RG_K2.Da[id]     + b_i[2]*RG_K3.Da[id]     + b_i[3]*RG_K4.Da[id]     + b_i[4]*RG_K5.Da[id] );
    metrics.Db[id]     = metrics.Db[id]       + dt*( b_i[0]*RG_K1.Db[id]     + b_i[1]*RG_K2.Db[id]     + b_i[2]*RG_K3.Db[id]     + b_i[3]*RG_K4.Db[id]     + b_i[4]*RG_K5.Db[id] );
    metrics.Kb[id]     = metrics.Kb[id]       + dt*( b_i[0]*RG_K1.Kb[id]     + b_i[1]*RG_K2.Kb[id]     + b_i[2]*RG_K3.Kb[id]     + b_i[3]*RG_K4.Kb[id]     + b_i[4]*RG_K5.Kb[id] );
    metrics.K[id]      = metrics.K[id]        + dt*( b_i[0]*RG_K1.K[id]      + b_i[1]*RG_K2.K[id]      + b_i[2]*RG_K3.K[id]      + b_i[3]*RG_K4.K[id]      + b_i[4]*RG_K5.K[id] );
    metrics.lambda[id] = metrics.lambda[id]   + dt*( b_i[0]*RG_K1.lambda[id] + b_i[1]*RG_K2.lambda[id] + b_i[2]*RG_K3.lambda[id] + b_i[3]*RG_K4.lambda[id] + b_i[4]*RG_K5.lambda[id] );
    metrics.U[id]      = metrics.U[id]        + dt*( b_i[0]*RG_K1.U[id]      + b_i[1]*RG_K2.U[id]      + b_i[2]*RG_K3.U[id]      + b_i[3]*RG_K4.U[id]      + b_i[4]*RG_K5.U[id] );
    if(id==0){
      metrics.Da[id]=0.0;
      metrics.Db[id]=0.0;
      metrics.lambda[id]=0.0;
      metrics.U[id]=0.0;
    }
  }
}
__global__ void next_step_classic(  field_Classic field_C,  field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5){
  int id =  threadIdx.x + blockDim.x*blockIdx.x;
  if(id<Nr){
    field_C.phi[id] = field_C.phi[id] + dt*( b_i[0]* FC_K1.phi[id]  + b_i[1]* FC_K2.phi[id] + b_i[2]* FC_K3.phi[id] + b_i[3]* FC_K4.phi[id] + b_i[4]* FC_K5.phi[id]);
    field_C.pi[id]  = field_C.pi[id]  + dt*( b_i[0]* FC_K1.pi[id]   + b_i[1]* FC_K2.pi[id]  + b_i[2]* FC_K3.pi[id]  + b_i[3]* FC_K4.pi[id]  + b_i[4]* FC_K5.pi[id]);
    field_C.psi[id] = field_C.psi[id] + dt*( b_i[0]* FC_K1.psi[id]  + b_i[1]* FC_K2.psi[id] + b_i[2]* FC_K3.psi[id] + b_i[3]* FC_K4.psi[id] + b_i[4]* FC_K5.psi[id]);
  }
}
__global__ void next_step_quantum(  field_Quantum field_Q,  field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5, int g){
  int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
  if(id < 6*dim_q){

    field_Q.real.u[id]   = field_Q.real.u[id]   + dt*( b_i[0]* FQ_K1.real.u[id]     + b_i[1]* FQ_K2.real.u[id]    + b_i[2]* FQ_K3.real.u[id]    + b_i[3]* FQ_K4.real.u[id]    + b_i[4]* FQ_K5.real.u[id]);
    field_Q.real.pi[id]  = field_Q.real.pi[id]  + dt*( b_i[0]* FQ_K1.real.pi[id]    + b_i[1]* FQ_K2.real.pi[id]   + b_i[2]* FQ_K3.real.pi[id]   + b_i[3]* FQ_K4.real.pi[id]   + b_i[4]* FQ_K5.real.pi[id]);
    field_Q.real.psi[id] = field_Q.real.psi[id] + dt*( b_i[0]* FQ_K1.real.psi[id]   + b_i[1]* FQ_K2.real.psi[id]  + b_i[2]* FQ_K3.real.psi[id]  + b_i[3]* FQ_K4.real.psi[id]  + b_i[4]* FQ_K5.real.psi[id]);

    field_Q.imag.u[id]   = field_Q.imag.u[id]   + dt*( b_i[0]* FQ_K1.imag.u[id]     + b_i[1]* FQ_K2.imag.u[id]    + b_i[2]* FQ_K3.imag.u[id]    + b_i[3]* FQ_K4.imag.u[id]    + b_i[4]* FQ_K5.imag.u[id]);
    field_Q.imag.pi[id]  = field_Q.imag.pi[id]  + dt*( b_i[0]* FQ_K1.imag.pi[id]    + b_i[1]* FQ_K2.imag.pi[id]   + b_i[2]* FQ_K3.imag.pi[id]   + b_i[3]* FQ_K4.imag.pi[id]   + b_i[4]* FQ_K5.imag.pi[id]);
    field_Q.imag.psi[id] = field_Q.imag.psi[id] + dt*( b_i[0]* FQ_K1.imag.psi[id]   + b_i[1]* FQ_K2.imag.psi[id]  + b_i[2]* FQ_K3.imag.psi[id]  + b_i[3]* FQ_K4.imag.psi[id]  + b_i[4]* FQ_K5.imag.psi[id]);
  }
}

__global__ void zeros_RK( Relativity_G RG_K1,Relativity_G RG_K2, Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5,
                          field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5,
                          field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5){
    int idx =threadIdx.x + blockDim.x*blockIdx.x;
    int idx_Q;
    if(idx<Nr){
      RG_K1.A[idx]=0.0;       RG_K2.A[idx]=0.0;       RG_K3.A[idx]=0.0;       RG_K4.A[idx]=0.0;       RG_K5.A[idx]=0.0;
      RG_K1.B[idx]=0.0;       RG_K2.B[idx]=0.0;       RG_K3.B[idx]=0.0;       RG_K4.B[idx]=0.0;       RG_K5.B[idx]=0.0;
      RG_K1.alpha[idx]=0.0;   RG_K2.alpha[idx]=0.0;   RG_K3.alpha[idx]=0.0;   RG_K4.alpha[idx]=0.0;   RG_K5.alpha[idx]=0.0;
      RG_K1.Da[idx]=0.0;      RG_K2.Da[idx]=0.0;      RG_K3.Da[idx]=0.0;      RG_K4.Da[idx]=0.0;      RG_K5.Da[idx]=0.0;
      RG_K1.Db[idx]=0.0;      RG_K2.Db[idx]=0.0;      RG_K3.Db[idx]=0.0;      RG_K4.Db[idx]=0.0;      RG_K5.Db[idx]=0.0;
      RG_K1.K[idx]=0.0;       RG_K2.K[idx]=0.0;       RG_K3.K[idx]=0.0;       RG_K4.K[idx]=0.0;       RG_K5.K[idx]=0.0;
      RG_K1.Kb[idx]=0.0;      RG_K2.Kb[idx]=0.0;      RG_K3.Kb[idx]=0.0;      RG_K4.Kb[idx]=0.0;      RG_K5.Kb[idx]=0.0;
      RG_K1.U[idx]=0.0;       RG_K2.U[idx]=0.0;       RG_K3.U[idx]=0.0;       RG_K4.U[idx]=0.0;       RG_K5.U[idx]=0.0;
      RG_K1.lambda[idx]=0.0;  RG_K2.lambda[idx]=0.0;  RG_K3.lambda[idx]=0.0;  RG_K4.lambda[idx]=0.0;  RG_K5.lambda[idx]=0.0;


      FC_K1.phi[idx]=0.0; FC_K2.phi[idx]=0.0; FC_K3.phi[idx]=0.0; FC_K4.phi[idx]=0.0; FC_K5.phi[idx]=0.0;
      FC_K1.psi[idx]=0.0; FC_K2.psi[idx]=0.0; FC_K3.psi[idx]=0.0; FC_K4.psi[idx]=0.0; FC_K5.psi[idx]=0.0;
      FC_K1.pi[idx]=0.0;  FC_K2.pi[idx]=0.0;  FC_K3.pi[idx]=0.0;  FC_K4.pi[idx]=0.0;  FC_K5.pi[idx]=0.0;
      for(int g=0;g<6;g++){
        for(int k=0;k<Nk;k++){
          for(int l=0;l<Nl;l++){
            idx_Q = g*dim_q + l*Nk*Nr +k*Nr + idx;
            FQ_K1.real.u[idx_Q]=0.0;       FQ_K2.real.u[idx_Q]=0.0;    FQ_K3.real.u[idx_Q]=0.0;    FQ_K4.real.u[idx_Q]=0.0;    FQ_K5.real.u[idx_Q]=0.0;
            FQ_K1.real.psi[idx_Q]=0.0;     FQ_K2.real.psi[idx_Q]=0.0;  FQ_K3.real.psi[idx_Q]=0.0;  FQ_K4.real.psi[idx_Q]=0.0;  FQ_K5.real.psi[idx_Q]=0.0;
            FQ_K1.real.pi[idx_Q]=0.0;      FQ_K2.real.pi[idx_Q]=0.0;   FQ_K3.real.pi[idx_Q]=0.0;   FQ_K4.real.pi[idx_Q]=0.0;   FQ_K5.real.pi[idx_Q]=0.0;

            FQ_K1.imag.u[idx_Q]=0.0;    FQ_K2.imag.u[idx_Q]=0.0;    FQ_K3.imag.u[idx_Q]=0.0;    FQ_K4.imag.u[idx_Q]=0.0;    FQ_K5.imag.u[idx_Q]=0.0;
            FQ_K1.imag.psi[idx_Q]=0.0;  FQ_K2.imag.psi[idx_Q]=0.0;  FQ_K3.imag.psi[idx_Q]=0.0;  FQ_K4.imag.psi[idx_Q]=0.0;  FQ_K5.imag.psi[idx_Q]=0.0;
            FQ_K1.imag.pi[idx_Q]=0.0;   FQ_K2.imag.pi[idx_Q]=0.0;   FQ_K3.imag.pi[idx_Q]=0.0;   FQ_K4.imag.pi[idx_Q]=0.0;   FQ_K5.imag.pi[idx_Q]=0.0;
          } 
        }
      }

    }
}
void cargar_coeficcientes(double *a_ij,double *b_i, double *c_i){
    FILE *arch_a;
    arch_a=fopen("a_ij.npy","rb");
    if (arch_a==NULL)
        exit(1);
    fread( a_ij , sizeof(double) , 25 , arch_a );
    fclose(arch_a);

    FILE *arch_b;
    arch_b=fopen("b_i.npy","rb");
    if (arch_b==NULL)
        exit(1);
    fread( b_i , sizeof(double) , 5 , arch_b );
    fclose(arch_b);

    FILE *arch_c;
    arch_c=fopen("c_i.npy","rb");
    if (arch_c==NULL)
        exit(1);
    fread( c_i , sizeof(double) , 5 , arch_c );
    fclose(arch_c);
}

//Metric fields...
__device__ double initial_phi(double radio){
  return amplitude*exp(- pow(radio/width,2));
}

__device__ double initial_psi(double radio){
  return -2.0* radio/pow(width,2) *amplitude*exp(- pow(radio/width,2));
}
__global__ void metric_initial(field_Classic field_C, Relativity_G metrics, double *cosmological_constant){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double r;
  if(idx<Nr){
    r = idx*dr;

    //Initial matters fields
    field_C.phi[idx] = initial_phi(r);
    field_C.psi[idx] = initial_psi(r);
    field_C.pi[idx] = 0.0; // Si se escoge otro valor para pi, se tiene que rescalar con *sqrt(A), debido que A no vale 1 en todo el espacio.
    ///Initial metrics fields
    metrics.A[idx]=1.0; //Valor provisorio para A, con el fin de tener A=1.0 en el origen y poder calcular la constante cosmologica, usando A_RK calculo nuevamente A
    metrics.B[idx]=1.0;
    metrics.alpha[idx]=1.0;
    metrics.Da[idx]=0.0;
    metrics.Db[idx]=0.0;
    metrics.K[idx]=0.0;
    metrics.Kb[idx]=0.0;

  }
}

__device__ double initial_A(double A, int i, double r, double cosmological_constant, int s){
  double dr_psi, sol;
  double rs;
 //double c_atrasada[3]={1.5, -2.0, 0.5};
  dr_psi=0.0;
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
  else if (i > Nr-(int)(order-1)/2-1 && i<Nr-1){
    dr_psi = (0.5*initial_psi(r+dr) - 0.5*initial_psi(r-dr));
  }
  else if (i == Nr-1){
    dr_psi = (initial_psi(r) - initial_psi(r-dr));
  }
  else{
    for (int m=0;m<order;m++){
      rs= r + dr*(m-(int)(order-1)/2);
      dr_psi += coefficient_centrada[m]*initial_psi(rs);
    }
  }
  dr_psi /= dr;
  if(r==0){
    sol = A*(pow(dr_psi,2)*0.5*r + r*A*cosmological_constant);
  }
  else{
    sol = A*( (1.0-A)/r + r*pow(dr_psi,2)*0.5 + ct*r*A*cosmological_constant);
  }

  return sol;
}
__global__ void A_RK(Relativity_G metrics, double *cosmological_constant){
  
  double K1,K2,K3,K4;
  double radio;

  metrics.A[0]=1.0;

  for (int i=0; i<Nr-1 ; i++){
    radio = i*dr;

    K1 = initial_A(metrics.A[i]             , i, radio          , cosmological_constant[0],0);
    
    K2 = initial_A(metrics.A[i] + dr*0.5*K1 , i, radio + 0.5*dr , cosmological_constant[0],1);

    K3 = initial_A(metrics.A[i] + dr*0.5*K2 , i, radio + 0.5*dr , cosmological_constant[0],2);

    K4 = initial_A(metrics.A[i] + dr*K3     , i, radio + dr     , cosmological_constant[0],3);

    metrics.A[i+1]=metrics.A[i] + dr*(1.0/6.0*K1 + 1.0/3.0*K2 + 1.0/3.0 *K3 + 1.0/6.0*K4);
    //printf("A[%d] = %.10f\n",i,metrics.A[i]);
  }
}
__global__ void initial_lambda(Relativity_G metrics){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;

  if(idx<Nr){
    radio = idx*dr;
    if(idx==0){
      metrics.lambda[0]=0.0;
    }
    else{
      metrics.lambda[idx] = (1.0 - metrics.A[idx]/metrics.B[idx])/radio;
    }
  }
}
__global__ void initial_U(Relativity_G metrics){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  double drA;
  if(idx<Nr){
    drA = derivate(metrics.A,0,idx,0);
    metrics.U[idx] = (drA - 4.0*metrics.lambda[idx])/metrics.A[idx];
      //printf("U[%d] = %.10f\n",idx,metrics.U[idx]);
  }
}
//C.I
void quantum_initial(field_Quantum field_Q, int k, int l, int field, double mass){
    int id_u= field + l*Nk*Nr + k*Nr ;
    double omega = sqrt( dk*(k+1)*dk*(k+1) + mass*mass ) ;
    for(int r=0; r<Nr; r++){
        //Hay un problema en r=0, debido a que u = 0/0, por lo que indetermina, por lo que, se eligio escoger una aproximacion asintotica.

        //Aproximacion a kr << 1, considerando dr=0.025 y dk=pi/15 ~ 0.2, entonces para que recien k*r tenga un valor igual a 1 
        // considerando dk*dr*i (k=1), entonces i debria ser 200 aproximadamente, ahora considerando k=Nk (por ahora Nk =20),
        // dk*21 ~ 4.2, por lo tanto para que dr*i baje este valor y usar una condicion asintotica  podemos considerar  un margen
        // de que kr<0.1 para que se cumpla, entonces dr*i < 0.1/4.2 ~ 0.025, entonces para que sea aceptable i debe ser 1
        // podemos elegir ahora nuestro rango de r para que cumpla la condicion asintotica, la cual se eligira  entre [0,10].

        // Nota: la aproximacion dependera de k, para frecuencias altas la condicion inicial en r pequeños no sera del todo precisa
        // lo cual dejará abierta la posibilidad de modificar esta seccion para que dependa de un rango apropiado de k tambien.
        if (r == 0 ){
          if(2*l+1 < GSL_SF_DOUBLEFACT_NMAX){
        //if(l==0){
                field_Q.real.u[id_u + r] = (dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1) ;
                field_Q.imag.u[id_u + r] = 0.0;
                
                field_Q.real.pi[id_u + r] = 0.0 ;
                field_Q.imag.pi[id_u + r] = -omega*(dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1);
                
                field_Q.real.psi[id_u + r] = (pow(dk*(k+1),l+3)/sqrt(consta_pi * omega)) * (-1)*(dr*r) /gsl_sf_doublefact(2*l+3);
                field_Q.imag.psi[id_u + r] = 0.0;
          }
          else{
                field_Q.real.u[id_u + r] = 0.0;
                field_Q.imag.u[id_u + r] = 0.0;
                
                field_Q.real.pi[id_u + r] = 0.0 ;
                field_Q.imag.pi[id_u + r] = 0.0;
                
                field_Q.real.psi[id_u + r] = 0.0;
                field_Q.imag.psi[id_u + r] = 0.0;
          }
            //u[id_u + r].x=0.0;
        }
        else{
                field_Q.real.u[id_u + r] = (dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l) ;
                field_Q.imag.u[id_u + r] = 0.0;
                
                field_Q.real.pi[id_u + r] = 0.0 ;
                field_Q.imag.pi[id_u + r] = -omega*(dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l);
                
                field_Q.real.psi[id_u + r] = ( pow(dk*(k+1),2) /sqrt(consta_pi * omega)) * (-1) * gsl_sf_bessel_jl( l +1, dk*(k+1)*dr*(r)  )/ (pow(dr*r,l));
                field_Q.imag.psi[id_u + r] = 0.0;
        }
    }
}

int main() {
    size_t bytes_Nr = Nr * sizeof(double);
    size_t bytes_Q = 6 * Nr*Nl*Nk * sizeof(double);
    // dejo el campo cuantico escalar y fantasmas  en un solo malloc, asi evito repetir funciones.
    double * a_temp;
    double * _host_b_i;
    double * _host_c_i;
    //Cargamos los coeficientes de runge kutta
    a_temp =(double *)malloc(25*sizeof(double));
    _host_b_i =(double *)malloc(5*sizeof(double));
    _host_c_i =(double *)malloc(5*sizeof(double));

    
    cargar_coeficcientes(a_temp,_host_b_i,_host_c_i);
    double (*_host_a_ij)[5]=(double(*)[5])a_temp;

    //Copiamos los coeficientes del runge kutta del host al device
    cudaMemcpyToSymbol(a_ij, _host_a_ij, 25*sizeof(double));
    cudaMemcpyToSymbol(b_i , _host_b_i,  5*sizeof(double));
    cudaMemcpyToSymbol(c_i , _host_c_i,  5*sizeof(double));


    // Asignar memoria para la estructura en el host
    Relativity_G metrics, RG_K1, RG_K2, RG_K3, RG_K4,RG_K5, y_tilde_M;
    field_Classic field_C, FC_K1, FC_K2, FC_K3, FC_K4, FC_K5, y_tilde_C;
    field_Quantum field_Q, FQ_K1, FQ_K2, FQ_K3, FQ_K4, FQ_K5, y_tilde_Q;
    double *rho, *ja, *SA, *SB;
    double * cosmological_constant;

    //field_Ghost field_G;
    cudaMalloc((void **)&(metrics), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(RG_K1), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(RG_K2), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(RG_K3), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(RG_K4), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(RG_K5), sizeof(Relativity_G)); 
    cudaMalloc((void **)&(y_tilde_M), sizeof(Relativity_G)); 

    cudaMalloc((void **)&(field_C), sizeof(field_Classic));
    cudaMalloc((void **)&(FC_K1), sizeof(field_Classic));
    cudaMalloc((void **)&(FC_K2), sizeof(field_Classic));
    cudaMalloc((void **)&(FC_K3), sizeof(field_Classic));
    cudaMalloc((void **)&(FC_K4), sizeof(field_Classic));
    cudaMalloc((void **)&(FC_K5), sizeof(field_Classic));
    cudaMalloc((void **)&(y_tilde_C), sizeof(field_Classic));

    cudaMalloc((void **)&(field_Q), sizeof(field_Quantum));
    cudaMalloc((void **)&(FQ_K1), sizeof(field_Quantum));
    cudaMalloc((void **)&(FQ_K2), sizeof(field_Quantum));
    cudaMalloc((void **)&(FQ_K3), sizeof(field_Quantum));
    cudaMalloc((void **)&(FQ_K4), sizeof(field_Quantum));
    cudaMalloc((void **)&(FQ_K5), sizeof(field_Quantum));
    cudaMalloc((void **)&(y_tilde_Q), sizeof(field_Quantum));


    Relativity_G *point_RG[]     = {&metrics, &RG_K1, &RG_K2, &RG_K3, &RG_K4, &RG_K5, &y_tilde_M};
    field_Classic *point_FC[]   = {&field_C, &FC_K1, &FC_K2, &FC_K3, &FC_K4, &FC_K5, &y_tilde_C};
    field_Quantum *point_FQ[]    = {&field_Q, &FQ_K1, &FQ_K2, &FQ_K3, &FQ_K4, &FQ_K5, &y_tilde_Q};
/*
    printf("si\n");


    // Asignar memoria para la estructura en el device
        printf("si\n");

    cudaMalloc((void **)&(metrics), sizeof(Relativity_G)); // Asigna memoria para la estructura en el device
    printf("si\n");

    // Asignar memoria para los arrays en el device
    //Metrics
    cudaMalloc((void **)&(metrics.A), bytes_Nr);  
    printf("si 7\n");

    cudaMalloc((void **)&(metrics.B     ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.alpha ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.Da    ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.Db    ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.K     ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.Kb    ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.U     ), bytes_Nr);   
    cudaMalloc((void **)&(metrics.lambda), bytes_Nr);   

    //Classic
    cudaMalloc((void **)&(field_C), sizeof(field_Classic)); 
    cudaMalloc((void **)&(field_C.phi), bytes_Nr);  
    cudaMalloc((void **)&(field_C.pi), bytes_Nr);   
    cudaMalloc((void **)&(field_C.psi), bytes_Nr);

    //Quamtum
    cudaMalloc((void **)&(field_Q), sizeof(field_Quantum)); 
    cudaMalloc((void **)&(field_Q.phi), bytes_Q);  
    cudaMalloc((void **)&(field_Q.pi), bytes_Q);   
    cudaMalloc((void **)&(field_Q.psi), bytes_Q);
*/

    for (int i = 0; i < sizeof(point_RG) / sizeof(point_RG[0]); ++i) {
        //Metrics
        //cudaMalloc((void **)&(point_RG[i]), sizeof(Relativity_G)); 
        cudaMalloc((void **)&(point_RG[i]->A), bytes_Nr);  
        cudaMalloc((void **)&(point_RG[i]->B     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->alpha ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->Da    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->Db    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->K     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->Kb    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->U     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i]->lambda), bytes_Nr); 
        
        //Classic
        //cudaMalloc((void **)&(point_FC[i]), sizeof(field_Classic)); 
        cudaMalloc((void **)&(point_FC[i]->phi), bytes_Nr);  
        cudaMalloc((void **)&(point_FC[i]->pi), bytes_Nr);   
        cudaMalloc((void **)&(point_FC[i]->psi), bytes_Nr);

        //Quamtum
        //cudaMalloc((void **)&(point_FQ[i]), sizeof(field_Quantum)); 
        //real
        cudaMalloc((void **)&(point_FQ[i]->real.u), bytes_Q);  
        cudaMalloc((void **)&(point_FQ[i]->real.pi), bytes_Q);   
        cudaMalloc((void **)&(point_FQ[i]->real.psi), bytes_Q);
        //imag
        cudaMalloc((void **)&(point_FQ[i]->imag.u), bytes_Q);  
        cudaMalloc((void **)&(point_FQ[i]->imag.pi), bytes_Q);   
        cudaMalloc((void **)&(point_FQ[i]->imag.psi), bytes_Q);
      }

    cudaMalloc((void **)&(cosmological_constant), sizeof(double));
    cudaMalloc((void **)&(rho), Nr*sizeof(double)); 
    cudaMalloc((void **)&(ja), Nr*sizeof(double)); 
    cudaMalloc((void **)&(SA), Nr*sizeof(double)); 
    cudaMalloc((void **)&(SB), Nr*sizeof(double)); 
    //cudaMalloc((void **)&(field_Q.real.u), bytes_Q);  
    //cudaMalloc((void **)&(field_Q.real.pi), bytes_Q);   
    //cudaMalloc((void **)&(field_Q.real.psi), bytes_Q);
    //Condiciones iniciales

    int thread = 64;
    ////Simulacion////
    dim3  bloque(thread);
    dim3  grid_radial ((int)ceil((float)Nr/thread));
    dim3  grid_quantum ((int)ceil((float)dim_q/thread));
    printf("thread = %d , block_radial = %d, block_quantum = %d \n", thread , (int)Nr/thread,(int)ceil((float)(Nr*Nk*Nl)/thread)  );
    printf("Nr = %d | Nt = %d | Nk = %d | Nl = %d | dr = %f | dt = %f | dk = %lf |\n",Nr,Nt,Nk,Nl, dr,dt,dk);
    printf("Amplitud = %f | width = %f\n",amplitude,width);


    //Metrica y campo clasico   
    metric_initial<<<grid_radial,bloque>>>( field_C,  metrics, cosmological_constant);
    double mass;
    A_RK<<<1,1>>>( metrics, cosmological_constant);

    cudaDeviceSynchronize();
    //Campo cuantico
    field_Quantum host_field_Q;
    host_field_Q.real.u =(double*)malloc(bytes_Q);
    host_field_Q.real.psi =(double*)malloc(bytes_Q);
    host_field_Q.real.pi =(double*)malloc(bytes_Q);

    host_field_Q.imag.u =(double*)malloc(bytes_Q);
    host_field_Q.imag.psi =(double*)malloc(bytes_Q);
    host_field_Q.imag.pi =(double*)malloc(bytes_Q);

    for( int g=0;g<6;g++){
      for (int k=0;k<Nk;k++){
        for (int l=0;l<Nl;l++){
          if(g==0){
            mass=0.0;
          }
          else if(g==1 || g==3){
            mass=1.0;
          }
          else if(g==2 || g==4){
            mass=1*sqrt(3.0);
          }
          else if(g==5){
            mass=1*sqrt(4.0);
          }
          quantum_initial( host_field_Q, k, l, g*dim_q, mass);
        }
      }
    }

    cudaMemcpy(field_Q.real.u,    host_field_Q.real.u   , bytes_Q, cudaMemcpyHostToDevice );
    cudaMemcpy(field_Q.real.psi,  host_field_Q.real.psi , bytes_Q, cudaMemcpyHostToDevice );
    cudaMemcpy(field_Q.real.pi,   host_field_Q.real.pi  , bytes_Q, cudaMemcpyHostToDevice );

    cudaMemcpy(field_Q.imag.u,    host_field_Q.imag.u   , bytes_Q, cudaMemcpyHostToDevice );
    cudaMemcpy(field_Q.imag.psi,  host_field_Q.imag.psi , bytes_Q, cudaMemcpyHostToDevice );
    cudaMemcpy(field_Q.imag.pi,   host_field_Q.imag.pi  , bytes_Q, cudaMemcpyHostToDevice );

    cudaDeviceSynchronize();

    T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
    cudaDeviceSynchronize();

    for (int g=0; g < 6; g++){
        Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, g,1);
    }

      cost_cosm_value<<<1,1>>>(rho,cosmological_constant);
    cudaDeviceSynchronize();

      A_RK<<<1,1>>>( metrics, cosmological_constant);

    cudaDeviceSynchronize();
    
      initial_lambda <<<grid_radial,bloque>>> (metrics);

    cudaDeviceSynchronize();
    
      initial_U <<<grid_radial,bloque>>> (metrics);

    cudaDeviceSynchronize();
    //Evolucion
    
    //Array temporales
    double *T_tt, *cuda_T_tt;
    double *T_rr, *cuda_T_rr;
    double *T_tr, *cuda_T_tr;
    double *T_00, *cuda_T_00;
    cudaMalloc((void **)&(cuda_T_tt), Nt*Nr*sizeof(double)); 
    cudaMalloc((void **)&(cuda_T_tr), Nt*Nr*sizeof(double)); 
    cudaMalloc((void **)&(cuda_T_rr), Nt*Nr*sizeof(double)); 
    cudaMalloc((void **)&(cuda_T_00), Nt*Nr*sizeof(double)); 

    //double *Nodos;



    for (int t=0 ; t<Nt ;t++){
      printf("t : %d\n",t);
        Tensor_tt<<< grid_radial, bloque >>>(cuda_T_tt, cuda_T_rr, cuda_T_tr, cuda_T_00, rho, SA, ja, SB, t);
        cudaDeviceSynchronize();

        T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
        for (int g=0; g < 6; g++){
          Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, g,0);
        }
        cudaDeviceSynchronize();

        zeros_RK<<< grid_radial, bloque >>>( RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5, FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5, FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5);
        cudaDeviceSynchronize();

        for(int i=0; i< 20;i++){
          RK_implicit_tenth(metrics,  RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5,  y_tilde_M,
                            field_C,  FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5,  y_tilde_C,
                            field_Q,  FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5,  y_tilde_Q,
                            rho, ja, SA, SB,  cosmological_constant );
        }
        cudaDeviceSynchronize();

        next_step_metric<<< grid_radial, bloque >>>(metrics, RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5);
        next_step_classic<<< grid_radial, bloque >>>(field_C,  FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5);
        for (int g=0; g < 6; g++){
          next_step_quantum<<< grid_quantum, bloque >>>(field_Q,  FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5, g);
        }
        cudaDeviceSynchronize();

    }
    




      //error cuda
      cudaError_t err = cudaGetLastError();
      printf("Error: %s\n",cudaGetErrorString(err));


      //Guardar datos



      T_tt =(double*)malloc(Nt*bytes_Nr);
      T_rr =(double*)malloc(Nt*bytes_Nr);
      T_tr =(double*)malloc(Nt*bytes_Nr);
      T_00 =(double*)malloc(Nt*bytes_Nr);

      cudaMemcpy(T_tt, cuda_T_tt, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
      cudaMemcpy(T_rr, cuda_T_rr ,Nt*bytes_Nr,cudaMemcpyDeviceToHost);
      cudaMemcpy(T_tr, cuda_T_tr ,Nt*bytes_Nr,cudaMemcpyDeviceToHost);
      cudaMemcpy(T_00, cuda_T_00 ,Nt*bytes_Nr,cudaMemcpyDeviceToHost);

      //guardar salida Tensor energia momentum
      for(int i =0; i<Nt;i++){
        printf("rho[%d]=%0.15f\n",i,T_tt[i*Nr + 20]);
      }
      guardar_salida_rho(T_tt,Nt);
      guardar_salida_SA(T_rr,Nt);
      guardar_salida_ja(T_tr,Nt);
      guardar_salida_SB(T_00,Nt);

      size_t free_memory, total_memory;
      cudaMemGetInfo( &free_memory, &total_memory );

      //Memoria
      printf("Memoria libre: %zu bytes\n", free_memory);
      printf("Memoria total: %zu bytes\n", total_memory);

      //  MB para mayor claridad
      printf("Memoria libre: %.2f MB\n", free_memory / (1024.0 * 1024.0));
      printf("Memoria total: %.2f MB\n", total_memory / (1024.0 * 1024.0));


    for (int i = 0; i < sizeof(point_RG) / sizeof(point_RG[0]); ++i) {
      //Metrics
      //cudaFree(point_RG[i]); 
      cudaFree(point_RG[i]->A);  
      cudaFree(point_RG[i]->B     );   
      cudaFree(point_RG[i]->alpha );   
      cudaFree(point_RG[i]->Da    );   
      cudaFree(point_RG[i]->Db    );   
      cudaFree(point_RG[i]->K     );   
      cudaFree(point_RG[i]->Kb    );   
      cudaFree(point_RG[i]->U     );   
      cudaFree(point_RG[i]->lambda); 
      
      //Classic
      //cudaFree(point_FC[i]); 
      cudaFree(point_FC[i]->phi);  
      cudaFree(point_FC[i]->pi);   
      cudaFree(point_FC[i]->psi);

      //Quamtum
      //cudaFree(point_FQ[i]); 
      cudaFree(point_FQ[i]->imag.u);  
      cudaFree(point_FQ[i]->imag.pi);   
      cudaFree(point_FQ[i]->imag.psi);

      cudaFree(point_FQ[i]->real.u);  
      cudaFree(point_FQ[i]->real.pi);   
      cudaFree(point_FQ[i]->real.psi);
  }
  cudaFree(cosmological_constant);

}