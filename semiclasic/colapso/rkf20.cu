
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

/*
__device__ double diff_tenth_order[order] 
    ={-504.*420.*21.*6.,  5.*1260.*420.*21.*6.,
     -25.*1260.*504.*21.*6.,  5.*1260.*504.*420.*6.,
    -5.*1260.*504.*25.*420.,  0.00000000e+00, 5.*1260.*504.*25.*420., 
    -5.*1260.*504.*420.*6.,25.*1260.*504.*21.*6., 
    -5.*1260.*420.*21.*6.,  504.*420.*21.*6.,
  };
#define denominador 1260.*504.*420.*21.*6.  

__device__ double diff_tenth_order[order] 
    ={-22.,  275.,
     -1650.,  6600.,
    -27500.,  0.0, 27500., 
    -6600.,1650., 
    -275.,  22.,
  };
#define denominador 27720.
*/ 
#define order 11
__device__ double diff_tenth_order[order] ={-1, 5*2.5, -5*15, 5*60, -5*210, 0.0, 5*210, -5*60, 5*15, -5*2.5, 1};
#define half_order_tenth_right (int)(order+1)/2
#define half_order_tenth_left (int)(order-1)/2
#define denominador 1260
/*

#define order 7
  __device__ double diff_tenth_order[order] ={-1, 3*3 , -3*15,0.0, 3*15, -3*3, 1};
#define half_order_tenth_right (int)(order+1)/2
#define half_order_tenth_left (int)(order-1)/2
#define denominador 60
*/

/*
#define derivate_twelfth 13
__device__ double diff_twelfth_order[derivate_twelfth] = {1.,  -12.,   66., -220.,  495., -792.,  924., -792.,  495.,
   -220.,   66., -12.,1.
  };
#define half_derivate_twelfth_right (int)(derivate_twelfth+1)/2
#define half_derivate_twelfth_left (int)(derivate_twelfth-1)/2
*/
#define derivate_twelfth 5
__device__ double diff_twelfth_order[derivate_twelfth] = {1.,-4.,6.,-4.,1.};
#define half_derivate_twelfth_right (int)(derivate_twelfth+1)/2
#define half_derivate_twelfth_left (int)(derivate_twelfth-1)/2

/*
#define derivate_twelfth 7
__device__ int diff_twelfth_order[derivate_twelfth] = {-1,2*6,-13*3,28*2,-13*3,2*6,-1};
#define half_derivate_twelfth_right (int)(derivate_twelfth+1)/2
#define half_derivate_twelfth_left (int)(derivate_twelfth-1)/2
*/
#define epsilon_metric_field 0.1 
#define epsilon_quantum_field 0.49

// Parameter
#define Nr 2000
#define dr 0.025
#define Nt 2000
#define dt dr/4.0

//Parameter classic
#define amplitude 5.0
#define width 1.0
#define r0 dr*(Nr-Nr)

// Parameter quantum
#define Nk 20
#define Nl 20
#define dk consta_pi/15.0

#define  dim_q Nk*Nl*Nr


#define ct 1.0
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
void guardar_salida_phi(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/campo_escalar.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_pi(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/pi.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_psi(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/psi.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_alpha(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/g_tt.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_B(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/g_00.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_nodo(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/nodo.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_A(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/g_rr.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_rho(double *data,int T) {
  FILE *fp = fopen("20x20/BH/2000/rho.dat", "wb");
  fwrite(data, sizeof(double), Nr*T, fp);
  fclose(fp);
}
void guardar_salida_SA(double *data, int T) {
    FILE *fp = fopen("20x20/BH/2000/SA.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_ja(double *data, int T) {
    FILE *fp = fopen("20x20/BH/2000/ja.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
void guardar_salida_SB(double *data,int T) {
    FILE *fp = fopen("20x20/BH/2000/SB.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
  void guardar_salida_geo(double *data,int T) {
    FILE *fp = fopen("20x20/BH/2000/geodesicas.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
  void guardar_salida_H(double *data,int T) {
    FILE *fp = fopen("20x20/BH/2000/Hamilton.dat", "wb");
    fwrite(data, sizeof(double), Nr*T, fp);
    fclose(fp);
  }
  __device__ double psi_dot(Relativity_G metrics, double *field_pi, int id_nodo, int idx){
    double temp,sym;
    double f,fm1,fm2,fm4,fm5,fp1,fp2,fm3,fp3;
    temp = 0.0;
    fm5=metrics.alpha[idx-5]/(sqrt(metrics.A[idx-5])*metrics.B[idx-5]) * field_pi[id_nodo + idx-5];
    fm4=metrics.alpha[idx-4]/(sqrt(metrics.A[idx-4])*metrics.B[idx-4]) * field_pi[id_nodo + idx-4];
    fm3=metrics.alpha[idx-3]/(sqrt(metrics.A[idx-3])*metrics.B[idx-3]) * field_pi[id_nodo + idx-3];
    fm2=metrics.alpha[idx-2]/(sqrt(metrics.A[idx-2])*metrics.B[idx-2]) * field_pi[id_nodo + idx-2];
    fm1=metrics.alpha[idx-1]/(sqrt(metrics.A[idx-1])*metrics.B[idx-1]) * field_pi[id_nodo + idx-1];
    f=metrics.alpha[idx]/(sqrt(metrics.A[idx])*metrics.B[idx]) * field_pi[id_nodo + idx];
    fp1=metrics.alpha[idx+1]/(sqrt(metrics.A[idx+1])*metrics.B[idx+1]) * field_pi[id_nodo + idx+1];
    fp2=metrics.alpha[idx+2]/(sqrt(metrics.A[idx+2])*metrics.B[idx+2]) * field_pi[id_nodo + idx+2];
    fp3=metrics.alpha[idx+3]/(sqrt(metrics.A[idx+3])*metrics.B[idx+3]) * field_pi[id_nodo + idx+3];

    if (idx < half_order_tenth_left){
        sym=1.0;
        for (int m=0; m < half_order_tenth_right + idx ; m++){
            f= metrics.alpha[m]/(sqrt(metrics.A[m])*metrics.B[m]) * field_pi[id_nodo + m];
            temp += diff_tenth_order[m + half_order_tenth_left -idx ]*(f);
        }
        for (int m=half_order_tenth_left - idx; m > 0 ; m--){
            f= metrics.alpha[m]/(sqrt(metrics.A[m])*metrics.B[m]) * field_pi[id_nodo + m];
            temp += sym*diff_tenth_order[ half_order_tenth_left - idx - m  ]*(f);
        }
        if (idx==0){
            temp=0.0;
          }
          temp=temp/denominador;

    }
    else if (idx == Nr-1){
        temp = (1.5*f - 2.0*fm1 + 0.5*fm2);
        //temp = (0.5*(f + (25*f - 4*12*fm1 + 12*3*fm2 - 4*4*fm3 + 3*1*fm4  )/12) - 0.5*fm1);
        
    }
    else if (idx == Nr-2 || idx == Nr-3){
      temp = (0.5*fp1 - 0.5*fm1);
             //temp = (1.5*f - 2.0*fm1 + 0.5*fm2);

    }
    else if (idx == Nr-4  || idx == Nr-5){
      temp = (1.0*fm2 - 2.0*4*fm1 + 2.0*4*fp1 - 1.0*fp2)/12.0;
    }
    else if (idx == Nr-6  || idx == Nr-7){
      temp = (-1.0*fm3 + 3.0*3*fm2 - 3.0*15*fm1 + 3.0*15*fp1 - 3.0*3*fp2 + 1.0*fp3)/60.0;
    }
    else{
        for (int m=0;m<order;m++){
            f=metrics.alpha[idx-half_order_tenth_left+m]/(sqrt(metrics.A[idx-half_order_tenth_left+m])*metrics.B[idx-half_order_tenth_left+m]) * field_pi[id_nodo + idx-half_order_tenth_left+m];
            temp += diff_tenth_order[m]*f;
        }
        temp=temp/denominador;

    }
    temp=temp/dr;
    return temp;
}
__device__ double f_pi_dot(field_Classic field_C, Relativity_G metrics, int idx){
  double temp,sym;
  double f,fm1,fm2,fm4,fm5,fp1,fp2,fm3,fp3;
  fm5=metrics.alpha[idx-5]/(sqrt(metrics.A[idx-5])*metrics.B[idx-5]) * field_C.psi[idx-5];
  fm4=metrics.alpha[idx-4]/(sqrt(metrics.A[idx-4])*metrics.B[idx-4]) * field_C.psi[idx-4];
      fm3=metrics.alpha[idx-3]*metrics.B[idx-3] * field_C.psi[idx-3]/sqrt(metrics.A[idx-3]);
    fm2=metrics.alpha[idx-2]*metrics.B[idx-2] * field_C.psi[idx-2]/sqrt(metrics.A[idx-2]);
    fm1=metrics.alpha[idx-1]*metrics.B[idx-1] * field_C.psi[idx-1]/sqrt(metrics.A[idx-1]);
    f=metrics.alpha[idx]*metrics.B[idx] * field_C.psi[idx]/sqrt(metrics.A[idx]);
    fp1=metrics.alpha[idx+1]*metrics.B[idx+1] * field_C.psi[idx+1]/sqrt(metrics.A[idx+1]);
    fp2=metrics.alpha[idx+2]*metrics.B[idx+2] * field_C.psi[idx+2]/sqrt(metrics.A[idx+2]);
    fp3=metrics.alpha[idx+3]*metrics.B[idx+3] * field_C.psi[idx+3]/sqrt(metrics.A[idx+3]);


  temp=0.0;

  if (idx < half_order_tenth_left){
      sym=-1.0;
      for (int m=0; m < half_order_tenth_right + idx ; m++){
            f=metrics.alpha[m]*metrics.B[m] * field_C.psi[m]/sqrt(metrics.A[m]);
            temp += diff_tenth_order[m + half_order_tenth_left -idx ]*(f);
      }
      for (int m=half_order_tenth_left - idx; m > 0 ; m--){

            f=metrics.alpha[m]*metrics.B[m] * field_C.psi[m]/sqrt(metrics.A[m]);
            temp += sym*diff_tenth_order[ half_order_tenth_left - idx - m  ]*(f);
      }
      temp=temp/denominador;

  }
    else if (idx == Nr-1){
        temp = (1.5*f - 2.0*fm1 + 0.5*fm2);
        //temp = (0.5*(f + (25*f - 4*12*fm1 + 12*3*fm2 - 4*4*fm3 + 3*1*fm4  )/12) - 0.5*fm1);

    }
    else if (idx == Nr-2 || idx == Nr-3){
    temp = (0.5*fp1 - 0.5*fm1);
    //temp = (1.5*f - 2.0*fm1 + 0.5*fm2);

    }
    else if (idx == Nr-4  || idx == Nr-5){
    temp = (1.0*fm2 - 2.0*4*fm1 + 2.0*4*fp1 - 1.0*fp2)/12.0;
    }
    else if (idx == Nr-6  || idx == Nr-7){
    temp = (-1.0*fm3 + 3.0*3*fm2 - 3.0*15*fm1 + 3.0*15*fp1 - 3.0*3*fp2 + 1.0*fp3)/60.0;
    }
  else{
      for (int m=0;m<order;m++){
          f=metrics.alpha[idx-half_order_tenth_left+m]*metrics.B[idx-half_order_tenth_left+m] * field_C.psi[idx-half_order_tenth_left+m]/sqrt(metrics.A[idx-half_order_tenth_left+m]);
          temp += diff_tenth_order[m]*f;
      }
              temp=temp/denominador;
  }
  temp=temp/dr;
  return temp;
}
__device__ double derivate_metric(Relativity_G metrics, int idx){
    double temp,sym;
    double f,fm1,fm2,fm4,fm5,fp1,fp2,fm3,fp3;
    temp=0.0;
    fp3=metrics.alpha[idx+3]*metrics.B[idx+3]/sqrt(metrics.A[idx+3]);
    fp2=metrics.alpha[idx+2]*metrics.B[idx+2]/sqrt(metrics.A[idx+2]);
    fp1=metrics.alpha[idx+1]*metrics.B[idx+1]/sqrt(metrics.A[idx+1]);
    f=metrics.alpha[idx]*metrics.B[idx]/sqrt(metrics.A[idx]);
    fm1=metrics.alpha[idx-1]*metrics.B[idx-1]/sqrt(metrics.A[idx-1]);
    fm2=metrics.alpha[idx-2]*metrics.B[idx-2]/sqrt(metrics.A[idx-2]);
    fm3=metrics.alpha[idx-3]*metrics.B[idx-3]/sqrt(metrics.A[idx-3]);
    fm4=metrics.alpha[idx-4]*metrics.B[idx-4]/sqrt(metrics.A[idx-4]);
    fm5=metrics.alpha[idx-5]*metrics.B[idx-5]/sqrt(metrics.A[idx-5]);
    if (idx < half_order_tenth_left){
        sym=1.0;
        for (int m=0; m < half_order_tenth_right + idx ; m++){
            f=metrics.alpha[m]*metrics.B[m]/sqrt(metrics.A[m]);
            temp += diff_tenth_order[m + half_order_tenth_left -idx ]*(f);
        }
        for (int m=half_order_tenth_left - idx; m > 0 ; m--){
            f=metrics.alpha[m]*metrics.B[m]/sqrt(metrics.A[m]);
            temp += sym*diff_tenth_order[ half_order_tenth_left - idx - m  ]*(f);
        }
        if (idx==0){
            temp=0.0;
          }
          temp=temp/denominador;

    }
    else if (idx == Nr-1){
        temp = (1.5*f - 2.0*fm1 + 0.5*fm2);
        //temp = (0.5*(f + (25*f - 4*12*fm1 + 12*3*fm2 - 4*4*fm3 + 3*1*fm4  )/12) - 0.5*fm1);

    }
    else if (idx == Nr-2 ){
      temp = (0.5*fp1 - 0.5*fm1);
      //temp = (1.5*f - 2.0*fm1 + 0.5*fm2);

    }
    else if (idx == Nr-4  || idx == Nr-5 || idx == Nr-3){
      temp = (1.0*fm2 - 2.0*4*fm1 + 2.0*4*fp1 - 1.0*fp2)/12.0;
    }
    else if (idx == Nr-6  || idx == Nr-7){
      temp = (-1.0*fm3 + 3.0*3*fm2 - 3.0*15*fm1 + 3.0*15*fp1 - 3.0*3*fp2 + 1.0*fp3)/60.0;
    }
    else{
        for (int m=0;m<order;m++){
            f=metrics.alpha[idx-half_order_tenth_left+m]*metrics.B[idx-half_order_tenth_left+m]/sqrt(metrics.A[idx-half_order_tenth_left+m]);
            temp += diff_tenth_order[m]*f;
        }
        temp=temp/denominador;
    }
    temp=temp/dr;
    return temp;

}
__device__ double derivate( double *f, int id_nodo, int idx, int symmetric ){
    double temp,sym;
    double fm0,fm1,fm2,fm3,fm4,fm5;
    fm0=f[id_nodo + idx];
    fm1=f[id_nodo + idx-1];
    fm2=f[id_nodo + idx-2];
    fm3=f[id_nodo + idx-3];
    fm4=f[id_nodo + idx-4];
    fm5=f[id_nodo + idx-5];
      temp=0.0;
  
      if (idx < half_order_tenth_left){
        if (symmetric==0){
          sym=1.0;
        }
        else if (symmetric == 1){
          sym=-1.0;
        }
  
        for (int m=0; m < half_order_tenth_right + idx ; m++){
  
            temp += diff_tenth_order[m + half_order_tenth_left -idx ]*(f[id_nodo + m]);
            
  
          }
        for (int m=half_order_tenth_left - idx; m > 0 ; m--){
  
            temp += sym*diff_tenth_order[ half_order_tenth_left - idx - m  ]*(f[id_nodo + m]);
            
          }
        temp=temp/(denominador*dr);
        if (symmetric==0 && idx==0){
            temp=0.0;
          }
      }
      
      //else if (idx > Nr-half_order_tenth_left-1 && idx<Nr-1-1){
     /*else if ( idx > Nr-6  ){

        //temp = (1*f[id_nodo + idx-2] - 4*2.0*f[id_nodo + idx -1] + 4*2.0*f[id_nodo + idx + 1] -1*f[id_nodo + idx+2])/12.0; 
        temp = -(-49*3*f[id_nodo + idx] + 6*60*f[id_nodo + idx-1] - 30*15*f[id_nodo + idx-2] + 20*20*f[id_nodo + idx-3] - 15*15*f[id_nodo + idx-4] + 6*12*f[id_nodo + idx-5] - 10*f[id_nodo + idx-6])/60.0;

        temp=temp/(dr);
      }
     
      else if (idx == Nr-1){
        //temp = (-25*f[id_nodo + idx] + 12*4*f[id_nodo + idx-1] - 12*3*f[id_nodo + idx-2] + 4*4*f[id_nodo + idx-3] - 3*1*f[id_nodo + idx-4])/12.0;
        //temp = (11*f[id_nodo + idx] - 6*3*f[id_nodo + idx-1] + 3*3*f[id_nodo + idx-2] - 2*1*f[id_nodo + idx-3])/6.0;
        temp = (1*f[id_nodo + idx] - 1*f[id_nodo + idx-1] );

        temp=temp/(dr);

      }
      else if (idx == Nr-2){
        temp = (1*f[id_nodo + idx+1] - 1*f[id_nodo + idx-1] );
        temp=temp/(2.0*dr);

      }
       
      else if (idx > Nr-half_order_tenth_left-1 ){
        //temp = (0.5*f[id_nodo + idx+1] - 0.5*f[id_nodo + idx-1]); 
        for (int m=0;m<order;m++){
          temp += diff_tenth_order[order-1 - m]*f[id_nodo + idx - m];
        }
        //temp =0.0;
      } */
 
      else if (idx == Nr-1){
        temp = (1.5*f[id_nodo + idx] - 2.0*f[id_nodo + idx-1] + 0.5*f[id_nodo + idx-2]);
        //temp = (0.5*(fm0 + (25*fm0 - 4*12*fm1 + 12*3*fm2 - 4*4*fm3 + 3*1*fm4  )/12) - 0.5*fm1);

        temp=temp/(dr);
      }

      else if (idx == Nr-2 ){
        temp = (0.5*f[id_nodo + idx + 1] - 0.5*f[id_nodo + idx-1]);
        //temp = (1.5*f[id_nodo + idx] - 2.0*f[id_nodo + idx-1] + 0.5*f[id_nodo + idx-2]);

        temp=temp/(dr);
      }
      else if (idx == Nr-4  || idx == Nr-5 || idx == Nr-3){
        temp = (1.0*f[id_nodo + idx-2] - 2.0*4*f[id_nodo + idx-1] + 2.0*4*f[id_nodo + idx+1] - 1.0*f[id_nodo + idx+2])/12.0;
        temp=temp/(dr);
      }
      else if (idx == Nr-6  || idx == Nr-7){
        temp = (-1.0*f[id_nodo + idx-3] + 3.0*3*f[id_nodo + idx-2] - 3.0*15*f[id_nodo + idx -1 ] + 3.0*15*f[id_nodo + idx +1] - 3.0*3*f[id_nodo + idx + 2] + 1.0*f[id_nodo + idx +3 ])/60.0;
        temp=temp/(dr);
      }
      else{
        for (int m=0;m<order;m++){
          temp += diff_tenth_order[m]*f[id_nodo + idx-half_order_tenth_left+m];
        }
        temp=temp/(denominador*dr);
      }
      return temp;
}

__device__ double twelfth_derivate( double *f, int id_nodo, int idx, int symmetric){
    double temp,sym;
  
      temp=0.0;
  
      if (idx < half_derivate_twelfth_left){
        if (symmetric==0){
          sym=1.0;
        }
        else if (symmetric == 1){
          sym=-1.0;
        }
  
        for (int m=0; m < half_derivate_twelfth_right + idx ; m++){
  
            temp += diff_twelfth_order[m + half_derivate_twelfth_left -idx ]*(f[id_nodo + m]);
  
  
          }
        for (int m=half_derivate_twelfth_left - idx; m > 0 ; m--){
  
            temp += sym*diff_twelfth_order[ half_derivate_twelfth_left - idx - m  ]*(f[id_nodo + m]);
  
          }
          if (symmetric==1 && idx==0){
              temp=0.0;
      }
      temp=temp/1.0;
      }

  
      else if (idx == Nr-1 ){
          //temp = (1.5*f[id_nodo + idx] - 2.0*f[id_nodo + idx-1] + 0.5*f[id_nodo + idx-2]);

          //temp = (1.5*f[id_nodo + idx] - 2.0*f[id_nodo + idx-1] + 0.5*f[id_nodo + idx-2]);
          //temp = (1*f[id_nodo + idx ] - 4*f[id_nodo + idx -1 ]  + 6*f[id_nodo + idx-2] - 4*f[id_nodo + idx - 3] + 1*f[id_nodo + idx - 4]);
          //temp = -(28*2*f[id_nodo + idx ] - 111*3*f[id_nodo + idx -1]  + 142*6*f[id_nodo + idx-2] - 1219*f[id_nodo + idx - 3] + 176*6*f[id_nodo + idx - 4] - 185*3*f[id_nodo + idx - 5]+82*2*f[id_nodo + idx - 6] -7*3*f[id_nodo + idx - 7])/6.0;
          temp = (3*f[id_nodo + idx ] - 14*f[id_nodo + idx -1 ]  + 26*f[id_nodo + idx-2] - 24*f[id_nodo + idx - 3] + 11*f[id_nodo + idx - 4] - 2*f[id_nodo + idx - 5]);
          //temp = -(1069*3*f[id_nodo + idx ] - 1316*16*f[id_nodo + idx -1]  + 15289*4*f[id_nodo + idx-2] - 2144*48*f[id_nodo + idx - 3] + 10993*10*f[id_nodo + idx - 4] - 4772*16*f[id_nodo + idx - 5] + 2803*12*f[id_nodo + idx - 6] - 536*16*f[id_nodo + idx - 7] + 967*f[id_nodo + idx - 8])/240.0;

        
      }
      else if ( idx == Nr-2){
        //temp = (1*f[id_nodo + idx + 1] - 2*f[id_nodo + idx] + 1*f[id_nodo + idx - 1]);
        //temp = (1*f[id_nodo + idx + 1] - 4*f[id_nodo + idx ]  + 6*f[id_nodo + idx-1] - 4*f[id_nodo + idx - 2] + 1*f[id_nodo + idx - 3]);
        //temp = -(28*2*f[id_nodo + idx ] - 111*3*f[id_nodo + idx -1]  + 142*6*f[id_nodo + idx-2] - 1219*f[id_nodo + idx - 3] + 176*6*f[id_nodo + idx - 4] - 185*3*f[id_nodo + idx - 5]+82*2*f[id_nodo + idx - 6] -7*3*f[id_nodo + idx - 7])/6.0;
        temp = (3*f[id_nodo + idx ] - 14*f[id_nodo + idx -1 ]  + 26*f[id_nodo + idx-2] - 24*f[id_nodo + idx - 3] + 11*f[id_nodo + idx - 4] - 2*f[id_nodo + idx - 5]);
        //temp = -(1069*3*f[id_nodo + idx ] - 1316*16*f[id_nodo + idx -1]  + 15289*4*f[id_nodo + idx-2] - 2144*48*f[id_nodo + idx - 3] + 10993*10*f[id_nodo + idx - 4] - 4772*16*f[id_nodo + idx - 5] + 2803*12*f[id_nodo + idx - 6] - 536*16*f[id_nodo + idx - 7] + 967*f[id_nodo + idx - 8])/240.0;

    }
        
      else{
        for (int m=0;m<derivate_twelfth;m++){
          temp += diff_twelfth_order[m]*f[id_nodo + idx-half_derivate_twelfth_left+m];
        }
       temp=temp/1.0; 
      }
      temp = temp/pow(dr,1);
  
      return temp;
  }

//G representa la derivada temporal, es decir los K_i de runge kutta
__global__ void Kreiss_Oliger(double *G, double *R, double epsilon,int symmetric, int g, int planck,double *alpha){
    int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
    int id_nodo = threadIdx.x + blockDim.x*blockIdx.x;;
    int r = id_nodo%Nr;
    int k = (int)id_nodo/Nr%Nk;
    int l = (int)id_nodo/Nr/Nl;
    int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
      if(alpha[r]==0.0){
          G[id] = G[id];
      }
      else{
      if  (planck==0){
      if(r<(int)Nr){
          G[r] = G[r] - epsilon*pow(dr,0)*twelfth_derivate(R, 0, r,symmetric);
  
      }
      }
      else if (planck==1){
      if(id < 6*dim_q && r<(int)Nr){
          G[id] = G[id] - epsilon*pow(dr,0)*twelfth_derivate(R, id_nodo_ghost, r,symmetric);
      }
      }
      }
  }
  
  void llamada_kernel(double *G, double* R,  double epsilon,int symmetric,int g, int block, int thread, int planck, double *alpha){
  
    cudaDeviceSynchronize();
    Kreiss_Oliger<<<block,thread>>>(G,R,epsilon,symmetric,g,planck,alpha);
    cudaDeviceSynchronize();
  }
  void dissipacion( Relativity_G RG_K, field_Classic FC_K, field_Quantum FQ_K, Relativity_G y_tilde_M, field_Classic y_tilde_C, field_Quantum y_tilde_Q, int bloque_radial, int bloque_cuantico, int thread){
  
    llamada_kernel(RG_K.A       ,y_tilde_M.A       , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.B       ,y_tilde_M.B       , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.alpha   ,y_tilde_M.alpha   , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.Da      ,y_tilde_M.Da      , epsilon_metric_field, 1, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.Db      ,y_tilde_M.Db      , epsilon_metric_field, 1, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.K       ,y_tilde_M.K       , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.Kb      ,y_tilde_M.Kb      , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.lambda  ,y_tilde_M.lambda  , epsilon_metric_field, 1, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(RG_K.U       ,y_tilde_M.U       , epsilon_metric_field, 1, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
  
    llamada_kernel(FC_K.phi      ,y_tilde_C.phi      , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(FC_K.psi      ,y_tilde_C.psi      , epsilon_metric_field, 1, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
    llamada_kernel(FC_K.pi       ,y_tilde_C.pi       , epsilon_metric_field, 0, 0, bloque_radial,thread, 0,y_tilde_M.alpha);
  
    if(ct==1.0){
  
      for (int g =0 ; g<6 ;g++){
        llamada_kernel(FQ_K.real.u    ,y_tilde_Q.real.u     , epsilon_quantum_field, 0, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
        llamada_kernel(FQ_K.real.psi  ,y_tilde_Q.real.psi   , epsilon_quantum_field, 1, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
        llamada_kernel(FQ_K.real.pi   ,y_tilde_Q.real.pi    , epsilon_quantum_field, 0, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
  
        llamada_kernel(FQ_K.imag.u    ,y_tilde_Q.imag.u     , epsilon_quantum_field, 0, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
        llamada_kernel(FQ_K.imag.psi  ,y_tilde_Q.imag.psi   , epsilon_quantum_field, 1, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
        llamada_kernel(FQ_K.imag.pi   ,y_tilde_Q.imag.pi    , epsilon_quantum_field, 0, g, bloque_cuantico,thread, 1,y_tilde_M.alpha);
      }
    }
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
  
      
    }
  }
  __global__ void guardar_datos(double *f,double *pi,double *psi,double *alpha_temp, double *nodo, double *A_temp,double *B_temp, Relativity_G metrics ,field_Classic field, field_Quantum field_Q, int t){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
  
    int idf;
    double temp[2];
    double A_B_alpha_prime;
    double mass,f1,f2,f3,f4,radio;
      mass=0.0;
      double cte= 1.0/(4.0*consta_pi);
      double u_snake_x;
      double u_snake_y;
  
      double inte_phi;
      double xphi2x;
  
      //radio
      if (idx < Nr){
  
        xphi2x=0.0;
  
          if(idx==0){
          radio=0.0;
          }
          else{
          radio=idx*dr;
          }
          for(int g=0 ; g<6 ; g++){
            for(int l=0 ; l<Nl ; l++){
  
              inte_phi=0.0;
  
  
              for (int k=0 ; k < Nk ; k++){
                  // indice
                  idf = g*dim_q + l*Nk*Nr + k*Nr +  idx; 
                  
  
                  if(idx==0){
                    u_snake_x = field_Q.real.u[ idf +1 ] * pow(radio +dr, l) ;
                    u_snake_y = field_Q.imag.u[ idf +1 ] * pow(radio +dr, l) ;
                  }
                  else{
                    u_snake_x = field_Q.real.u[ idf ] * pow(radio , l) ;
                    u_snake_y = field_Q.imag.u[ idf ] * pow(radio , l) ;
                  }
  
                  inte_phi += ( pow(u_snake_x,2) + pow(u_snake_y,2) );
  
              }
  
              //fluctuacion phi
              
              xphi2x += inte_phi * (2*l + 1) * cte* dk;
          }
        }
        xphi2x   = field.phi[idx]*field.phi[idx] + xphi2x;
        }
        int id =  threadIdx.x + blockDim.x*blockIdx.x;
        int id_nodo = threadIdx.x + blockDim.x*blockIdx.x;;
        int r = id_nodo%Nr;
        int k = (int)id_nodo/Nr%Nk;
        int l = (int)id_nodo/Nr/Nl;
        int id_nodo_ghost =l*Nr*Nk + k*Nr;
    if(idx<Nr){
      f[ t*Nr + idx] = field.phi[idx];
      pi[ t*Nr + idx] = field.pi[idx];
      psi[ t*Nr + idx] = field.psi[idx];
  
      alpha_temp[ t*Nr + idx] = metrics.alpha[idx];
      A_temp[ t*Nr + idx] = metrics.A[idx];
      B_temp[ t*Nr + idx] = metrics.B[idx];
      nodo[t*Nr + idx]=xphi2x;//field_Q.imag.psi[5*dim_q + 2*Nr*Nk + 2*Nr + id];
      
  
    }
  }
  __global__ void Tensor_energy_momentum(double *rho, double *ja, double *SA, double *SB, Relativity_G metrics, field_Classic field_C, field_Quantum field_Q, int g, int siono){
      int idx = threadIdx.x + blockDim.x*blockIdx.x;
      int id;
      int cinetica;
      double radio,mass,c;
      //temp_fluct (5xNr)
      double cte= 1.0/(4.0*consta_pi);
      double u_snake_x;
      double u_snake_y;
      
      double d_u_temp_x;
      double d_u_temp_y;
      double d_u_radial_x;
      double d_u_radial_y;
      
      double inte_pi, inte_psi, inte_pi_psi, inte_theta,inte_phi;
      double xpi2x, xpsi2x, xpi_psix, xtheta2x,xphi2x;
      //radio
      if (idx < Nr){
  
          xpi2x=0.0;
          xpsi2x=0.0;
          xpi_psix=0.0;
          xtheta2x=0.0;    
          xphi2x=0.0;
  
          if(idx==0){
          radio=0.0;
          }
          else{
          radio=idx*dr;
          }
          if(ct==1.0){
  
            for(int l=0 ; l<Nl ; l++){
  
                inte_pi=0.0;
                inte_psi=0.0;
                inte_pi_psi=0.0;
                inte_theta=0.0;
                inte_phi=0.0;
                for (int k=0 ; k < Nk ; k++){
                    // indice
                    id = g*dim_q + l*Nk*Nr + k*Nr +  idx; 
                    if(g==0){
                      mass=0.0;
                      c=1.0;
                    }
                    else if(g==1 || g==3){
                        mass=1.0;
                        c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
                    }
                    else if(g==2 || g==4){
                      mass=1.0 * sqrt(3.0);
                      c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
                    }
                    else if( g==5){
                      mass=1.0 *sqrt(4.0);
                      c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
                    }
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
                    inte_phi += mass*mass*c*c*( pow(u_snake_x,2) + pow(u_snake_y,2) );
  
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
                //xphi2x += inte_phi * (2*l + 1) * cte* dk;
            }
          }
          if(g==0){
              cinetica=1;
              xpi2x   = 1.0*field_C.pi[idx]*field_C.pi[idx] + xpi2x;
              xpsi2x  = 1.0*field_C.psi[idx]*field_C.psi[idx] + xpsi2x;
              xpi_psix = 1.0*field_C.pi[idx]*field_C.psi[idx] + xpi_psix;
              //xphi2x =1.0*field_C.phi[idx]*field_C.phi[idx] + xphi2x;
            }
          else if(g==1 || g==3){
              cinetica=-1;
          }
          else if(g==2 || g==4){
            cinetica=1;
          }
          else if( g==5){
            cinetica=-1;
          }
          if(idx==0){
            radio=dr;
            }
            else{
            radio=idx*dr;
            }
          rho[idx] +=     cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) + 1.0/(metrics.B[idx]*radio*radio)*xtheta2x - 0.5*xphi2x) ;
    
          ja[idx] += -    cinetica*xpi_psix/(sqrt(metrics.A[idx])*metrics.B[idx]);
      
          SA[idx] +=      cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) - 1.0/(metrics.B[idx]*radio*radio)*xtheta2x - 0.5*xphi2x) ;
      
          SB[idx] +=      cinetica*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) - xpsi2x) -  0.5*xphi2x) ;
          
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
  
  
        if(id<Nr){
          metrics_RK.alpha[id] = -2.0*metrics.alpha[id]*metrics.K[id];
          metrics_RK.A[id] = -2.0*metrics.alpha[id]*metrics.A[id]*(metrics.K[id] - 2.0*metrics.Kb[id]);
          metrics_RK.B[id] = -2.0*metrics.alpha[id]*metrics.B[id]*metrics.Kb[id];
        }
        else{
          metrics_RK.alpha[id]  = -derivate(metrics.alpha, 0, id, 0);
          metrics_RK.B[id]      = -derivate(metrics.B, 0, id, 0);
          metrics_RK.A[id]      = -derivate(metrics.A, 0, id, 0);
        }
        //                                                ********metrica_prima********
        if(id<Nr-1){
          metrics_RK.Da[id] = -2.0*derivate(metrics.K, 0, id, 0);
          metrics_RK.Db[id] = -2.0*(derivate(metrics.alpha, 0, id, 0)*metrics.Kb[id] + metrics.alpha[id]*derivate(metrics.Kb,0, id,0) ); 
        }
        else{
          metrics_RK.Da[id]      = -derivate(metrics.Da, 0, id, 1) - metrics.Da[id]/radio;
          metrics_RK.Db[id]      = -derivate(metrics.Db, 0, id, 1) - metrics.Db[id]/radio;
        }
        //                                                ********Curvatura extrinseca********
        //Determinamos un valor para una expresion que se repite bastante
        double temp;
        temp = (metrics.U[id] + 4.0*metrics.lambda[id]*metrics.B[id]/metrics.A[id] );
  
        //K
        f1 =(metrics.K[id]*(metrics.K[id] - 4.0*metrics.Kb[id]) + 6.0*pow(metrics.Kb[id],2));
        f3= (rho[id] + SA[id] + 2.0*SB[id] - 2.0*Cosm[0]);
        if(id==0){
            f2=(3.0*derivate(metrics.Da,0,id,1) + metrics.Da[id]*(metrics.Da[id] - 0.5* temp) );
        }
        else{
            f2=(derivate(metrics.Da,0,id,1) +  2.0*metrics.Da[id]/radio + metrics.Da[id]*(metrics.Da[id] - 0.5* temp) );
        }
        if(id<Nr-1){
          metrics_RK.K[id] =  metrics.alpha[id]* f1 - metrics.alpha[id]/metrics.A[id] * f2 + 0.5*metrics.alpha[id] * f3;
        }
        else{
          metrics_RK.K[id] =-derivate(metrics.K,0,id,0) - metrics.K[id]/radio;
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
        if(id<Nr-1){
          metrics_RK.Kb[id] = metrics.alpha[id]/metrics.A[id] * (  f1 + f2  )+ 0.5*metrics.alpha[id] * f3;
        }
        else{
          metrics_RK.Kb[id] =-derivate(metrics.Kb,0,id,0) - metrics.Kb[id]/radio;
        }
  
        //                                                ********Regularacion********
        if(id<Nr-1){
          metrics_RK.U[id] = -2.0*metrics.alpha[id] * (derivate( metrics.K,0,id,0) + metrics.Da[id]* ( metrics.K[id] - 4.0*metrics.Kb[id] ) 
                                                          - 2.0* ( metrics.K[id] - 3.0*metrics.Kb[id] ) * ( metrics.Db[id] - 2.0*metrics.lambda[id]*metrics.B[id]/metrics.A[id] ))
                              -4.0*metrics.alpha[id] * ja[id];
  
          temp = 2.0*metrics.alpha[id]*metrics.A[id]/metrics.B[id];
          metrics_RK.lambda[id] = temp * ( derivate(metrics.Kb,0, id, 0) - 0.5*metrics.Db[id] * ( metrics.K[id] - 3.0*metrics.Kb[id] ) + 0.5*ja[id] );
        }
        else{
          metrics_RK.U[id]      =-derivate(metrics.U,0,id,1) - metrics.U[id]/radio;
          metrics_RK.lambda[id] =-derivate(metrics.lambda,0,id,1) - metrics.lambda[id]/radio;
        }
  
      }
  }
  __global__ void evo_fields_classics(field_Classic field_C, field_Classic field_C_RK, Relativity_G metrics){
    int id = threadIdx.x + blockDim.x*blockIdx.x;;
  
  double radio,temp;
  double c_efectivo;
  if(id < Nr){
    c_efectivo=1.0;//metrics.alpha[Nr-1]/sqrt(metrics.A[Nr-1]);
  
    if(id==0){radio=dr;}
    else{radio=id*dr;}
    
    temp = metrics.alpha[id]/(sqrt(metrics.A[id])*metrics.B[id]);
    if(id<Nr-1){
  
  
    field_C_RK.phi[id] = temp*field_C.pi[id];
    field_C_RK.psi[id] = psi_dot(metrics,field_C.pi,0,id);
      if(id==0){
          field_C_RK.pi[id] =  3.0*f_pi_dot(field_C,metrics,id) ;  
      }
      else{
          field_C_RK.pi[id] =  f_pi_dot(field_C,metrics,id) + 2.0*metrics.alpha[id]*metrics.B[id] * field_C.psi[id]/sqrt(metrics.A[id])/radio;  
      }
    }
      else{
        field_C_RK.phi[id]  =   temp*field_C.pi[id];
        field_C_RK.psi[id]  = - c_efectivo*derivate(field_C.psi,0,id,0) -  field_C.psi[id]/radio;//psi_dot(metrics,field_C.pi,0,id);
        field_C_RK.pi[id]   = - c_efectivo*derivate(field_C.pi,0,id,0)  -  field_C.pi[id]/radio;// f_pi_dot(field_C,metrics,id) + 2.0*metrics.alpha[id]*metrics.B[id] * field_C.psi[id]/sqrt(metrics.A[id])/radio;  
  
      }
  }
  }
  __global__ void evo_fields_quantums(  field_Quantum field_Q, field_Quantum field_Q_RK, Relativity_G metrics,  int g, int s){
      int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
      int id_nodo = threadIdx.x + blockDim.x*blockIdx.x;;
      int r = (int)id_nodo%Nr;
      int k = (int)id_nodo/Nr%Nk;
      int l = (int)id_nodo/Nr/Nl;
      int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
      double temp[2];
      double A_B_alpha_prime;
      double mass,c,radio;
      if(g==0){
        mass=0.0;
        c=1.0;
      }
      else if(g==1 || g==3){
        mass=1.0;
        c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
  
      }
      else if(g==2 || g==4){
        mass=1*sqrt(3.0);
        c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
  
      }
      else if(g==5){
        mass=1*sqrt(4.0);
        c=(dk*(k+1))/sqrt(dk*(k+1)*dk*(k+1)+mass*mass);
  
      }
      if(id < 6*dim_q){
        c=1.0;

        if (r==0){
          radio = 0.0;
      }
      else{
          radio = dr*r;
      }
        if(r<Nr){
          field_Q_RK.real.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.real.pi[id];
          field_Q_RK.imag.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.imag.pi[id];
        }
        else{
          field_Q_RK.real.u[id] = -c*field_Q.real.psi[id];
          field_Q_RK.imag.u[id] = -c*field_Q.imag.psi[id];
        }
  
        if(r==0){
          temp[0] = 0.0;
          temp[1] = 0.0;
        }
        else if(r==Nr-1){
          temp[0]=-c*derivate(field_Q.real.psi,id_nodo_ghost,r,1) - field_Q.real.psi[id]/radio;
          temp[0]=-c*derivate(field_Q.imag.psi,id_nodo_ghost,r,1) - field_Q.imag.psi[id]/radio;
  
        }
        else{
          temp[0] = psi_dot(metrics, field_Q.real.pi, id_nodo_ghost ,r);
          temp[1] = psi_dot(metrics, field_Q.imag.pi, id_nodo_ghost ,r);
        }
  
  
        field_Q_RK.real.psi[id] = temp[0];
        field_Q_RK.imag.psi[id] = temp[1];
  
        double f1, f2, f3, f4;
  
        A_B_alpha_prime = derivate_metric(metrics, r);
        //pi real
        if(r==0){
            f1 = 0.0; // cero debido a A_B_alpha_prime
            f2 = (2*l+3) * derivate(field_Q.real.psi,id_nodo_ghost,r,1);
            f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.real.u[id];
        }
        else{
            f1 = l/radio * (field_Q.real.u[id]) + (field_Q.real.psi[id]);
            f2 = (2*l+2)/radio * field_Q.real.psi[id] + derivate(field_Q.real.psi,id_nodo_ghost,r,1);
            f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.real.u[id];
        }
        f4 = mass * mass* field_Q.real.u[id];//* field_Q.real.u[id];
        
  
        //Nota : averiguar si f3 tiene un menos o no.
        if(r<Nr-1){
          field_Q_RK.real.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 + f3) - metrics.alpha[r]*metrics.B[r]*(sqrt(metrics.A[r]))*f4;
          //field_Q_RK.real.pi[id] = field_Q_RK.real.pi[id]*c;
  
        }
        else if(r==Nr-1){
          field_Q_RK.real.pi[id] = -c*derivate(field_Q.real.pi,id_nodo_ghost,r,0) - field_Q.real.pi[id]/radio;
        }
        //pi imaginario
        if(r==0){
            f1 = 0.0; // cero
            f2 = (2*l+3) *  derivate(field_Q.imag.psi,id_nodo_ghost,r,1);
            f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.imag.u[id];
        }
        else{
  
            f1 = l/radio * (field_Q.imag.u[id]) + (field_Q.imag.psi[id]);
            f2 = (2*l+2)/radio * field_Q.imag.psi[id] + derivate(field_Q.imag.psi,id_nodo_ghost,r,1);
            f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.imag.u[id];
        }
        f4 = mass * mass* field_Q.imag.u[id];//* field_Q.imag.u[id];
  
        if(r<Nr-1){
          field_Q_RK.imag.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 + f3) - metrics.alpha[r]*metrics.B[r]*(sqrt(metrics.A[r]))*f4;
          //field_Q_RK.imag.pi[id] = field_Q_RK.imag.pi[id]*c;
        }
        else if(r==Nr-1){
          field_Q_RK.imag.pi[id] = -c*derivate(field_Q.imag.pi,id_nodo_ghost,r,0) - field_Q.imag.pi[id]/radio;
        }      
        
  
    }
  }
  __global__ void y_tilde_metrics( Relativity_G metrics , Relativity_G RG_K1, Relativity_G RG_K2,  Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5, Relativity_G y_tilde_M, int s){
      int id = threadIdx.x + blockDim.x*blockIdx.x;
  
      if(id < Nr){
        y_tilde_M.A[id]      = metrics.A[id]        + dt*( a_ij[s][0]*RG_K1.A[id]      + a_ij[s][1]*RG_K2.A[id]      + a_ij[s][2]*RG_K3.A[id]      + a_ij[s][3]*RG_K4.A[id]      + a_ij[s][4]*RG_K5.A[id] );
        y_tilde_M.B[id]      = metrics.B[id]        + dt*( a_ij[s][0]*RG_K1.B[id]      + a_ij[s][1]*RG_K2.B[id]      + a_ij[s][2]*RG_K3.B[id]      + a_ij[s][3]*RG_K4.B[id]      + a_ij[s][4]*RG_K5.B[id] );
        y_tilde_M.alpha[id]  = metrics.alpha[id]    + dt*( a_ij[s][0]*RG_K1.alpha[id]  + a_ij[s][1]*RG_K2.alpha[id]  + a_ij[s][2]*RG_K3.alpha[id]  + a_ij[s][3]*RG_K4.alpha[id]  + a_ij[s][4]*RG_K5.alpha[id] );
        if(y_tilde_M.alpha[id] < 0.00000001 ){
          y_tilde_M.alpha[id] = 0.0;
        }
        if(y_tilde_M.alpha[id] >1.0 ){
          y_tilde_M.alpha[id] = 1.0;
        }
        if(y_tilde_M.A[id] < 0.0 ){
          y_tilde_M.alpha[id] = 0.0;
        }
        y_tilde_M.Da[id]     = metrics.Da[id]       + dt*( a_ij[s][0]*RG_K1.Da[id]     + a_ij[s][1]*RG_K2.Da[id]     + a_ij[s][2]*RG_K3.Da[id]     + a_ij[s][3]*RG_K4.Da[id]     + a_ij[s][4]*RG_K5.Da[id] );
        y_tilde_M.Db[id]     = metrics.Db[id]       + dt*( a_ij[s][0]*RG_K1.Db[id]     + a_ij[s][1]*RG_K2.Db[id]     + a_ij[s][2]*RG_K3.Db[id]     + a_ij[s][3]*RG_K4.Db[id]     + a_ij[s][4]*RG_K5.Db[id] );
        y_tilde_M.Kb[id]     = metrics.Kb[id]       + dt*( a_ij[s][0]*RG_K1.Kb[id]     + a_ij[s][1]*RG_K2.Kb[id]     + a_ij[s][2]*RG_K3.Kb[id]     + a_ij[s][3]*RG_K4.Kb[id]     + a_ij[s][4]*RG_K5.Kb[id] );
        y_tilde_M.K[id]      = metrics.K[id]        + dt*( a_ij[s][0]*RG_K1.K[id]      + a_ij[s][1]*RG_K2.K[id]      + a_ij[s][2]*RG_K3.K[id]      + a_ij[s][3]*RG_K4.K[id]      + a_ij[s][4]*RG_K5.K[id] );
        y_tilde_M.lambda[id] = metrics.lambda[id]   + dt*( a_ij[s][0]*RG_K1.lambda[id] + a_ij[s][1]*RG_K2.lambda[id] + a_ij[s][2]*RG_K3.lambda[id] + a_ij[s][3]*RG_K4.lambda[id] + a_ij[s][4]*RG_K5.lambda[id] );
        y_tilde_M.U[id]      = metrics.U[id]        + dt*( a_ij[s][0]*RG_K1.U[id]      + a_ij[s][1]*RG_K2.U[id]      + a_ij[s][2]*RG_K3.U[id]      + a_ij[s][3]*RG_K4.U[id]      + a_ij[s][4]*RG_K5.U[id] );
  
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
      int id_nodo= threadIdx.x + blockDim.x*blockIdx.x;
      int r = (int)id_nodo%Nr;
      int k = (int)id_nodo/Nr%Nk;
      int l = (int)id_nodo/Nr/Nl;
      int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
      double c;
  
      if(id < 6*dim_q){
        if(r<Nr){
          y_tilde_Q.real.u[id] = field_Q.real.u[id]     + dt*( a_ij[s][0]* FQ_K1.real.u[id]     + a_ij[s][1]* FQ_K2.real.u[id]    + a_ij[s][2]* FQ_K3.real.u[id]    + a_ij[s][3]* FQ_K4.real.u[id]    + a_ij[s][4]* FQ_K5.real.u[id]);
          y_tilde_Q.real.pi[id]  = field_Q.real.pi[id]  + dt*( a_ij[s][0]* FQ_K1.real.pi[id]    + a_ij[s][1]* FQ_K2.real.pi[id]   + a_ij[s][2]* FQ_K3.real.pi[id]   + a_ij[s][3]* FQ_K4.real.pi[id]   + a_ij[s][4]* FQ_K5.real.pi[id]);
          y_tilde_Q.real.psi[id] = field_Q.real.psi[id] + dt*( a_ij[s][0]* FQ_K1.real.psi[id]   + a_ij[s][1]* FQ_K2.real.psi[id]  + a_ij[s][2]* FQ_K3.real.psi[id]  + a_ij[s][3]* FQ_K4.real.psi[id]  + a_ij[s][4]* FQ_K5.real.psi[id]);
    
          y_tilde_Q.imag.u[id] = field_Q.imag.u[id]     + dt*( a_ij[s][0]* FQ_K1.imag.u[id]     + a_ij[s][1]* FQ_K2.imag.u[id]    + a_ij[s][2]* FQ_K3.imag.u[id]    + a_ij[s][3]* FQ_K4.imag.u[id]    + a_ij[s][4]* FQ_K5.imag.u[id]);
          y_tilde_Q.imag.pi[id]  = field_Q.imag.pi[id]  + dt*( a_ij[s][0]* FQ_K1.imag.pi[id]    + a_ij[s][1]* FQ_K2.imag.pi[id]   + a_ij[s][2]* FQ_K3.imag.pi[id]   + a_ij[s][3]* FQ_K4.imag.pi[id]   + a_ij[s][4]* FQ_K5.imag.pi[id]);
          y_tilde_Q.imag.psi[id] = field_Q.imag.psi[id] + dt*( a_ij[s][0]* FQ_K1.imag.psi[id]   + a_ij[s][1]* FQ_K2.imag.psi[id]  + a_ij[s][2]* FQ_K3.imag.psi[id]  + a_ij[s][3]* FQ_K4.imag.psi[id]  + a_ij[s][4]* FQ_K5.imag.psi[id]);
  
        }
  
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
        //dim3  bloque(thread);
        //dim3  grid_radial ((int)ceil((float)Nr/thread));
        //dim3  grid_quantum ((int)ceil((float)dim_q/thread));
        int bloque =thread;
        int grid_radial =(int)ceil((float)Nr/thread);
        int grid_quantum=  ((int)ceil((float)dim_q/thread));
      //ciclo for de la cantidad de pasos
      for(int s=0 ; s<step  ; s++){
  
          cudaDeviceSynchronize();
          //Calculo de y_(n+1)
          y_tilde_metrics         <<< grid_radial, bloque >>>(metrics,RG_K1,RG_K2,RG_K3,RG_K4,RG_K5, y_tilde_M, s);
          y_tilde_field_classic   <<< grid_radial, bloque >>>(field_C,FC_K1,FC_K2,FC_K3,FC_K4,FC_K5, y_tilde_C, s);
          if(ct==1.0){
            for (int g=0; g < 6; g++){
                y_tilde_field_quantum   <<< grid_quantum, bloque >>>(field_Q,FQ_K1,FQ_K2,FQ_K3,FQ_K4,FQ_K5, y_tilde_Q, s, g);
            }
        }
          cudaDeviceSynchronize();
  
          //Calculo del tensor energia momentum en el tiempo n+1 
          T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
          if(ct==1.0){
              for (int g=0; g < 6; g++){
                  Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, y_tilde_M, y_tilde_C, y_tilde_Q, g,1);
              } 
          }
          else{
              Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, y_tilde_M, y_tilde_C, y_tilde_Q, 0,1);
          }
     
          cudaDeviceSynchronize();
  
          //Actualizacion de los K
          if(s==0){
              evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K1, rho, ja, SA, SB, Cosm);
              cudaDeviceSynchronize();
              evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K1,  y_tilde_M);
  
              if(ct==1.0){
                for (int g=0; g < 6; g++){
                    evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K1,  y_tilde_M, g,s);
                }
              }
  
              cudaDeviceSynchronize();
  
              dissipacion(RG_K1,FC_K1,FQ_K1, y_tilde_M,y_tilde_C,y_tilde_Q,grid_radial,grid_quantum,bloque);
          }
          else if(s==1){
              evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K2, rho, ja, SA, SB, Cosm);
              cudaDeviceSynchronize();
              evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K2,  y_tilde_M);
  
              if(ct==1.0){
                for (int g=0; g < 6; g++){
                    evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K2,  y_tilde_M, g,s);
                }
              }
  
              cudaDeviceSynchronize();
  
              //dissipacion(RG_K2,FC_K2,FQ_K2, y_tilde_M,y_tilde_C,y_tilde_Q,grid_radial,grid_quantum,bloque);
  
          }
          else if(s==2){
              evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K3, rho, ja, SA, SB, Cosm);
              cudaDeviceSynchronize();
              evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K3,  y_tilde_M);
              if(ct==1.0){
                for (int g=0; g < 6; g++){
                    evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K3,  y_tilde_M, g,s);
                }
              }
              cudaDeviceSynchronize();
  
              dissipacion(RG_K3,FC_K3,FQ_K3, y_tilde_M,y_tilde_C,y_tilde_Q,grid_radial,grid_quantum,bloque);
  
          }
          else if(s==3){
              evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K4, rho, ja, SA, SB, Cosm);
              cudaDeviceSynchronize();
              evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K4,  y_tilde_M);
              if(ct==1.0){
                for (int g=0; g < 6; g++){
                    evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K4,  y_tilde_M, g,s);
                }
              }
              cudaDeviceSynchronize();
  
              //dissipacion(RG_K4,FC_K4,FQ_K4, y_tilde_M,y_tilde_C,y_tilde_Q,grid_radial,grid_quantum,bloque);
  
          }
          else if(s==4){
              evo_metrics         <<< grid_radial, bloque >>>( y_tilde_M, RG_K5, rho, ja, SA, SB, Cosm);
              cudaDeviceSynchronize();
              evo_fields_classics <<< grid_radial, bloque >>>( y_tilde_C,  FC_K5,  y_tilde_M);
              if(ct==1.0){
                for (int g=0; g < 6; g++){
                    evo_fields_quantums <<< grid_quantum, bloque >>>( y_tilde_Q,  FQ_K5,  y_tilde_M, g,s);
                }
              }
              cudaDeviceSynchronize();
  
              dissipacion(RG_K5,FC_K5,FQ_K5, y_tilde_M,y_tilde_C,y_tilde_Q,grid_radial,grid_quantum,bloque);
  
          }
      }
  }
  
  __global__ void next_step_metric(   Relativity_G metrics,  Relativity_G RG_K1,Relativity_G RG_K2, Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5){
    int id =  threadIdx.x + blockDim.x*blockIdx.x;
    if(id<Nr){
  
      metrics.A[id]      = metrics.A[id]        + dt*( b_i[0]*RG_K1.A[id]      + b_i[1]*RG_K2.A[id]      + b_i[2]*RG_K3.A[id]      + b_i[3]*RG_K4.A[id]      + b_i[4]*RG_K5.A[id] );
      metrics.B[id]      = metrics.B[id]        + dt*( b_i[0]*RG_K1.B[id]      + b_i[1]*RG_K2.B[id]      + b_i[2]*RG_K3.B[id]      + b_i[3]*RG_K4.B[id]      + b_i[4]*RG_K5.B[id] );
      metrics.alpha[id]  = metrics.alpha[id]    + dt*( b_i[0]*RG_K1.alpha[id]  + b_i[1]*RG_K2.alpha[id]  + b_i[2]*RG_K3.alpha[id]  + b_i[3]*RG_K4.alpha[id]  + b_i[4]*RG_K5.alpha[id] );
  
      if(metrics.alpha[id] < 0.00000001     ){
        metrics.alpha[id] = 0.0;
      }
  
      if(metrics.alpha[id] >1.0 ){
        metrics.alpha[id] = 1.0;
      }
      if(metrics.A[id] < 0.0 ){
        metrics.alpha[id] = 0.0;
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
    int id_nodo= threadIdx.x + blockDim.x*blockIdx.x;
    int r = (int)id_nodo%Nr;
    int k = (int)id_nodo/Nr%Nk;
    int l = (int)id_nodo/Nr/Nl;
    int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
    double c;
    if(id < 6*dim_q){
      if(r<Nr ){
  
      field_Q.real.u[id]   = field_Q.real.u[id]   + dt*( b_i[0]* FQ_K1.real.u[id]     + b_i[1]* FQ_K2.real.u[id]    + b_i[2]* FQ_K3.real.u[id]    + b_i[3]* FQ_K4.real.u[id]    + b_i[4]* FQ_K5.real.u[id]);
      field_Q.real.pi[id]  = field_Q.real.pi[id]  + dt*( b_i[0]* FQ_K1.real.pi[id]    + b_i[1]* FQ_K2.real.pi[id]   + b_i[2]* FQ_K3.real.pi[id]   + b_i[3]* FQ_K4.real.pi[id]   + b_i[4]* FQ_K5.real.pi[id]);
      field_Q.real.psi[id] = field_Q.real.psi[id] + dt*( b_i[0]* FQ_K1.real.psi[id]   + b_i[1]* FQ_K2.real.psi[id]  + b_i[2]* FQ_K3.real.psi[id]  + b_i[3]* FQ_K4.real.psi[id]  + b_i[4]* FQ_K5.real.psi[id]);
  
      field_Q.imag.u[id]   = field_Q.imag.u[id]   + dt*( b_i[0]* FQ_K1.imag.u[id]     + b_i[1]* FQ_K2.imag.u[id]    + b_i[2]* FQ_K3.imag.u[id]    + b_i[3]* FQ_K4.imag.u[id]    + b_i[4]* FQ_K5.imag.u[id]);
      field_Q.imag.pi[id]  = field_Q.imag.pi[id]  + dt*( b_i[0]* FQ_K1.imag.pi[id]    + b_i[1]* FQ_K2.imag.pi[id]   + b_i[2]* FQ_K3.imag.pi[id]   + b_i[3]* FQ_K4.imag.pi[id]   + b_i[4]* FQ_K5.imag.pi[id]);
      field_Q.imag.psi[id] = field_Q.imag.psi[id] + dt*( b_i[0]* FQ_K1.imag.psi[id]   + b_i[1]* FQ_K2.imag.psi[id]  + b_i[2]* FQ_K3.imag.psi[id]  + b_i[3]* FQ_K4.imag.psi[id]  + b_i[4]* FQ_K5.imag.psi[id]);
  
    }
  
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
  
  __global__ void calc_geodesic(double *Geodesic, Relativity_G metrics, int t){
    int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double radio;
    if(idx<Nr){
      if(idx==0){
        radio = dr;
      }
      else{
        radio = idx*dr;
      }
      Geodesic[ t*Nr + idx ] = (2.0/radio + metrics.Db[idx])/sqrt(metrics.A[idx]) - 2.0*metrics.Kb[idx];
  
    }
  }
  
  __global__ void Hamilton(double *H, double *rho, Relativity_G metrics, double* cosm, int t){
    int idx =threadIdx.x + blockDim.x*blockIdx.x;
    double radio,f1,f2,f3;
    if(idx<Nr){
      if(idx==0){
        radio = dr;
      }
      else{
        radio = idx*dr;
      }
      if(idx==0){
        f1=(derivate(metrics.lambda,0,idx,1) +derivate(metrics.Db,0,idx,1) - derivate(metrics.U,0,idx,1) - 4.0*derivate(metrics.lambda,0,idx,1)*metrics.B[idx]/metrics.A[idx]);
      }
      else{
        f1= 1/radio * (metrics.lambda[idx] + metrics.Db[idx] -metrics.U[idx]- 4.0*metrics.lambda[idx]*metrics.B[idx]/metrics.A[idx]);
      }
      f2=(0.25*metrics.Db[idx] + 0.5*metrics.U[idx] + 2.0*metrics.lambda[idx]*metrics.B[idx]/metrics.A[idx]);
      f3=(metrics.A[idx]*metrics.Kb[idx]* ( 2.0*metrics.K[idx]-3.0*metrics.Kb[idx]));
  
      H[ t*Nr + idx ] = derivate(metrics.Db,0,idx,1) + f1 - metrics.Db[idx]*f2 - f3 + metrics.A[idx] *(rho[idx] + cosm[0]) ;
  
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
    return amplitude*exp(-pow(radio/width,2));
    //return amplitude;
  }
  
  __device__ double initial_psi(double radio){
    return - 2.0*radio/pow(width,2) *amplitude*exp(- pow(radio/width,2));
    //return 0.0;
  
  }
  __global__ void field_initial(field_Classic field_C){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double r;
    if(idx<Nr){
      r = idx*dr;
  
      //Initial matters fields
      field_C.phi[idx] = initial_phi(r-r0);
  
      field_C.psi[idx] = initial_psi(r-r0);
      
    }
  }
  __global__ void metric_initial( Relativity_G metrics, double *cosmological_constant){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    double r;
    if(idx<Nr){
      r = idx*dr;
  
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
    double sol;
    double rs;
  
    if(r==0){
      sol = 0.0;
    }
    else{
      sol = A*( (1.0-A)/r + r*pow(initial_psi(r-r0),2)*0.5 + ct*r*A*cosmological_constant);
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
  
      if(idx==0){
        metrics.U[idx]=0.0;
      }
    }
  }
  //C.I
  void quantum_initial(field_Quantum field_Q, int k, int l, int space, double mass){
      int id_u= space + l*Nk*Nr + k*Nr ;
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
                  field_Q.real.u[id_u + r] = ct*(dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1) ;
                  field_Q.imag.u[id_u + r] = 0.0;
                  
                  field_Q.real.pi[id_u + r] = 0.0 ;
                  field_Q.imag.pi[id_u + r] = -ct*omega*(dk*(k+1)/sqrt(consta_pi * omega)) *pow(dk*(k+1),l)/gsl_sf_doublefact(2*l+1);
                  
                  field_Q.real.psi[id_u + r] = ct*(pow(dk*(k+1),l+3)/sqrt(consta_pi * omega)) * (-1)*(dr*r) /gsl_sf_doublefact(2*l+3);
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
                  field_Q.real.u[id_u + r] = ct*(dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l) ;
                  field_Q.imag.u[id_u + r] = 0.0;
                  
                  field_Q.real.pi[id_u + r] = 0.0 ;
                  field_Q.imag.pi[id_u + r] = -ct*omega*(dk*(k+1)/sqrt(consta_pi * omega)) * gsl_sf_bessel_jl(l,dk*(k+1)*dr*r) / pow(dr*r,l);
                  
                  field_Q.real.psi[id_u + r] = ct*( pow(dk*(k+1),2) /sqrt(consta_pi * omega)) * (-1) * gsl_sf_bessel_jl( l +1, dk*(k+1)*dr*(r)  )/ (pow(dr*r,l));
                  field_Q.imag.psi[id_u + r] = 0.0;
          }
      }
  }
  __global__ void modificar_pi(field_Quantum field_Q ,  double *A,int g){
    int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
    int id_nodo = threadIdx.x + blockDim.x*blockIdx.x;;
    int r = (int)id_nodo%Nr;
    int k = (int)id_nodo/Nr%Nk;
    int l = (int)id_nodo/Nr/Nl;
    int id_nodo_ghost =g*dim_q +l*Nr*Nk + k*Nr;
  
      field_Q.imag.pi[id_nodo_ghost + r] = sqrt(A[r])*field_Q.imag.pi[id_nodo_ghost + r];
    
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
      metric_initial<<<grid_radial,bloque>>>(   metrics, cosmological_constant);
      double mass;
      //A_RK<<<1,1>>>( metrics, cosmological_constant);
  
      cudaDeviceSynchronize();
      //Campo cuantico
      if(ct==1.0){
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
    
      }
  
      cudaDeviceSynchronize();
  
      T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
      cudaDeviceSynchronize();
  
      if(ct==1.0){
        for (int g=0; g < 6; g++){
            Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, g,1);
        }
      }
      else{
          Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, 0,1);
      }
  
        cost_cosm_value<<<1,1>>>(rho,cosmological_constant);
      cudaDeviceSynchronize();
      field_initial<<< grid_radial, bloque >>>(field_C);
      cudaDeviceSynchronize();
  
        A_RK<<<1,1>>>( metrics, cosmological_constant);
  
      cudaDeviceSynchronize();
      if(ct==1.0){
        for (int g =0; g<6 ; g++){
          modificar_pi<<< grid_quantum, bloque >>>(field_Q,metrics.A,g);
        }
      }
  
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
      double *cuda_xphix,*cuda_xpix,*cuda_xpsix;
      double *cuda_alpha;
      double *cuda_A,*cuda_B;
      double *cuda_nodo;
      double *cuda_H_constrain;
      double *cuda_geodesic;
  
      cudaMalloc((void **)&(cuda_T_tt), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_T_tr), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_T_rr), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_T_00), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_xphix), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_xpix), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_xpsix), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_alpha), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_A), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_B), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_nodo), Nt*Nr*sizeof(double)); 
      cudaMalloc((void **)&(cuda_H_constrain), Nt*Nr*sizeof(double)); 
  
      cudaMalloc((void **)&(cuda_geodesic), Nt*Nr*sizeof(double)); 
  
      //double *Nodos;
  
      //Array dissipacion
      double *R_temp;
      cudaMalloc((void **)&(R_temp), Nr*sizeof(double)); 
      double *R_temp_q;
      cudaMalloc((void **)&(R_temp_q), dim_q*sizeof(double)); 
  
      for (int t=0 ; t<Nt ;t++){
        //printf("t : %d\n",t);
          Tensor_tt<<< grid_radial, bloque >>>(cuda_T_tt, cuda_T_rr, cuda_T_tr, cuda_T_00, rho, SA, ja, SB, t);
          cudaDeviceSynchronize();
          guardar_datos<<< grid_radial, bloque >>>(cuda_xphix,cuda_xpix,cuda_xpsix,cuda_alpha,cuda_nodo,cuda_A,cuda_B,metrics, field_C,field_Q , t);
          cudaDeviceSynchronize();
          calc_geodesic<<< grid_radial, bloque >>>(cuda_geodesic, metrics,t);
          cudaDeviceSynchronize();
          T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
          if(ct==1.0){
              for (int g=0; g < 6; g++){
                  Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, g,0);
              }
          }
          else{
              Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, metrics, field_C, field_Q, 0,0);
          }
          cudaDeviceSynchronize();
  
          zeros_RK<<< grid_radial, bloque >>>( RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5, FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5, FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5);
          cudaDeviceSynchronize();
  
          for(int i=0; i< 10;i++){
            RK_implicit_tenth(metrics,  RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5,  y_tilde_M,
                              field_C,  FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5,  y_tilde_C,
                              field_Q,  FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5,  y_tilde_Q,
                              rho, ja, SA, SB,  cosmological_constant );
          }
          cudaDeviceSynchronize();
  
          next_step_metric<<< grid_radial, bloque >>>(metrics, RG_K1, RG_K2,  RG_K3,  RG_K4,  RG_K5);
          next_step_classic<<< grid_radial, bloque >>>(field_C,  FC_K1,  FC_K2,  FC_K3,  FC_K4,  FC_K5);
          if(ct==1.0){
              for (int g=0; g < 6; g++){
              next_step_quantum<<< grid_quantum, bloque >>>(field_Q,  FQ_K1,  FQ_K2,  FQ_K3,  FQ_K4,  FQ_K5, g);
              }
          }
          cudaDeviceSynchronize();
          Hamilton<<< grid_radial, bloque >>>(cuda_H_constrain, rho,metrics,cosmological_constant,t);
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
  
        double *xphix,*xpsix,*xpix,*alpha,*A, *B,*nodo;
        xphix =(double*)malloc(Nt*bytes_Nr);
        xpix =(double*)malloc(Nt*bytes_Nr);
        xpsix =(double*)malloc(Nt*bytes_Nr);
        alpha =(double*)malloc(Nt*bytes_Nr);
        A =(double*)malloc(Nt*bytes_Nr);
        B =(double*)malloc(Nt*bytes_Nr);
        nodo =(double*)malloc(Nt*bytes_Nr);
  
        cudaMemcpy(xphix, cuda_xphix, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(xpsix, cuda_xpsix, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(xpix, cuda_xpix, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(alpha, cuda_alpha, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(A, cuda_A, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(B, cuda_B, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(nodo, cuda_nodo, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
  
        double *geo,*H_cosntrain;
        geo =(double*)malloc(Nt*bytes_Nr);
        H_cosntrain =(double*)malloc(Nt*bytes_Nr);
  
        cudaMemcpy(geo, cuda_geodesic, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
        cudaMemcpy(H_cosntrain, cuda_H_constrain, Nt*bytes_Nr,cudaMemcpyDeviceToHost);
  
        //guardar salida Tensor energia momentum
        //for(int i =0; i<Nt;i++){
          //printf("rho[%d]=%0.15f\n",i,T_tt[i*Nr + 20]);
        //}
        guardar_salida_phi(xphix,Nt);
        guardar_salida_psi(xpsix,Nt);
        guardar_salida_pi(xpix,Nt);
  
        guardar_salida_alpha(alpha,Nt);
        guardar_salida_A(A,Nt);
        guardar_salida_B(B,Nt);
  
        guardar_salida_nodo(nodo,Nt);
  
        guardar_salida_rho(T_tt,Nt);
        guardar_salida_SA(T_rr,Nt);
        guardar_salida_ja(T_tr,Nt);
        guardar_salida_SB(T_00,Nt);
  
        guardar_salida_geo(geo,Nt);
        guardar_salida_H(H_cosntrain,Nt);
  
  
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