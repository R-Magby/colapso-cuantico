
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

// Constant
#define consta_pi 3.1415926535897932384626433832795


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
#define dr 0.25
#define Nt 10
#define dt dr/4.0

// Parameter quantum
#define Nk 20
#define Nl 20
#define dk consta_pi/15.0

#define  dim_q Nk*Nl*Nr

//Runge-Kutta tenth order implicit
#define step 5

__device__ double a_ij[25];
__device__ double b_i[5];
__device__ double c_i[5];


struct Relativity_G{
    double *A, *B, *alpha;
    double *Da, *Db;
    double *K, *Kb;
    double *U, *lambda;
}
struct field_Classic{
    double *phi, *psi, *pi;
}
struct field_imag{
    double *u, *psi, *pi;
}
struct field_real{
    double *u, *psi, *pi;
}
struct field_Quantum{
    struct field_real real;
    struct field_imag imag;
}


__device__ double psi_dot(Relativity_G metrics, *double field_pi, id_nodo, idx){
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
__device__ double f_pi_dot(fields_classic field_C, Relativity_G metrics, int id){
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

__global__ void Tensor_energy_momentum(double *rho, double *ja, double *SA, double *SB, Relativity_G metrics, field_Classic field_C, field_Quantum field_Q, int g){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int id;
    int kinetic;
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
        for(int l=0 : l<Nl ; l++){

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
  
        rho[idx] +=     kinetic*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) + 1.0/(metrics.B[idx]*radio*radio)*xtheta2x );
  
        ja[idx] += -    kinetic*xpi_psix/(sqrt(metrics.A[idx])*metrics.B[idx]);
    
        SA[idx] +=      kinetic*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) + xpsi2x) - 1.0/(metrics.B[idx]*radio*radio)*xtheta2x );
    
        SB[idx] +=      kinetic*(1.0/(2.0*metrics.A[idx]) * (xpi2x/(metrics.B[idx]*metrics.B[idx]) - xpsi2x) );
    }
}
  
__global__ void T_zeros(double *rho, double *ja, double *SA, double *SB){
    idx=threadIdx.x + blockDim.x*blockIdx.x;
    if (idx<Nr){
        rho[idx]=0.0;
        SA[idx]=0.0;
        SB[idx]=0.0;
        ja[idx]=0.0;
    }
}
__global__ void evo_metrics(Relativity_G metrics,Relativity_G metrics_RK, double *rho, double *ja, double *SA, double*SB,
                            double Cosm, int id, double dr){

  int id = threadIdx.x + blockDim.x*blockIdx.x;
  double radio;
  double f1,f2,f3;
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
    f1 =(power(metrics.K[id],2) - 4.0*metrics.K[id]*metrics.Kb[id] + 6.0*power(metrics.Kb[id],2));
    f3= (rho[id] + SA[id] + 2.0*SB[id] - 2.0*Cosm);
    if(id==0){
        f2=(3.0*derivate(metrics.Da,0,id,1) + power(metrics.Da[id],2) - 0.5*metrics.Da[id]* temp );
    }
    else{
        f2=(derivate(metrics.Da,0,id,1) + power(metrics.Da[id],2) + 2.0*metrics.Da[id]/radio- 0.5*metrics.Da[id]* temp );
    }

    metrics_RK.K[id] =  metrics.alpha[id]* f1 - metrics.alpha[id]/metrics.A[id] * f2 + 0.5*metrics.alpha[id] * f3;

    //Kb
    //Solamente derivaremos las funciones que son impares, ya que las demas actuan como constantes en r = 0,
    if(id==0){
         f1 = (0.5*(derivate( metrics.U,0,id,1) + 4*metrics.B[id]/metrics.A[id]*derivate(metrics.lambda,0,id,1)) - derivate(metrics.Db,0,id,1) - derivate(metrics.lambda,0,id,1) - derivate(metrics.Da,0,id,1) );
    }
    else{
        f1 = 1.0/radio * (0.5*temp - metrics.Db[id] - metrics.lambda[id] - metrics.Da[id] );
    }
    f2=(-0.5*metrics.Da[id]*metrics.Db[id] - 0.5*derivate(metrics.Db,0,id,1) + 0.25*metrics.Db[id]* temp + metrics.A[id]*metrics.K[id]*metrics.Kb[id]);
    f3=(SA[id] - rho[id] - 2.0*Cosm);

    metrics_RK.Kb[id] = metrics.alpha[id]/metrics.A[id] * (  f1 + f2  )+ 0.5*metrics.alpha[id] * f3;

    //                                                ********Regularacion********

    metrics_RK.U[id] = -2.0*metrics.alpha[id] * (derivate( metrics.K,0,id,0) + metrics.Da[id]* ( metrics.K[id] - 4.0*mertics.Kb[id] ) 
                                                    - 2.0* ( metrics.K[id] - 3.0*metrics.Kb[id] ) * ( metrics.Db[id] - 2.0*metrics.lambda[id]*metrics.B[id]/metrics.A[id] ))
                        -4.0*metrics.alpha[id] * ja[id];

    temp = 2.0*metrics.alpha[id]*metrics.A[id]/metrics.B[id];
    metrics_RK.lambda[id] = temp * ( derivate(metrics.Kb,0, id, 0) - 0.5*metrics.Db[id] * ( metrics.K[id] - 3.0*metrics.Kb[id] ) + 0.5*ja[id] );

}
__global__ void evo_fields_classics(fields_classic field_C, fields_classic field_C_RK, Relativity_G metrics){
    int id = threadIdx.x + blockDim.x*blockIdx.x;;

  double radio,temp;
  if(id==0){radio=dr;}
  else{radio=id*dr;}
  
  temp = metrics.alpha[id]/(sqrt(metrics.A[id])*metrics.B[id]);
  
  field_C_RK.phi[id] = temp*field_C.pi[id];
  field_C_RK.psi[id] = psi_dot(metrics,field_C.pi,0,id);
    if(id==0){
        field_C_RK.pi[id] =  3.0*f_pi_dot(field_C,metrics,radio,id) ;  
    }
    else{
        field_C_RK.pi[id] =  f_pi_dot(field_C,metrics,radio,id) + 2.0*metrics.alpha[id]*metrics.B[id] * field_C.psi[id]/sqrt(metrics.A[id])/radio;  
    }
}
__global__ void evo_fields_quantums( Relativity_G metrics,   field_Quantum field_Q, field_Quantum field_Q_RK, double psi_Q_prime,  double mass, int g){
    int id = g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;;
    int r = id%Nr;
    int k = (int)id/Nr%Nk;
    int l = (int)id/Nr/Nl;
    int id_nodo =g*dim_q +l*Nr*Nk + K*Nr;
    double temp[2];
    double A_B_alpha_prime;
    field_Q_RK.real.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.real.pi[id];
    field_Q_RK.imag.u[id] = metrics.alpha[r]/(sqrt(metrics.A[r])*metrics.B[r]) * field_Q.imag.pi[id];

    temp[0] = psi_dot(metrics, field_Q.real.pi, id_nodo ,r);
    temp[1] = psi_dot(metrics, field_Q.imag.pi, id_nodo ,r);

    field_Q_RK.real.psi[id] = temp[0];
    field_Q_RK.real.psi[id] = temp[1];

    double f1, f2, f3, f4, radio;
    if (r==0){
        radio = o;
    }
    else{
        radio = dr*r;
    }
    A_B_alpha_prime = derivate_metric(metrics, id);
    //pi real
    if(r==0){
        f1 = (l+1) * ( field_Q.real.psi[id]); // cero
        f2 = (2*l+3) * derivate(field_Q.real.psi,id_nodo,r,1);
        f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.real.u[id];
    }
    else{
        f1 = l/radio * (field_Q.real.u[id]) + (field_Q.real.psi[id]);
        f2 = (2*l+2)/radio * field_Q.real.psi[id] + temp[0];
        f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.real.u[id];
    }
    f4 = mass * mass* field_Q.real.u[id]* field_Q.real.u[id];

    field_Q_RK.real.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 +f3) - metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*f4;

    //pi imaginario
    if(r==0){
        f1 = (l+1) * ( field_Q.imag.psi[id]); // cero
        f2 = (2*l+3) *  derivate(field_Q.imag.psi,id_nodo,r,1);
        f3 = l*(l+1) * derivate(metrics.lambda, 0, r,1)*field_Q.imag.u[id];
    }
    else{

        f1 = l/radio * (field_Q.imag.u[id]) + (field_Q.imag.psi[id]);
        f2 = (2*l+2)/radio * field_Q.imag.psi[id] + temp[1];
        f3 = l*(l+1)/radio * metrics.lambda[r]*field_Q.reimagal.u[id];
    }
    f4 = mass * mass* field_Q.imag.u[id]* field_Q.imag.u[id];

    field_Q_RK.imag.pi[id] = A_B_alpha_prime*f1 +  metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*(f2 +f3) - metrics.alpha[r]*metrics.B[r]/(sqrt(metrics.A[r]))*f4;
}

__global__ void y_tilda_metrics( Relativity_G metrics , Relativity_G RG_K1, Relativity_G RG_K2,  Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5, Relativity_G y_tilde_M, int s){
    int id = threadIdx.x + blockDim.x*blockIdx.x;;
    y_tilda_M.A[id]      = metrics.A[id]        + dt*( a_ij[s][0]*RG_K1.A[id]      + a_ij[s][1]*RG_K2.A[id]      + a_ij[s][2]*RG_K3.A[id]      + a_ij[s][3]*RG_K4.A[id]      + a_ij[s][4]*RG_K5.A[id] );
    y_tilda_M.B[id]      = metrics.B[id]        + dt*( a_ij[s][0]*RG_K1.B[id]      + a_ij[s][1]*RG_K2.B[id]      + a_ij[s][2]*RG_K3.B[id]      + a_ij[s][3]*RG_K4.B[id]      + a_ij[s][4]*RG_K5.B[id] );
    y_tilda_M.alpha[id]  = metrics.alpha[id]    + dt*( a_ij[s][0]*RG_K1.alpha[id]  + a_ij[s][1]*RG_K2.alpha[id]  + a_ij[s][2]*RG_K3.alpha[id]  + a_ij[s][3]*RG_K4.alpha[id]  + a_ij[s][4]*RG_K5.alpha[id] );
    y_tilda_M.Da[id]     = metrics.Da[id]       + dt*( a_ij[s][0]*RG_K1.Da[id]     + a_ij[s][1]*RG_K2.Da[id]     + a_ij[s][2]*RG_K3.Da[id]     + a_ij[s][3]*RG_K4.Da[id]     + a_ij[s][4]*RG_K5.Da[id] );
    y_tilda_M.Db[id]     = metrics.Db[id]       + dt*( a_ij[s][0]*RG_K1.Db[id]     + a_ij[s][1]*RG_K2.Db[id]     + a_ij[s][2]*RG_K3.Db[id]     + a_ij[s][3]*RG_K4.Db[id]     + a_ij[s][4]*RG_K5.Db[id] );
    y_tilda_M.Kb[id]     = metrics.Kb[id]       + dt*( a_ij[s][0]*RG_K1.Kb[id]     + a_ij[s][1]*RG_K2.Kb[id]     + a_ij[s][2]*RG_K3.Kb[id]     + a_ij[s][3]*RG_K4.Kb[id]     + a_ij[s][4]*RG_K5.Kb[id] );
    y_tilda_M.K[id]      = metrics.K[id]        + dt*( a_ij[s][0]*RG_K1.K[id]      + a_ij[s][1]*RG_K2.K[id]      + a_ij[s][2]*RG_K3.K[id]      + a_ij[s][3]*RG_K4.K[id]      + a_ij[s][4]*RG_K5.K[id] );
    y_tilda_M.lambda[id] = metrics.lambda[id]   + dt*( a_ij[s][0]*RG_K1.lambda[id] + a_ij[s][1]*RG_K2.lambda[id] + a_ij[s][2]*RG_K3.lambda[id] + a_ij[s][3]*RG_K4.lambda[id] + a_ij[s][4]*RG_K5.lambda[id] );
    y_tilda_M.U[id]      = metrics.U[id]        + dt*( a_ij[s][0]*RG_K1.U[id]      + a_ij[s][1]*RG_K2.U[id]      + a_ij[s][2]*RG_K3.U[id]      + a_ij[s][3]*RG_K4.U[id]      + a_ij[s][4]*RG_K5.U[id] );
}
    
__global__ void y_tilda_field_classic(field_Classic field_C, field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5, field_Classic y_tilda_C, int s){

    int id = threadIdx.x + blockDim.x*blockIdx.x;;
    y_tilda_C.phi[id] = field_C.phi[id] + dt*( a_ij[s][0]* FC_K1.phi[id] + a_ij[s][1]* FC_K2.phi[id] + a_ij[s][2]* FC_K3.phi[id] + a_ij[s][3]* FC_K4.phi[id] + a_ij[s][4]* FC_K5.phi[id]);
    y_tilda_C.pi[id]  = field_C.pi[id]  + dt*( a_ij[s][0]* FC_K1.pi[id] + a_ij[s][1]* FC_K2.pi[id] + a_ij[s][2]* FC_K3.pi[id] + a_ij[s][3]* FC_K4.pi[id] + a_ij[s][4]* FC_K5.pi[id]);
    y_tilda_C.psi[id] = field_C.psi[id] + dt*( a_ij[s][0]* FC_K1.psi[id] + a_ij[s][1]* FC_K2.psi[id] + a_ij[s][2]* FC_K3.psi[id] + a_ij[s][3]* FC_K4.psi[id] + a_ij[s][4]* FC_K5.psi[id]);

  
}
__global__ void y_tilda_field_quantum(field_Quantum field_Q, field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5, field_Quantum y_tilda_Q, int s, int g){

    int id =  g*dim_q + threadIdx.x + blockDim.x*blockIdx.x;
    y_tilda_Q.real.u[id] = field_Q.real.u[id]     + dt*( a_ij[s][0]* FQ_K1.real.u[id]     + a_ij[s][1]* FQ_K2.real.u[id]    + a_ij[s][2]* FQ_K3.real.u[id]    + a_ij[s][3]* FQ_K4.real.u[id]    + a_ij[s][4]* FQ_K5.real.u[id]);
    y_tilda_Q.real.pi[id]  = field_Q.real.pi[id]  + dt*( a_ij[s][0]* FQ_K1.real.pi[id]    + a_ij[s][1]* FQ_K2.real.pi[id]   + a_ij[s][2]* FQ_K3.real.pi[id]   + a_ij[s][3]* FQ_K4.real.pi[id]   + a_ij[s][4]* FQ_K5.real.pi[id]);
    y_tilda_Q.real.psi[id] = field_Q.real.psi[id] + dt*( a_ij[s][0]* FQ_K1.real.psi[id]   + a_ij[s][1]* FQ_K2.real.psi[id]  + a_ij[s][2]* FQ_K3.real.psi[id]  + a_ij[s][3]* FQ_K4.real.psi[id]  + a_ij[s][4]* FQ_K5.real.psi[id]);

    y_tilda_Q.imag.u[id] = field_Q.imag.u[id]     + dt*( a_ij[s][0]* FQ_K1.imag.u[id]     + a_ij[s][1]* FQ_K2.imag.u[id]    + a_ij[s][2]* FQ_K3.imag.u[id]    + a_ij[s][3]* FQ_K4.imag.u[id]    + a_ij[s][4]* FQ_K5.imag.u[id]);
    y_tilda_Q.imag.pi[id]  = field_Q.imag.pi[id]  + dt*( a_ij[s][0]* FQ_K1.imag.pi[id]    + a_ij[s][1]* FQ_K2.imag.pi[id]   + a_ij[s][2]* FQ_K3.imag.pi[id]   + a_ij[s][3]* FQ_K4.imag.pi[id]   + a_ij[s][4]* FQ_K5.imag.pi[id]);
    y_tilda_Q.imag.psi[id] = field_Q.imag.psi[id] + dt*( a_ij[s][0]* FQ_K1.imag.psi[id]   + a_ij[s][1]* FQ_K2.imag.psi[id]  + a_ij[s][2]* FQ_K3.imag.psi[id]  + a_ij[s][3]* FQ_K4.imag.psi[id]  + a_ij[s][4]* FQ_K5.imag.psi[id]);
  
}


//Metodo runge kutta de decimo orden implicito, ocupando el metodo de gauss-legendre. 
//Orden del metodo es p = S*2, siendo S el numero de pasos ("step")
void RK_implicit_tenth(Relativity_G metrics, Relativity_G RG_K1,Relativity_G RG_K2, Relativity_G RG_K3, Relativity_G RG_K4, Relativity_G RG_K5, Relativity_G y_tilde_M,
    field_Classic field_C, field_Classic FC_K1, field_Classic FC_K2, field_Classic FC_K3, field_Classic FC_K4, field_Classic FC_K5, field_Classic y_tilda_C,
    field_Quantum field_Q, field_Quantum FQ_K1, field_Quantum FQ_K2, field_Quantum FQ_K3, field_Quantum FQ_K4, field_Quantum FQ_K5, field_Quantum y_tilda_Q,
    double *rho, double *ja, double *SA, double *SB, double Cosm ){
    int thread = 64;
    dim3  bloque(thread);
    dim3  grid_radial ((int)Nr/thread);
    dim3  grid_quantum ((int)dim_q/thread);

    //ciclo for de la cantidad de pasos
    for(int s=0 ; s<step  ; s++){

        //Calculo de y_(n+1)
        y_tilda_metrics         <<< grid_radial, bloque >>>(metrics,RG_K1,RG_K2,RG_K3,RG_K4,RG_K5, y_tilde_M, s);
        y_tilda_field_classic   <<< grid_radial, bloque >>>(field_C,FC_K1,FC_K2,FC_K3,FC_K4,FC_K5, y_tilde_C, s);
        for (int g=0; g < 6; g++){
            y_tilda_field_quantum   <<< grid_quantum, bloque >>>(field_Q,FQ_K1,FQ_K2,FQ_K3,FQ_K4,FQ_K5, y_tilde_Q, s, g);
        }

        //Actualizacion de los K
        if(s==0){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilda_M, RG_K1, rho, ja, SA, SB, Cosm, id, dr);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K1,  y_tilda_M);
            for (int g=0; g < 6; g++){
                evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K1,  y_tilda_M, g);
            }
        }
        else if(s==1){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilda_M, RG_K2, rho, ja, SA, SB, Cosm, id, dr);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K2,  y_tilda_M);
            for (int g=0; g < 6; g++){
                evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K2,  y_tilda_M, g);
            }
        }
        else if(s==2){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilda_M, RG_K3, rho, ja, SA, SB, Cosm, id, dr);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K3,  y_tilda_M);
            for (int g=0; g < 6; g++){
                evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K3,  y_tilda_M, g);
            }
        }
        else if(s==3){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilda_M, RG_K4, rho, ja, SA, SB, Cosm, id, dr);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K4,  y_tilda_M);
            for (int g=0; g < 6; g++){
                evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K4,  y_tilda_M, g);
            }
        }
        else if(s==4){
            evo_metrics         <<< grid_radial, bloque >>>( y_tilda_M, RG_K5, rho, ja, SA, SB, Cosm, id, dr);
            evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K5,  y_tilda_M);
            for (int g=0; g < 6; g++){
                evo_fields_classics <<< grid_radial, bloque >>>( y_tilda_C,  FC_K5,  y_tilda_M, g);
            }
        }

        //Calculo del tensor energia momentum en el tiempo n+1 
        T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
        for (int g=0; g < 6; g++){
            Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, y_tilda_M, y_tilda_C, y_tilda_Q, g);
        }
    }
}

void cargar_coeficcientes(double *a_ij,double *b_i, double *c_i){
    FILE *arch;
    arch=fopen("a_ij.npy","rb");
    if (arch==NULL)
        exit(1);
    fread( a_ij , sizeof(double) , 25 , arch );
    fclose(arch);

    FILE *arch;
    arch=fopen("b_i.npy","rb");
    if (arch==NULL)
        exit(1);
    fread( b_i , sizeof(double) , 5 , arch );
    fclose(arch);

    FILE *arch;
    arch=fopen("c_i.npy","rb");
    if (arch==NULL)
        exit(1);
    fread( c_i , sizeof(double) , 5 , arch );
    fclose(arch);
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
//C.I
__globa__ void u_initial(field_Quantum field_Q, int k, int l, int field, int Nr,double dr, double dk , double mass){
    int r = threadIdx.x + blockDim.x*blockIdx.x;
    int id_u= field + l*Nk*Nr + k*Nr ;
    double omega = sqrt( dk*(k+1)*dk*(k+1) + mass*mass ) ;

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

int main() {
    size_t bytes_Nr = Nr * sizeof(double);
    size_t bytes_Q = 6 * Nr*Nl*Nk * sizeof(double);
    // dejo el campo cuantico escalar y fantasmas  en un solo malloc, asi evito repetir funciones.
    double * _host_a_ij;
    double * _host_b_i;
    double * _host_c_i;
    //Cargamos los coeficientes de runge kutta
    _host_a_ij =(double *)malloc(25*sizeof(double));
    _host_b_i =(double *)malloc(5*sizeof(double));
    _host_c_i =(double *)malloc(5*sizeof(double));

    cargar_coeficcientes(_host_a_ij,_host_b_i,_host_c_i);
    //Copiamos los coeficientes del runge kutta del host al device
    cudaMemcpyToSymbol(a_ij, _host_a_ij, sizeof(_host_a_ij));
    cudaMemcpyToSymbol(b_i , _host_b_i,  sizeof(_host_b_i));
    cudaMemcpyToSymbol(c_i , _host_c_i,  sizeof(_host_c_i));


    // Asignar memoria para la estructura en el host
    Relativity_G metrics, RG_K1, RG_K2, RG_K3, RG_K4,RG_K5, y_tilda_M;
    field_Classic field_C, FC_K1, FC_K2, FC_K3, FC_K4, FC_K5, y_tilda_C;
    field_Quantum field_Q, FQ_K1, FQ_K2, FQ_K3, FQ_K4, FQ_K5, y_tilda_Q;
    //field_Ghost field_G;

    Relativity_G point_RG[]     = {metrics, RG_K1, RG_K2, RG_K3, RG_K4, RG_K5, y_tilda_M};
    field_Classic poiint_FQ[]   = {field_C, FC_K1, FC_K2, FC_K3, FC_K4, FC_K5, y_tilda_C};
    field_Quantum point_FC[]    = {field_Q, FQ_K1, FQ_K2, FQ_K3, FQ_K4, FQ_K5, y_tilda_Q};

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
    cudaMalloc((void **)&(field_C), sizeof(field_clasic)); 
    cudaMalloc((void **)&(field_C.phi), bytes_Nr);  
    cudaMalloc((void **)&(field_C.pi), bytes_Nr);   
    cudaMalloc((void **)&(field_C.psi), bytes_Nr);

    //Quamtum
    cudaMalloc((void **)&(field_Q), sizeof(field_Quantum)); 
    cudaMalloc((void **)&(field_Q.phi), bytes_Q);  
    cudaMalloc((void **)&(field_Q.pi), bytes_Q);   
    cudaMalloc((void **)&(field_Q.psi), bytes_Q);

    for (int i = 0; i < sizeof(point_RG) / sizeof(point_RG[0]); ++i) {
        //Metrics
        cudaMalloc((void **)&(point_RG[i]), sizeof(Relativity_G)); 
        cudaMalloc((void **)&(point_RG[i].A), bytes_Nr);  
        cudaMalloc((void **)&(point_RG[i].B     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].alpha ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].Da    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].Db    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].K     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].Kb    ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].U     ), bytes_Nr);   
        cudaMalloc((void **)&(point_RG[i].lambda), bytes_Nr); 
        
        //Classic
        cudaMalloc((void **)&(point_FC[i]), sizeof(field_clasic)); 
        cudaMalloc((void **)&(point_FC[i].phi), bytes_Nr);  
        cudaMalloc((void **)&(point_FC[i].pi), bytes_Nr);   
        cudaMalloc((void **)&(point_FC[i].psi), bytes_Nr);

        //Quamtum
        cudaMalloc((void **)&(point_FQ[i]), sizeof(field_Quantum)); 
        cudaMalloc((void **)&(point_FQ[i].phi), bytes_Q);  
        cudaMalloc((void **)&(point_FQ[i].pi), bytes_Q);   
        cudaMalloc((void **)&(point_FQ[i].psi), bytes_Q);
    }

    //Condiciones iniciales


    //Evolucion
    for (int t=0 ; t<Nt ;t++){
        T_zeros <<< grid_radial, bloque >>>(rho, ja, SA,SB);
        for (int g=0; g < 6; g++){
            Tensor_energy_momentum <<< grid_radial, bloque >>>(rho, ja, SA, SB, y_tilda_M, y_tilda_C, y_tilda_Q, g);
        }

    }


}
