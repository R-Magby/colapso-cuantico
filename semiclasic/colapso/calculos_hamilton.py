import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import scipy.interpolate as sci
T=1000   ;L=500
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
#clasico
with open("clasico/rho.dat") as f:
    clasic_rho = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/g_rr.dat") as f:
    clasic_g_rr = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/g_00.dat") as f:
    clasic_g_00 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/geodesicas.dat") as f:
    clasic_geo = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/Hamilton.dat") as f:
    clasic_Ham = np.fromfile(f, dtype=np.dtype(np.float64))

with open("5x5/BH/rho.dat") as f:
    rho_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/BH/g_rr.dat") as f:
    g_rr_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/BH/g_00.dat") as f:
    g_00_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/BH/geodesicas.dat") as f:
    geo_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/BH/Hamilton.dat") as f:
    Ham_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))

with open("10x10/BH/rho.dat") as f:
    rho_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/BH/g_rr.dat") as f:
    g_rr_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/BH/g_00.dat") as f:
    g_00_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/BH/geodesicas.dat") as f:
    geo_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/BH/Hamilton.dat") as f:
    Ham_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/BH/rho.dat") as f:
    rho_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/BH/g_rr.dat") as f:
    g_rr_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/BH/g_00.dat") as f:
    g_00_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/BH/geodesicas.dat") as f:
    geo_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/BH/Hamilton.dat") as f:
    Ham_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))

L_reduce=L
clasic_rho=clasic_rho.reshape((T,L))[:    T,:L_reduce]
clasic_g_rr=clasic_g_rr.reshape((T,L))[:T,:L_reduce]
clasic_g_00=clasic_g_00.reshape((T,L))[:T,:L_reduce]
clasic_geo=clasic_geo.reshape((T,L))[:T,:L_reduce]
clasic_Ham=clasic_Ham.reshape((T,L))[:T,:L_reduce]

rho_5x5=rho_5x5.reshape((T,L))[:    T,:L_reduce]
g_rr_5x5=g_rr_5x5.reshape((T,L))[:T,:L_reduce]
g_00_5x5=g_00_5x5.reshape((T,L))[:T,:L_reduce]
geo_5x5=geo_5x5.reshape((T,L))[:T,:L_reduce]
Ham_5x5=Ham_5x5.reshape((T,L))[:T,:L_reduce]

rho_10x10=rho_10x10.reshape((T,L))[:    T,:L_reduce]
g_rr_10x10=g_rr_10x10.reshape((T,L))[:T,:L_reduce]
g_00_10x10=g_00_10x10.reshape((T,L))[:T,:L_reduce]
geo_10x10=geo_10x10.reshape((T,L))[:T,:L_reduce]
Ham_10x10=Ham_10x10.reshape((T,L))[:T,:L_reduce]

rho_20x20=rho_20x20.reshape((T,L))[:    T,:L_reduce]
g_rr_20x20=g_rr_20x20.reshape((T,L))[:T,:L_reduce]
g_00_20x20=g_00_20x20.reshape((T,L))[:T,:L_reduce]
geo_20x20=geo_20x20.reshape((T,L))[:T,:L_reduce]
Ham_20x20=Ham_20x20.reshape((T,L))[:T,:L_reduce]

x=list(range(50));y=list(range(50))

posicion=np.linspace(0.0,0.025*L_reduce,L_reduce)

dr=0.025
#Horionte aparente
def horizonte_aparente(geo):
    ha=np.zeros(T)
    ha_av=0
    m =0
    for t  in np.arange(T): 
        for r in np.arange(1,L_reduce):
            temp=0
            if geo[t,r-1]*geo[t,r]==0:
                ha[t]=r*dr

            elif geo[t,r-1]*geo[t,r]<0:
                i=0
                a=r*dr -dr
                b=r*dr
                m = (a+b)/2
                ha_av = (geo[t,r-1] + geo[t,r])/2
                #pendiente = (g_00[t,r-1]-g_00[t,r])/dr
                #intercepto = g_00[t,r-1] - pendiente*(r-1)*dr
                #B_predic = pendiente*m + intercepto
                while abs(ha_av) >=0.0001  :
                    i+=1

                    if geo[t,r-1]*ha_av<0:
                        b=m
                        m = (a + b)/2
                        ha_av = (geo[t,r-1] + ha_av)/2

                        #pendiente = (g_00[t,r-1] - B_predic)/(b-a)
                        ##intercepto = g_00[t,r-1] - pendiente*b*(b-a)
                        #B_predic = pendiente*m + intercepto
                    else:
                        a=m
                        m = (a + b)/2
                        ha_av = (geo[t,r] + ha_av)/2

                        #pendiente = (B_predic - g_00[t,r])/(b-a)
                        #intercepto = g_00[t,r] - pendiente*(a)*(b-a)
                        #B_predic = pendiente*(m) + intercepto
                    if i==1000:
                        break
                ha[t]=m

    return ha

ha_clasic=horizonte_aparente(clasic_geo)
ha_5=horizonte_aparente(geo_5x5)
ha_10=horizonte_aparente(geo_10x10)
ha_20=horizonte_aparente(geo_20x20)



plt.plot(np.arange(T)*dr/4,ha_clasic)
plt.plot(np.arange(T)*dr/4,ha_5)
plt.plot(np.arange(T)*dr/4,ha_10)
plt.plot(np.arange(T)*dr/4,ha_20)

plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{10x10}$",r"$r_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()

##Area
def B_predict(g_00,ha):
    B_rah=np.zeros(T)
    for i,r in enumerate(ha[:-1]/0.025):
        a=(r.astype(int))
        b=(r.astype(int))+1
        pendiente = (g_00[i,b]-g_00[i,a])/dr
        intercepto = g_00[i,a] - pendiente*a*dr
        if (r.astype(int))==0:
            B_rah[i]=0.0
        else:
            B_rah[i] = pendiente*r*dr + intercepto
    return B_rah

B_bh_c  = B_predict(clasic_g_00    ,ha_clasic)
B_bh_5  = B_predict(g_00_5x5       ,ha_5)
B_bh_10 = B_predict(g_00_10x10    ,ha_10)
B_bh_20 = B_predict(g_00_20x20    ,ha_20)

Area_HA_c   = 4*np.pi* ha_clasic**2 *   B_bh_c
Area_HA_5   = 4*np.pi* ha_5**2 *        B_bh_5
Area_HA_10  = 4*np.pi* ha_10**2 *       B_bh_10
Area_HA_20  = 4*np.pi* ha_20**2 *       B_bh_20

plt.plot(np.arange(T-1)*dr/4,Area_HA_c[:-1]/(4*np.pi))
plt.plot(np.arange(T-1)*dr/4,Area_HA_5[:-1]/(4*np.pi))
plt.plot(np.arange(T-1)*dr/4,Area_HA_10[:-1]/(4*np.pi))
plt.plot(np.arange(T-1)*dr/4,Area_HA_20[:-1]/(4*np.pi))

#plt.plot(np.arange(T),np.ones(T)*promedio)

plt.legend([r"$A_{clasic}$",r"$A_{5x5}$",r"$A_{10x10}$",r"$A_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$Area_{BH}$")
plt.title(r"Area del agujero negro")
plt.show()

#Masa BH

M_bh_c=ha_clasic*np.sqrt(B_bh_c)/2
M_bh_5=ha_5*np.sqrt(B_bh_5)/2
M_bh_10=ha_10*np.sqrt(B_bh_10)/2
M_bh_20=ha_20*np.sqrt(B_bh_20)/2

plt.plot(np.arange(T-1)*dr/4,M_bh_c[:-1])
plt.plot(np.arange(T-1)*dr/4,M_bh_5[:-1])
plt.plot(np.arange(T-1)*dr/4,M_bh_10[:-1])
plt.plot(np.arange(T-1)*dr/4,M_bh_20[:-1])

plt.legend([r"$M_{clasic}$",r"$M_{5x5}$",r"$M_{10x10}$",r"$M_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$Masa_{BH}$")
plt.title(r"$M_{BH}=\frac{r_{AH}\sqrt{B(r=r_{HA})}}{2}$")
plt.show()

def delta_mass_BH(mass,mass_clasica,t):
    return 100*np.abs(mass[t]-mass_clasica[t])/mass_clasica[t]
delt_m_bh=[(delta_mass_BH(M_bh_5,M_bh_c,240)),(delta_mass_BH(M_bh_10,M_bh_c,240)),delta_mass_BH(M_bh_20,M_bh_c,240)]

plt.plot([5,5,5], delt_m_bh,"-o", color="red")
#plt.legend([r"$\delta{M_{clasic}}$",r"$\delta{M_{5x5}}$",r"$\delta{M_{10x10}}$"])
plt.xlabel("a")
plt.ylabel(r"$\delta{M_{BH}}$")
plt.title(r"$\delta{M_{BH}} $")
plt.show()


#Masa "Schwarzschild-like"

#g_rr g_00




def mass_sch(A,B,r):
    masa_sch=np.zeros(T)
    dA=np.zeros(T)
    for t  in np.arange(T): 
        Area_sph    = 4*np.pi*(r*dr)**2 * B[t,r]
        if r <L_reduce-1:
            Area_sph_p1 = 4*np.pi*((r+1)*dr)**2 * B[t,r+1]
        if r>0:
            Area_sph_m1 = 4*np.pi*((r-1)*dr)**2 * B[t,r-1]

        if r==L_reduce-1:
            dA_sph = (Area_sph-Area_sph_m1)/dr
        elif r==0:
            dA_sph = (Area_sph_p1-Area_sph)/dr
        else:
            if t==240:
                 m2 = 4*np.pi*((r-2)*dr)**2 * B[t,r-2]
                 p2 = 4*np.pi*((r+2)*dr)**2 * B[t,r+2]
                 dA_sph = (-p2 + 2*4*Area_sph_p1 - 2*4*Area_sph_m1 + m2)/(12*dr)
            dA_sph = (Area_sph_p1-Area_sph_m1)/(2*dr)
        dA[t]=dA_sph
        masa_sch[t] = np.sqrt(Area_sph/(16*np.pi)) * (1- (dA_sph**2)/(16*np.pi*A[t,r]*Area_sph))
    return  masa_sch, dA

R_paper=320
msch_c,dAc  = mass_sch(clasic_g_rr  ,clasic_g_00     ,R_paper)
msch_5,dA5  = mass_sch(g_rr_5x5     ,g_00_5x5        ,R_paper)
msch_10,dA10 = mass_sch(g_rr_10x10   ,g_00_10x10      ,R_paper)
msch_20,dA20 = mass_sch(g_rr_20x20   ,g_00_20x20      ,R_paper)
plt.plot(np.arange(T)*dr/4,dA10)
plt.plot(np.arange(T)*dr/4,dA20)


plt.legend([r"$A$",r"$B$"])
plt.xlabel("t")
plt.ylabel(r"$g$")
plt.title(r"$A y B en r : $" + str(320*dr))
plt.show()


plt.plot(np.arange(T)*dr/4,msch_c)
plt.plot(np.arange(T)*dr/4,msch_5)
plt.plot(np.arange(T)*dr/4,msch_10)
plt.plot(np.arange(T)*dr/4,msch_20)

plt.legend([r"$M_{clasic}$",r"$M_{5x5}$",r"$M_{10x10}$",r"$M_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$Masa_{BH}$")
plt.title(r"$M_{BH}=\frac{r_{AH}\sqrt{B(r=r_{HA})}}{2}$")
plt.show()

## Delta masa ADM
  
def delta_mass_ADM(mass,t):
    return np.abs(mass[t]-mass[0])/mass[0]

t_paper=240

delt_masas=[np.log10(delta_mass_ADM(msch_c,t_paper)),np.log10(delta_mass_ADM(msch_5,t_paper)),np.log10(delta_mass_ADM(msch_10,t_paper)),np.log10(delta_mass_ADM(msch_20,t_paper))]
plt.plot([0.0,25,100,20*20], delt_masas,"-o", color="red"       )

print((delt_masas))

#plt.legend([r"$\delta{M_{clasic}}$",r"$\delta{M_{5x5}}$",r"$\delta{M_{10x10}}$"])
plt.xlabel("N")
plt.ylabel(r"$\log{\delta{M_{ADM}}}$")
plt.title(r"$\delta{M_{ADM}} $ \| t="+str(t_paper*dr/4))
plt.show()


# hamiltoniano en funcion del tiempo
def hamilton(ham):
    ham_sist=np.zeros(T)
    for t in np.arange(T):
        suma=0.0
        for r in np.arange(1,L_reduce//2-1):
            r2=dr*r*dr*r
            if r%3==0:
                suma+=np.abs(ham[t,r])**2
            if r%3==1:
                suma+=np.abs(ham[t,r])**2
            if r%3==2:
                suma+=np.abs(ham[t,r])**2
        suma +=np.abs(ham[t,r])**2
        suma +=np.abs(ham[t,r])**2
        suma = np.sqrt(suma)/L_reduce
        
        ham_sist[t]=suma
    return ham_sist
ham_sist_c = hamilton(clasic_Ham)
ham_sist_5 = hamilton(Ham_5x5)
ham_sist_10 = hamilton(Ham_10x10)
ham_sist_20 = hamilton(Ham_20x20)





plt.plot(np.arange(T)*dr/4,ham_sist_c,"-r")
plt.plot(np.arange(T)*dr/4,ham_sist_5,"-k")
plt.plot(np.arange(T)*dr/4,ham_sist_10,"--g")
plt.plot(np.arange(T)*dr/4,ham_sist_20,"-.b")


plt.legend([r"$H_{clasic}$",r"$H_{5x5}$",r"$H_{10x10}$",r"$H_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$\log{ \|H_{L_{2}}\|}$")
plt.title(r"Hamiltoniano$")
plt.show()

def logham(ham_sistema):
    suma=0
    for t in np.arange(T):
        if t%3==0:
            suma+=ham_sistema[t]**1
        if t%3==1:
            suma+=ham_sistema[t]**1
        if t%3==2:
            suma+=ham_sistema[t]**1
    suma +=ham_sistema[0]**1
    suma +=ham_sistema[-1]**1
    return np.sqrt(suma)/T


delt_masas=[np.log10(logham(ham_sist_c)),np.log10(logham(ham_sist_5)),np.log10(logham(ham_sist_10)),np.log10(logham(ham_sist_20))]
plt.plot([500,500,500,500], delt_masas,"-o", color="red"       )

print((delt_masas))

#plt.legend([r"$\delta{M_{clasic}}$",r"$\delta{M_{5x5}}$",r"$\delta{M_{10x10}}$"])
plt.xlabel("N")
plt.ylabel(r"$\log{\delta{M_{ADM}}}$")
plt.title(r"$\delta{M_{ADM}} $ \| t="+str(t_paper*dr/4))
plt.show()
