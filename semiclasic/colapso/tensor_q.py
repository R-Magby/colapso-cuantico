import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import scipy.interpolate as sci
T=1000   ;L=500
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
#clasico
with open("clasico/rho.dat") as f:
    clasic_rho = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/ja.dat") as f:
    clasic_ja = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/SA.dat") as f:
    clasic_SA = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/SB.dat") as f:
    clasic_SB = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/g_rr.dat") as f:
    clasic_g_rr = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/g_00.dat") as f:
    clasic_g_00 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/campo_escalar.dat") as f:
    clasic_phi = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/pi.dat") as f:
    clasic_pi = np.fromfile(f, dtype=np.dtype(np.float64))
with open("clasico/psi.dat") as f:
    clasic_psi = np.fromfile(f, dtype=np.dtype(np.float64))

with open("5x5/rho.dat") as f:
    rho_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/ja.dat") as f:
    ja_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/SA.dat") as f:
    SA_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/SB.dat") as f:
    SB_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/g_rr.dat") as f:
    g_rr_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/g_00.dat") as f:
    g_00_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/campo_escalar.dat") as f:
    phi_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/pi.dat") as f:
    pi_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("5x5/psi.dat") as f:
    psi_5x5 = np.fromfile(f, dtype=np.dtype(np.float64))

with open("10x10/rho.dat") as f:
    rho_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/ja.dat") as f:
    ja_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/SA.dat") as f:
    SA_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/SB.dat") as f:
    SB_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/g_rr.dat") as f:
    g_rr_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/g_00.dat") as f:
    g_00_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/campo_escalar.dat") as f:
    phi_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/pi.dat") as f:
    pi_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("10x10/psi.dat") as f:
    psi_10x10 = np.fromfile(f, dtype=np.dtype(np.float64))


with open("20x20/rho.dat") as f:
    rho_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/ja.dat") as f:
    ja_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/SA.dat") as f:
    SA_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/SB.dat") as f:
    SB_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/g_rr.dat") as f:
    g_rr_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/g_00.dat") as f:
    g_00_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/campo_escalar.dat") as f:
    phi_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/pi.dat") as f:
    pi_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))
with open("20x20/psi.dat") as f:
    psi_20x20 = np.fromfile(f, dtype=np.dtype(np.float64))

L_reduce=L
clasic_rho=clasic_rho.reshape((T,L))[:    T,:L_reduce]
clasic_Sa=clasic_SA.reshape((T,L))[:T,:L_reduce]
clasic_ja=clasic_ja.reshape((T,L))[:T,:L_reduce]
clasic_Sb=clasic_SB.reshape((T,L))[:T,:L_reduce]
clasic_g_rr=clasic_g_rr.reshape((T,L))[:T,:L_reduce]
clasic_g_00=clasic_g_00.reshape((T,L))[:T,:L_reduce]
clasic_phi=clasic_phi.reshape((T,L))[:T,:L_reduce]
clasic_pi=clasic_pi.reshape((T,L))[:T,:L_reduce]
clasic_psi=clasic_psi.reshape((T,L))[:T,:L_reduce]

rho_5x5=rho_5x5.reshape((T,L))[:    T,:L_reduce]
ja_5x5=ja_5x5.reshape((T,L))[:    T,:L_reduce]
SA_5x5=SA_5x5.reshape((T,L))[:    T,:L_reduce]
SB_5x5=SB_5x5.reshape((T,L))[:    T,:L_reduce]

g_rr_5x5=g_rr_5x5.reshape((T,L))[:T,:L_reduce]
g_00_5x5=g_00_5x5.reshape((T,L))[:T,:L_reduce]
phi_5x5=phi_5x5.reshape((T,L))[:T,:L_reduce]
pi_5x5=pi_5x5.reshape((T,L))[:T,:L_reduce]
psi_5x5=psi_5x5.reshape((T,L))[:T,:L_reduce]

rho_10x10=rho_10x10.reshape((T,L))[:    T,:L_reduce]
ja_10x10=ja_10x10.reshape((T,L))[:    T,:L_reduce]
SA_10x10=SA_10x10.reshape((T,L))[:    T,:L_reduce]
SB_10x10=SB_10x10.reshape((T,L))[:    T,:L_reduce]

g_rr_10x10=g_rr_10x10.reshape((T,L))[:T,:L_reduce]
g_00_10x10=g_00_10x10.reshape((T,L))[:T,:L_reduce]
phi_10x10=phi_10x10.reshape((T,L))[:T,:L_reduce]
pi_10x10=pi_10x10.reshape((T,L))[:T,:L_reduce]
psi_10x10=psi_10x10.reshape((T,L))[:T,:L_reduce]

rho_20x20=rho_20x20.reshape((T,L))[:    T,:L_reduce]
ja_20x20=ja_20x20.reshape((T,L))[:    T,:L_reduce]
SA_20x20=SA_20x20.reshape((T,L))[:    T,:L_reduce]
SB_20x20=SB_20x20.reshape((T,L))[:    T,:L_reduce]

g_rr_20x20=g_rr_20x20.reshape((T,L))[:T,:L_reduce]
g_00_20x20=g_00_20x20.reshape((T,L))[:T,:L_reduce]
phi_20x20=phi_20x20.reshape((T,L))[:T,:L_reduce]
pi_20x20=pi_20x20.reshape((T,L))[:T,:L_reduce]
psi_20x20=psi_20x20.reshape((T,L))[:T,:L_reduce]
x=list(range(50));y=list(range(50))

posicion=np.linspace(0.0,0.025*L_reduce,L_reduce)

dr=0.025
t_paper=240



plt.plot(posicion,clasic_rho[480,:])
plt.plot(posicion,clasic_ja[480,:])
plt.plot(posicion,clasic_Sa[480,:])
plt.plot(posicion,clasic_Sb[480,:])


#plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{10x10}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()

#5x5
cosmo=-rho_5x5[0,0]
print("5x5 : ",cosmo)
rho_q=(rho_5x5[t_paper,:] + cosmo) - (1/(2*g_rr_5x5[t_paper,:]) * (pi_5x5[t_paper,:]**2/(g_00_5x5[t_paper,:]**2) + psi_5x5[t_paper,:]**2))
ja_q=ja_5x5[t_paper,:] + (pi_5x5[t_paper,:]*psi_5x5[t_paper,:]/(g_00_5x5[t_paper,:]*np.sqrt(g_rr_5x5[t_paper,:])) )
SA_q=(SA_5x5[t_paper,:] - cosmo) - (1/(2*g_rr_5x5[t_paper,:]) * (pi_5x5[t_paper,:]**2/(g_00_5x5[t_paper,:]**2) + psi_5x5[t_paper,:]**2))
SB_q=(SB_5x5[t_paper,:] - cosmo) - (1/(2*g_rr_5x5[t_paper,:]) * (pi_5x5[t_paper,:]**2/(g_00_5x5[t_paper,:]**2) - psi_5x5[t_paper,:]**2))

plt.plot(posicion,-rho_q)
plt.plot(posicion,ja_q)
plt.plot(posicion,SA_q)
plt.plot(posicion,SB_q)


#plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{10x10}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()

#10x10
cosmo=-rho_10x10[0,0]
print("10x10 : ",cosmo)

rho_q=(rho_10x10[t_paper,:] + cosmo) - (1/(2*g_rr_10x10[t_paper,:]) * (pi_10x10[t_paper,:]**2/(g_00_10x10[t_paper,:]**2) + psi_10x10[t_paper,:]**2))
ja_q=ja_10x10[t_paper,:] + (pi_10x10[t_paper,:]*psi_10x10[t_paper,:]/(g_00_10x10[t_paper,:]*np.sqrt(g_rr_10x10[t_paper,:])) )
SA_q=(SA_10x10[t_paper,:] - cosmo) - (1/(2*g_rr_10x10[t_paper,:]) * (pi_10x10[t_paper,:]**2/(g_00_10x10[t_paper,:]**2) + psi_10x10[t_paper,:]**2))
SB_q=(SB_10x10[t_paper,:] - cosmo) - (1/(2*g_rr_10x10[t_paper,:]) * (pi_10x10[t_paper,:]**2/(g_00_10x10[t_paper,:]**2) - psi_10x10[t_paper,:]**2))

plt.plot(posicion,rho_q)
plt.plot(posicion,ja_q)
plt.plot(posicion,SA_q)
plt.plot(posicion,SB_q)


#plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{10x10}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()

#20x20

rho_q=rho_20x20[t_paper,:] - (1/(2*g_rr_20x20[t_paper,:]) * (pi_20x20[t_paper,:]**2/(g_00_20x20[t_paper,:]**2) + psi_20x20[t_paper,:]**2))
ja_q=ja_20x20[t_paper,:] + (pi_20x20[t_paper,:]*psi_20x20[t_paper,:]/(g_00_20x20[t_paper,:]*np.sqrt(g_rr_20x20[t_paper,:])) )
SA_q=SA_20x20[t_paper,:] - (1/(2*g_rr_20x20[t_paper,:]) * (pi_20x20[t_paper,:]**2/(g_00_20x20[t_paper,:]**2) + psi_20x20[t_paper,:]**2))
SB_q=SB_20x20[t_paper,:] - (1/(2*g_rr_20x20[t_paper,:]) * (pi_20x20[t_paper,:]**2/(g_00_20x20[t_paper,:]**2) - psi_20x20[t_paper,:]**2))

plt.plot(posicion,rho_q)
plt.plot(posicion,ja_q)
plt.plot(posicion,SA_q)
plt.plot(posicion,SB_q)


#plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()

#20x20 - clasico

rho_q=rho_20x20[t_paper,:] - (1/(2*clasic_g_rr[t_paper,:]) * (clasic_pi[t_paper,:]**2/(clasic_g_00[t_paper,:]**2) + clasic_psi[t_paper,:]**2))
ja_q=ja_20x20[t_paper,:] + (clasic_pi[t_paper,:]*clasic_psi[t_paper,:]/(clasic_g_00[t_paper,:]*np.sqrt(clasic_g_rr[t_paper,:])) )
SA_q=SA_20x20[t_paper,:] - (1/(2*clasic_g_rr[t_paper,:]) * (clasic_pi[t_paper,:]**2/(clasic_g_00[t_paper,:]**2) + clasic_psi[t_paper,:]**2))
SB_q=SB_20x20[t_paper,:] - (1/(2*clasic_g_rr[t_paper,:]) * (clasic_pi[t_paper,:]**2/(clasic_g_00[t_paper,:]**2) - clasic_psi[t_paper,:]**2))

plt.plot(posicion,rho_q)
plt.plot(posicion,ja_q)
plt.plot(posicion,SA_q)
plt.plot(posicion,SB_q)


#plt.legend([r"$r_{clasic}$",r"$r_{5x5}$",r"$r_{20x20}$"])
plt.xlabel("t")
plt.ylabel(r"$r_{HA}$")
plt.title(r"Radio horizonte aparente")
plt.show()


