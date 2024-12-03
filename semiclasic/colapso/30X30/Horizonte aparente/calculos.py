import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
T=1000   ;L=2000
#data=np.loadtxt("campo_escalar.dat").reshape((T,L))
with open("rho2.dat") as f:
    rho = np.fromfile(f, dtype=np.dtype(np.float64))
with open("SA2.dat") as f:
    Sa = np.fromfile(f, dtype=np.dtype(np.float64))
with open("ja2.dat") as f:
    ja = np.fromfile(f, dtype=np.dtype(np.float64))
with open("SB2.dat") as f:
    Sb = np.fromfile(f, dtype=np.dtype(np.float64))
with open("campo_escalar.dat") as f:
    phi = np.fromfile(f, dtype=np.dtype(np.float64))
with open("g_rr.dat") as f:
    g_rr = np.fromfile(f, dtype=np.dtype(np.float64))
with open("g_tt.dat") as f:
    g_tt = np.fromfile(f, dtype=np.dtype(np.float64))
with open("geodesicas.dat") as f:
    geo = np.fromfile(f, dtype=np.dtype(np.float64))
with open("nodo.dat") as f:
    nodo = np.fromfile(f, dtype=np.dtype(np.float64))
L_reduce=int(10/0.025)
rho=rho.reshape((T,L))[:    T,:L_reduce]
Sa=Sa.reshape((T,L))[:T,:L_reduce]
ja=ja.reshape((T,L))[:T,:L_reduce]
Sb=Sb.reshape((T,L))[:T,:L_reduce]

phi=phi.reshape((T,L))[:T,:L_reduce]
g_rr=g_rr.reshape((T,L))[:T,:L_reduce]
g_tt=g_tt.reshape((T,L))[:T,:L_reduce]
geo=geo.reshape((T,L))[:T,:L_reduce]
nodo=nodo.reshape((T,L))[:T,:L_reduce]
x=list(range(50));y=list(range(50))

posicion=np.linspace(0.0,0.025*L_reduce,L_reduce)
ha=np.zeros(T)
#Horionte aparente
for t  in np.arange(T): 
    for r in np.arange(1,L_reduce):
        temp=0
        if geo[t,r-1]*geo[t,r]<0:
            ha[t]=r
#print(ha)
'''
#plt.plot(np.arange(T),g_tt[:,0])
plt.plot(np.arange(T),ha)
plt.legend([r"$r_{HA}$"])
plt.xlabel("t")
plt.ylabel(r"r")
plt.title(r"Radio horizonte aparente")
plt.show()


#Masa inicial
#Solamente valida para t=0, con la evolucion aparecen terminos de curvatura en la integral.
suma=0.0
dr=0.025
for r in np.arange(1,L_reduce-1):
    r2=dr*r*dr*r
    if r%3==0:
        suma+=2*rho[t,r]*r2
    if r%3==1:
        suma+=3*rho[t,r]*r2
    if r%3==2:
        suma+=3*rho[t,r]*r2
suma +=rho[t,0]*0
suma +=rho[t,-1]*dr*(L_reduce-1)*dr*(L_reduce-1)
suma = suma*dr*4*np.pi*3/8
print(suma)

#Evolucion de la masa 
masa_t=np.zeros(T)
dr=0.025
for t in np.arange(T):
    suma=0.0
    for r in np.arange(1,L_reduce-1):
        r2=dr*r*dr*r
        if r%3==0:
            suma+=2*rho[t,r]*r2
        if r%3==1:
            suma+=3*rho[t,r]*r2
        if r%3==2:
            suma+=3*rho[t,r]*r2
    suma +=rho[t,0]*0
    suma +=rho[t,-1]*dr*(L_reduce-1)*dr*(L_reduce-1)
    suma = suma*dr*4*np.pi*3/8
    masa_t[t]=suma
print(masa_t)
print(masa_t[350])
plt.plot(np.arange(T),np.ones(T)*masa_t[0])
plt.plot(np.arange(T),np.ones(T)*masa_t[1])
plt.plot(np.arange(T),masa_t)
plt.legend(["Vacio cuantico","Estado inicial","Masa "])
plt.xlabel("t")
plt.ylabel("Masa")
plt.title(r"Masa usando $4\pi\int_0^\infty \rho r^2dr$")
plt.show()

#Masa de Schwarzschild

M_sch=np.zeros(T)
for t in np.arange(T):
    M_sch[t] = L_reduce*dr/2 * (1-1/g_rr[t,L_reduce-1])

plt.plot(np.arange(T),np.ones(T)*M_sch[0])
plt.plot(np.arange(T),np.ones(T)*M_sch[1])
plt.plot(np.arange(T),M_sch)
plt.legend(["Vacio cuantico","Estado inicial","Masa "])
plt.xlabel("t")
plt.ylabel("Masa")
plt.title(r"Masa usando $\lim_{r\rightarrow\infty}\frac{r}{2}(1-\frac{1}{A(r)})$")
plt.show()

print("Masa constrain : " + str(masa_t[1]) + "| Masa Schwarzschild : " +str(M_sch[0]))
'''
#Instantaneas del lapso
plt.plot([ha[T-1],ha[T-1]],[0,1],"--r",linewidth = 1.0)
plt.plot(np.arange(L_reduce),g_tt[1,:],"-k",linewidth = 0.8)
plt.plot(np.arange(L_reduce),g_tt[5,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[12,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[20,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[30,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[40,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[50,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[60,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[80,:],"-k",linewidth =  0.8)

plt.plot(np.arange(L_reduce),g_tt[100,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[150,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[200,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[300,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[500,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[600,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[750,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[900,:],"-k",linewidth =  0.8)
plt.plot(np.arange(L_reduce),g_tt[T-1,:],"-k",linewidth =  0.8)

plt.legend([r"$r_{HA}$",r"$\alpha^2$"])
plt.xlabel("r")
plt.ylabel(r"$\alpha(t,r)^2$")
plt.title(r"Instantaneas del lapso")
plt.show()