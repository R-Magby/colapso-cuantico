import numpy as np
from scipy.optimize import fsolve,root,minimize,newton_krylov
from itertools import chain
from scipy.special import legendre
from scipy.linalg import solve

def runge_kutta_10th_order(s,r):
    size_a=0
    c_i=np.zeros(s)
    c_i[0] = 0.0
    c_i[ s-1] = 1.0


    gamma_1=0.5*(1 - np.sqrt(( 7 - 2 * np.sqrt(7)) / 21))
    gamma_2=0.5*(1 - np.sqrt(( 7 + 2 * np.sqrt(7)) / 21))
    gamma_3= 1 - gamma_2
    gamma_4= 1 - gamma_1

    c_i[ 8]=gamma_1
    c_i[ 9]=gamma_3
    c_i[ 10]=gamma_4
    c_i[ 11]=gamma_2

    c_i[1]=0.5
    c_i[5]=0.7666539862535488
    c_i[12]=c_i[5]
    c_i[15]=c_i[1]
    def equations_for_c(vars):
        for i,num in enumerate([2,3,4,6,7,13,14]):
            c_i[num]=vars[i]

        eqs = []


        #for r in [0,8,9,10,11,16]:
        #    eqs.append(np.dot(b_i,c_i**r)-1/(r+1))
        eqs.append(c_i[13]-c_i[6])
        eqs.append(c_i[14]-c_i[2])
        eqs.append(c_i[2]-2.0/3.0*c_i[3]); eqs.append(c_i[4]-(4.0*c_i[3]-3.0*c_i[5])/(6.0*c_i[3]-4.0*c_i[5])*c_i[5])
        eqs.append(c_i[7]-(20.0*c_i[5]*c_i[6] - 15.0*c_i[8]*(c_i[5]+c_i[6]) + 12.0*c_i[8]**2)/(30.0*c_i[5]*c_i[6] - 20.0*c_i[8]*(c_i[5]+c_i[6]) + 15.0*c_i[8]**2)*c_i[8])
        eqs.append(1-c_i[13]-(4.0*(1.0-c_i[12])-3.0*(1-c_i[11]))/(6.0*(1.0-c_i[12])-4.0*(1-c_i[11])) * (1-c_i[11]))
        eqs.append(1-c_i[14]-2.0/3.0*(1.0-c_i[13]))
        return eqs
    

    # Inic_iializar valores 
    initial_guess = np.random.rand(s-10)
    solution_c = root(equations_for_c, initial_guess)
    for i,num in enumerate([2,3,4,6,7,13,14]):
        c_i[num]=solution_c.x[i]
    b_i=np.zeros(s)

    b_i[0]=1.0/30.0
    b_i[16]=b_i[0]
    b_i[3]=0.0
    b_i[4]=0.0
    b_i[7]=0.0
    b_i[12]=0.13
    b_i[13]=0.18
    b_i[14]=0.12
    b_i[15]=1.0/30

    for i in range(1,5):
        b_i[7+i]=1.0/(30.0*(np.polynomial.legendre.Legendre([0,0,0,0,0,1],[0,1],[-1,1])(c_i[7+i]))**2)

    def equations_for_b(vars):
        for i,num in enumerate([1,2,5,6]):
            b_i[num]=vars[i]

        eqs = []

        #for r in [0,8,9,10,11,16]:
        #    eqs.append(np.dot(b_i,c_i**r)-1/(r+1))
        eqs.append(b_i[1]+b_i[15]); eqs.append(b_i[2]+b_i[14]); eqs.append(b_i[5]+b_i[12])
        eqs.append(b_i[6]+b_i[13])
        return eqs
    

    initial_guess = np.random.rand(s-13)
    solution_b = root(equations_for_b, initial_guess)
    
    for i,num in enumerate([1,2,5,6]):
        b_i[num]=solution_b.x[i]


    a_ij=np.zeros((s,s))
    for i in range(3,14):
        a_ij[i,1]=0.0

    for i in range(5,14):
        a_ij[i,2]=0.0

    for i in chain(range(7,12),[s-1]):
        a_ij[i,3]=0.0
        
    for i in chain(range(8,12),[s-1]):
        a_ij[i,4]=0.0

    for j in chain(range(3,14),[1]):
        a_ij[15,j]=0.0

    for j in [2,3,4,7,8,9,10,11]:
        a_ij[14,j]=0.0

    #step1
    a_ij[1,0]=c_i[1]
    def step1(vars):    
        a_sol=[]
        for i in range(2,10):
            array_vander=c_i[:i]
            if i==2:
                c_sol=[c_i[i],c_i[i]**2/2]
                n_col=len(c_sol)
            elif i==3 or i==4:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3]
                n_col=len(c_sol)
                if i==4:
                    array_vander=c_i[[0,2,3]]
            elif i>=5 and i<8:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,c_i[i]**4/4]
                n_col=len(c_sol)
                if i==5:
                    array_vander=c_i[[0,2,3,4]]
                elif i==6:
                    array_vander=c_i[[0,3,4,5]]
                elif i==7:
                    array_vander=c_i[[0,4,5,6]]
            elif i==8 or i==9:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,c_i[i]**4/4,c_i[i]**5/5]
                n_col=len(c_sol)
                if i==8:
                    array_vander=c_i[[0,4,5,6,7]]
                elif i==9:
                    array_vander=c_i[[0,5,6,7,8]]
            c_vander=np.transpose(np.vander(array_vander, increasing=True))

            a_sol.append(np.dot(np.linalg.inv(c_vander),c_sol))
        return a_sol

    initial_guess = np.random.rand(s-13)

    solution_a = step1(initial_guess)
    print(solution_a)
    for id,i in enumerate(range(2,10)):
        if i<=3:
            a_ij[i,:i]=solution_a[id]

        elif  i==4:
            a_ij[i,[0,2,3]]=solution_a[id]

        elif i>=5 and i<8:
            if i==5:
                a_ij[i,[0,2,3,4]]=solution_a[id]
            elif i==6:
                a_ij[i,[0,3,4,5]]=solution_a[id]
            elif i==7:
                a_ij[i,[0,4,5,6]]=solution_a[id]
        elif i==8 or i==9:
            if i==8:
                a_ij[i,[0,4,5,6,7]]=solution_a[id]
            elif i==9:
                a_ij[i,[0,5,6,7,8]]=solution_a[id]

    #a17,16
    a_ij[16,15]=b_i[15]*(1-c_i[15])/b_i[16]

    def step2(vars): 
        eqs=[]
        a_sol=[]
        for j in range(10,15):
            array_vander=c_i[j+1:]
            if j==14:
                c_sol=[b_i[j]*(1.0-c_i[j])  , b_i[j]*(1.0-c_i[j]**2) /2]
                n_col=len(c_sol)
                b_daig=np.diag(b_i[j+1:])

            elif j>=12 and j<=13:
                c_sol=[b_i[j]*(1.0-c_i[j])  , b_i[j]*(1.0-c_i[j]**2) /2,  b_i[j]*(1.0-c_i[j]**3) /3]
                n_col=len(c_sol)
                b_daig=np.diag(b_i[j+1:])

                if j==12:
                    id=[13,14,16]
                    array_vander=c_i[id]
                    b_daig=np.diag(b_i[id])

            elif j<=11:
                c_sol=[b_i[j]*(1.0-c_i[j])  , b_i[j]*(1.0-c_i[j]**2) /2,  b_i[j]*(1.0-c_i[j]**3) /3,  b_i[j]*(1.0-c_i[j]**4) /4]
                n_col=len(c_sol)
                if j==10:
                    id=[11,12,13,16]
                    array_vander=c_i[id]
                    b_daig=np.diag(b_i[id])
                elif j==11:
                    id=[12,13,14,16]
                    array_vander=c_i[id]
                    b_daig=np.diag(b_i[id])



            bc_vander=np.transpose(np.dot(b_daig,np.vander(array_vander, increasing=True)))

            a_sol.append(np.dot(np.linalg.inv(bc_vander),c_sol))
       
        return a_sol
    solution_a=step2(initial_guess)

    for m,i in enumerate(range(10,15)):
        if i==14:
            a_ij[i+1:,i]=solution_a[m]

        elif  i==12 or i==13:

            if i==12:
                id=[13,14,16]
                a_ij[id,i]=solution_a[m]
            if i==13:
                a_ij[i+1:,i]=solution_a[m]

        elif i<=11:
            if i==10:
                id=[11,12,13,16]
                a_ij[id,i]=solution_a[m]
            elif i==11:
                id=[12,13,14,16]
                a_ij[id,i]=solution_a[m]
    initial_guess = np.random.rand(16)
    print(a_ij)


    #step3
    a_ij[14,5]=-a_ij[14,12]
    a_ij[14,6] =- a_ij[14,13]
    a_ij[15,2] =- a_ij[15,14]
    a_ij[16,1] =- a_ij[16,15]
    a_ij[16,2] =- a_ij[16,14]

    a_ij[12,3] = a_ij[5,3]
    a_ij[12,4] = a_ij[5,4]
    a_ij[13,3] = a_ij[6,3]
    a_ij[13,4] = a_ij[6,4]
    a_ij[14,1] = a_ij[2,1]

    def step3(vars):
        eqs=[]
        a_ij[[14,15],0]=vars
        k=1
        for i in [14,15]:
            eqs.append(sum(a_ij[i, :i] * (c_i[:i] ** (k - 1))) - (c_i[i] ** k) / k)
        return eqs
    initial_guess = np.random.rand(2)
    a_ij[[14,15],0]=root(step3,initial_guess).x
    def step4_1(vars):
        eqs=[]
        count=0
        for k in range(0,5):
            suma_ext=0

            for i in range(0,s):
                suma_int=0
                if i in [10,11,12,13,16]:
                    if i==16:
                        suma_int += vars[-1]
                    else:
                        suma_int += vars[i%10]
                else:
                    for j in range(0,i):    
                        suma_int += a_ij[i,j]*c_i[j]**5
                suma_ext += b_i[i]*c_i[i]**k*suma_int

            eqs.append(suma_ext-1.0/((7.0+k)*6.0)) 
        return eqs
    
    initial_guess=np.random.rand(5)
    print(root(step4_1,initial_guess).x)
    def step4_2(temp_a):
        a_sol=[]
        for i in [10,11,12,13,16]:
            array_vander=c_i[[0,5,6,7,8,9]]
            if i<=11 :
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,c_i[i]**4/4,c_i[i]**5/5,temp_a[i%10]]
            elif i==12:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,sum(a_ij[5,:5]*c_i[5]**4),c_i[i]**5/5,temp_a[i%10]]
            elif i==13:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,sum(a_ij[6,:6]*c_i[6]**4),c_i[i]**5/5,temp_a[i%10]]
            elif i==16:
                c_sol=[c_i[i],c_i[i]**2/2,c_i[i]**3/3,c_i[i]**4/4,c_i[i]**5/5,temp_a[4]]

            c_vander=np.transpose(np.vander(array_vander, increasing=True))

            a_sol.append(np.dot(np.linalg.inv(c_vander),c_sol))
        return a_sol
    print(step4_2(root(step4_1,initial_guess).x))



    return  c_i,b_i

#s = 17;r=10
s = 17; r = 10
c,b = runge_kutta_10th_order(s,r)


print("\nCoeficientes c_i:")
print(c)
print("\nCoeficientes b_i:")
print(b)
#Resultados correctos con la permutacion correcta



