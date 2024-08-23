import numpy as np
N=11
M=np.ones((N,N))
M*=range(-int((N-1)/2),int((N+1)/2))
for n in range(N):
    M[n,:]**=n

b=np.zeros(N)
b[1]+=1
A=np.dot((np.linalg.inv(M)),b)
print(M[:,:])
print(b)
A[int((N-1)/2)]=0.0
print(A)
fh = open("coeff_det_centrada.npy", "bw")
A.tofile(fh)


N=11
M=np.ones((N,N))
M*=range(0,N)
for n in range(N):
    M[n,:]**=n

b=np.zeros(N)
b[1]+=1
A=np.dot((np.linalg.inv(M)),b)
print(M[:,:])
print(b)
print(A)
fh = open("coeff_det_adelantada.npy", "bw")
A.tofile(fh)

N=11
M=np.ones((N,N))
M*=range(-N+1,1)
for n in range(N):
    M[n,:]**=n

b=np.zeros(N)
b[1]+=1
A=np.dot((np.linalg.inv(M)),b)
print(M[:,:])
print(b)
print(A)
fh = open("coeff_det_atrasada.npy", "bw")
A.tofile(fh)