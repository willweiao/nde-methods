import numpy as np
from scipy.linalg import lu_factor, lu_solve

def is_explicit_butcher(A):
    s=A.shape[0]

    for i in range(s):
        for j in range(i,s):
            
            if A[i,j]!=0:
                
                return False
    
    return True


def implicit_rk_step(c, A, b, f, D_yf, t_n, y_n, h,k0,TOL,max_its):
    
    s=A.shape[0]

    # Check if the butcher table implies an explicit method or not
    explicit=is_explicit_butcher(A)
    
    k=np.transpose(np.tile(k0, (s, 1)))

    if explicit==True:
        k[:,0] = f(t_n+c[0]*h, y_n)
        for i in range(1,s):
            k[:,i] = f(t_n+c[i]*h, y_n + h*np.dot(k[:,:i], A[i,:i]))
    else:
        k=quasi_newton(c,A,b,f,D_yf,h,t_n,y_n,TOL,max_its)

    y_new = y_n + h*np.dot(k, b)
    k_end=k[:,-1]

    return y_new, k_end

    


def quasi_newton(c, A, b,f,D_yf,h,t_n,y_n,TOL, max_its):
    
    # define the stage
    s,d=len(b),len(y_n)

    #initialize stopping criterion paras
    norm_dk_m = 1
    it = 0
    e_m = np.Inf

    K=np.zeros((s,d))

    while abs(e_m)>TOL:

        dyf = D_yf(t_n,y_n)
        DG = np.eye(s*d)- np.kron(h*A, dyf)
        lu, piv = lu_factor(DG) 
      
        G=np.zeros((s,d))

        for i in range(s):

            ti=t_n+c[i]*h
            yi=y_n+h*np.dot(A[i],K)
            G[i] = K[i] - f(ti, yi)

        # Reshape G to a 1D vector for solving
        G_flat = G.flatten()

        # Solve for delta_K using LU decomposition
        delta_K_flat = lu_solve((lu, piv), -G_flat)

        # Reshape delta_K back to original shape
        delta_K = delta_K_flat.reshape(s, d)
        
        # Compute theta_k and check stopping criterion
        norm_dk_m1=norm_dk_m
        norm_dk_m=np.linalg.norm(delta_K, 'fro')

        if it == 0:
            # Skip e_m calculation in the first iteration
            e_m = np.inf
        else:
            # Limit theta_m to prevent divide-by-zero
            theta_m = min(norm_dk_m / norm_dk_m1, 0.99)
            e_m = norm_dk_m * theta_m / (1 - theta_m)

        
        it += 1

        if it > max_its:
            print("Newton iteration did not converge")
            break

    print(f"Newton Method coverges after {it} iterations")

    return K





            






