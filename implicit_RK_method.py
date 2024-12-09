import numpy as np
from scipy.linalg import lu, solve_triangular

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
        k=quasi_newton(c,A,b,f,D_yf,h,t_n,y_n,k,TOL,max_its)

    y_new = y_n + h*np.dot(k, b)
    k_end=k[:,-1]

    return y_new, k_end

    


def quasi_newton(c, A, b,f,D_yf,h,t_n,y_n,k,TOL, max_its):
    
    # define the stage
    s,d=len(b),len(y_n)

    #initialize stopping criterion paras
    norm_dk_m = 1
    it = 0
    e_m = np.Inf

    dyf = D_yf(t_n,y_n)
    DG = np.eye(s * d) - h * np.kron(A, dyf)
    P, L, U = lu(DG)

    res = np.zeros(d*s)

    while abs(e_m)>TOL:

        for i in range(s):

            ti=t_n+c[i]*h
            yi=y_n+h*np.dot(k[:,:],A[i,:])
            res[i*d:(i+1)*d] = - (k[:,i] - f(ti, yi))

        res = np.dot(np.transpose(P), res)
        res = solve_triangular(L, res, lower=True,  unit_diagonal=True,  overwrite_b=True)
        delta_k  = solve_triangular(U, res, lower=False, unit_diagonal=False, overwrite_b=True)
        
        delta_k = np.reshape(delta_k, (d,s), order='F')

        k+=delta_k
        
        # Compute theta_k and check stopping criterion
        norm_dk_m1=norm_dk_m
        norm_dk_m=np.linalg.norm(delta_k, 'fro')

        if it == 0:
            # Skip e_m calculation in the first iteration
            e_m = np.inf
        else:
            # Limit theta_m to prevent divide-by-zero
            theta_m = norm_dk_m / norm_dk_m1 if norm_dk_m1 > 1e-12 else 1.0
            if abs(1 - theta_m) < 1e-6:  
                print("Stopping due to small denominator in theta_m.")
                break
            e_m = norm_dk_m * theta_m / (1 - theta_m)

        it += 1

        if it > max_its:
            raise RuntimeError("Newton iteration did not converge within max iterations.")


    print(f"Newton Method coverges after {it} iterations")

    return k





            






