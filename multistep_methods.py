import numpy as np


def q_step_method(q, A, b, f, h, ind, t_arr, y_arr):
    t_qstep=t_arr[ind-q: ind]
    y_qstep=y_arr[ind-q: ind]
    y_arr[ind+1]=np.dot(A[ind-q: ind,ind-q: ind],y_qstep)+h*np.dot(b[ind-q: ind], f(t_qstep, y_qstep))
    return y_arr

def explicit_RK4(f, t_arr, y_arr, h, N):
    for i in range(N):
        y_n=y_arr[i]
        k1= f(t_arr[i], y_arr[i])
        k2 = f(t_arr[i] + h / 2, y_arr[i]+ h / 2 * k1)
        k3 = f(t_arr[i] + h / 2, y_arr[i] + h / 2 * k2)
        k4 = f(t_arr[i] + h, y_arr[i] + h * k3) 

        t_arr[i+1] = t_arr[i] + h
        y_arr[i+1] = y_arr[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_arr, y_arr