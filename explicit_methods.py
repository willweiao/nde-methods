import numpy as np


def explicit_euler(u_n, t_n, dt, f):
    u_new=u_n + dt* f(t_n, u_n)
    return u_new

def modified_euler(u_n, t_n, dt, f):
    t_x=t_n+dt/2
    u_x= u_n+(dt/2)*f(t_n, u_n)
    u_new=u_n+dt*f(t_x, u_x)
    return u_new

def heun(u_n, t_n, dt, f):
    t_x=t_n+dt
    u_x=u_n+dt*f(t_n,u_n)
    u_new=u_n+(dt/2)*(f(t_n,u_n)+f(t_x,u_x))
    return u_new



    