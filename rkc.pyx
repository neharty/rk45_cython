import numpy as np
cimport numpy as np
cimport cython

def solveode(rhs, double t0, double tf, y0, double tol):
    if type(y0) == float or type(y0) == int:
        return solveode_s(rhs, t0, tf, y0, tol)
    else:
        return solveode_v(rhs, t0, tf, y0, tol)

# copying from scipy RK45 defns
cdef np.ndarray[double, ndim=2] rk_step_s(object rhs, double t, float yn, double h):
    #scalar step
    cdef double k1 = rhs(t, yn)
    cdef double k2 = rhs(t + h/5.0, yn + h*k1/5.0) 
    cdef double k3 = rhs(t + 3*h/10.0, yn + h*(3*k1/40 + 9*k2/40.0))
    cdef double k4 = rhs(t + 4*h/5.0, yn + h*(44*k1/45 - 56*k2/15 + 32*k3/9))
    cdef double k5 = rhs(t + 8*h/9.0, yn + h*(19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729))
    cdef double k6 = rhs(t + h, yn + h*(9017*k1/3168 -355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656))
    cdef double k7 = rhs(t + h, yn + h*(35*k1/384 + 500*k3/1113 + 125*k4/192 -2187*k5/6784 + 11*k6/84))
    cdef double y1 = yn + h*(35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84)
    cdef double y2 = yn + h*(5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + k7/40)
    cdef double err = y2 - y1

    return np.array([y1, err])

cdef np.ndarray[double, ndim = 2] rk_step_v(object rhs, double t, np.ndarray[double, ndim = 1] yn, double h):
    #vector step
    cdef np.ndarray[double, ndim=1] k1 = rhs(t, yn)
    cdef np.ndarray[double, ndim=1] k2 = rhs(t + h/5.0, yn + h*k1/5.0)
    cdef np.ndarray[double, ndim=1] k3 = rhs(t + 3*h/10.0, yn + h*(3*k1/40 + 9*k2/40.0))
    cdef np.ndarray[double, ndim=1] k4 = rhs(t + 4*h/5.0, yn + h*(44*k1/45 - 56*k2/15 + 32*k3/9))
    cdef np.ndarray[double, ndim=1] k5 = rhs(t + 8*h/9.0, yn + h*(19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729))
    cdef np.ndarray[double, ndim=1] k6 = rhs(t + h, yn + h*(9017*k1/3168 -355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656))
    cdef np.ndarray[double, ndim=1] k7 = rhs(t + h, yn + h*(35*k1/384 + 500*k3/1113 + 125*k4/192 -2187*k5/6784 + 11*k6/84))
    cdef np.ndarray[double, ndim=1] y1 = yn + h*(35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84)
    cdef np.ndarray[double, ndim=1] y2 = yn + h*(5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + k7/40)
    cdef double err = np.max(np.abs(y2 - y1))

    return np.array([y1, err])


cdef np.ndarray[double, ndim=2] solveode_s(object rhs, double t0, double tf, double y0, double tol):
    cdef double t = t0
    cdef double h = min(0.5*tol**(1/5.), (tf-t0)/3)
    cdef int n = 0
    cdef double ynew, err, maxerr, q
    cdef double yold = y0

    while t < tf:
        ynew, err = rk_step_s(rhs, t, yold, h)
        maxerr = tol*(1 + ynew)
        
        if err < maxerr:
            t = t + h
            yold = ynew
            n+=1
        else:
            q = 0.8*(maxerr/err)**(1.0/5)
            q = min(q, (tf-t0)/3)
            h = min(q*h, h)
    
    cdef np.ndarray[double, ndim=1] ts = np.zeros(n+1)
    ts[0] = t0
    
    cdef np.ndarray[double, ndim=1] ys = np.zeros(n+1)
    ys[0] = y0
    
    t = t0
    n = 0

    while t < tf:
        ynew, err = rk_step_s(rhs, t, ys[n], h)
        maxerr = tol*(1 + ynew)

        if err < maxerr:
            t = t + h
            ts[n+1] = t
            ys[n+1] = ynew
            n+=1
        else:
            q = 0.8*(maxerr/err)**(1.0/5)
            q = min(q, (tf-t0)/3)
            h = min(q*h, h)

    return np.array([ts, ys])

cdef np.ndarray[double, ndim=2] solveode_v(object rhs, double t0, double tf, np.ndarray[double, ndim=1] y0, double tol):
    cdef int ydim = len(y0)
    cdef double t = t0
    cdef double h = min(0.5*tol**(1/5.), (tf-t0)/3)
    cdef int n = 0
    cdef double err, maxerr, q
    cdef np.ndarray[double, ndim=1] yold = y0, ynew

    while t < tf:
        ynew, err = rk_step_v(rhs, t, yold, h)
        maxerr = tol*(1 + np.max(ynew))

        if err < maxerr:
            t = t + h
            yold[:] = ynew[:]
            n+=1
        else:
            q = 0.8*(maxerr/err)**(1.0/5)
            q = min(q, (tf-t0)/3)
            h = min(q*h, h)

    cdef np.ndarray[double, ndim=1] ts = np.zeros(n+1)
    ts[0] = t0
    
    cdef np.ndarray[double, ndim=2] ys = np.zeros((ydim, n+1))
    ys[:, 0] = y0
    
    t = t0
    n=0

    while t < tf:
        ynew, err = rk_step_v(rhs, t, ys[:, n], h)
        maxerr = tol*(1 + np.max(ynew))

        if err < maxerr:
            t = t + h
            ts[n+1] = t
            ys[:, n+1] = ynew[:]
            n+=1
        else:
            q = 0.8*(maxerr/err)**(1.0/5)
            q = min(q, (tf-t0)/3)
            h = min(q*h, h)

    return np.array([ts, ys])
