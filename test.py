import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from numba import jit

def rhs(t, y):
    alpha = 0.1
    n = lambda z : 2 - alpha*z
    dndz = lambda z : - alpha

    return np.array([1/np.tan(y[1]), -dndz(y[0])/n(y[0])])

import rkc

t0 = 0.
tf = 10.
y0 = np.array([0., np.pi/4])
#y0 = 0.
h=1e-2

tstart = time.time()
ts, ys = rkc.solveode(rhs, t0, tf, y0, h)
print('my solver:', time.time()-tstart)
#print(ts, ys)

tstart = time.time()
sol = solve_ivp(rhs, [t0, tf], y0, method='RK45', max_step = 1e-2)
print('their solver:', time.time()-tstart)

plt.plot(sol.t, sol.y[0, :], '-.')
plt.plot(ts, ys[0, :], '.')

#print(max(np.abs(ys[0, :]-np.sin(ts))))
#print(max(np.abs(sol.y[0, :] - np.sin(sol.t))))

plt.show()
