from vpython import *
import math
import random
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


scene = canvas()

tau = 0
dtau = 1

x=0
y=0
z=0

u=np.pi/4
v=0
R = 3
r = 1

path = sphere(radius=0.1,
              pos=vector((R + (r) * np.cos(v)) * np.cos(u), (R + (r) * np.cos(v)) * np.sin(u), (r) * np.sin(v)),
              color=color.red,
              make_trail= True,
              trail_type= 'curve')

udot = 0.01
vdot = 0.1

uddot= 2*r*np.sin(v)*udot*vdot/(R+r*np.cos(v))
vddot= -(np.sin(v)*(R+r*np.cos(v))*udot**2)/r

pts = 0.0*np.ones((3, 301))

fig = plt.figure(figsize=(4, 4))

ax = fig.add_subplot(111, projection='3d')


while tau<300:

    u += udot * dtau
    v += vdot * dtau

    #print(u, v)

    uddot = 2 * r * np.sin(v) * udot * vdot / (R + r * np.cos(v))
    vddot = -(np.sin(v) * (R + r * np.cos(v)) * udot ** 2) / r

    udot += uddot * dtau
    vdot += vddot * dtau


    x = (R + (r) * np.cos(v)) * np.cos(u)
    y = (R + (r) * np.cos(v)) * np.sin(u)
    z = (r) * np.sin(v)
    #print(x, y, z)

    pts[0][tau] = x
    pts[1][tau] = y
    pts[2][tau] = z


    tau += dtau
    print(tau)





xs = pts[0,:]
ys= pts[1,:]
zs = pts[2,:]

ax.scatter(xs, ys, zs)

plt.show()


