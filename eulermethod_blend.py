
from bpy import context, data, ops
from math import cos, pi, sin, tan
from random import TWOPI, randint, uniform

# Create a bezier circle and enter edit mode.
ops.curve.primitive_bezier_circle_add(radius=1.0,
                                      location=(0.0, 0.0, 0.0),
                                      enter_editmode=True)

# Subdivide the curve by a number of cuts, giving the
# random vertex function more points to work with.
ops.curve.subdivide(number_cuts=160)



# Cache a reference to the curve.
curve = context.active_object

# Locate the array of bezier points.
bez_points = curve.data.splines[0].bezier_points



tau = 0
dtau = 1

x=0
y=0
z=0

u = pi/4
v = 0
R = 4
r = 1

#path = sphere(radius=0.1,
#              pos=vector((R + (r) * np.cos(v)) * np.cos(u), (R + (r) * np.cos(v)) * np.sin(u), (r) * np.sin(v)),
#              color=color.red,
#              make_trail= True,
#              trail_type= 'curve')

udot = 0.01
vdot = 0.1

uddot= 2*r*sin(v)*udot*vdot/(R+r*cos(v))
vddot= -(sin(v)*(R+r*cos(v))*udot**2)/r


sz = len(bez_points)
for i in range(0, sz, 1):

    u += udot * dtau
    v += vdot * dtau

#    print(u, v)

    uddot = 2 * r * sin(v) * udot * vdot / (R + r * cos(v))
    vddot = -(sin(v) * (R + r * cos(v)) * udot ** 2) / r

    udot += uddot * dtau
    vdot += vddot * dtau


    bez_points[i].co.x = (R + (r) * cos(v)) * cos(u)
    bez_points[i].co.y = (R + (r) * cos(v)) * sin(u)
    bez_points[i].co.z = (r) * sin(v)
    #print(x, y, z)

    tau += dtau
#    print(tau)


# Scale the curve while in edit mode.
#ops.transform.resize(value=(2.0, 2.0, 3.0))

# Return to object mode.
ops.object.mode_set(mode='OBJECT')

# Store a shortcut to the curve object's data.
obj_data = context.active_object.data



# Smoothness of the segments on the curve.
obj_data.resolution_u = 200
obj_data.render_resolution_u = 320

# Return to object mode.
ops.object.mode_set(mode='OBJECT')