import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['allow_extrapolation'] = True

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)

spy = 60**2*24*365
L = 100000.
n = 3.0
g = 9.81

rho_i = 917.
rho_w = 1000.0

h_r = df.Constant(0.1)
l_r = df.Constant(2.0)
A = df.Constant(2./9.*3.5e-17)

u_b = df.Constant(1e-6*spy)
alpha = df.Constant(5./4.)
beta = df.Constant(3./2.)

k = df.Constant(.005*spy)

e_v = df.Constant(1e-3)

dt_float = 1e-1
dt = df.Constant(dt_float)

mesh = df.Mesh('data/isunnguata_sermia.xml')
nhat = df.FacetNormal(mesh)

E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
Q_cg = df.FunctionSpace(mesh,E_cg)

E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)

E = df.MixedElement([E_cg,E_cg,E_dg])
V = df.FunctionSpace(mesh,E)

B_dg = df.Function(Q_dg,'data/Bed/B_d.xml')
B = df.Function(Q_cg,'data/Bed/B_c.xml')

H0 = df.Function(Q_dg,'data/Thk/H_d.xml')
H0_c = df.Function(Q_cg,'data/Thk/H_c.xml')


edgefunction = df.MeshFunction('size_t',mesh,1)

mesh.init()
for f in df.facets(mesh):
    if f.exterior():
        print('here')
        edgefunction[f] = 2
        if H0_c(f.midpoint().x(),f.midpoint().y())<200:
            edgefunction[f]=1

u_b = df.Function(Q_dg,'data/Vel_x/u_d.xml')
v_b = df.Function(Q_dg,'data/Vel_y/v_d.xml')

U = df.Function(V)
Phi = df.TestFunction(V)
dU = df.TrialFunction(V)

u,v,H = df.split(U)
phi_x,phi_y,xsi = df.split(Phi)

adot = df.Function(Q_dg,'data/SMB/smb_d.xml')
adot.vector()[:] = adot.vector()[:]/1000.

u0 = df.Function(Q_cg)
v0 = df.Function(Q_cg)

Be = df.Constant(215443.)
eta = Be/2.*(u.dx(0)**2 + v.dx(1)**2 + u.dx(0)*v.dx(1) + 0.25*(u.dx(1) + v.dx(0))**2 + 1e-5)**((1-n)/(2*n))
beta2 = df.Constant(100)

N = 0.2*H*rho_i*g

tau_bx = -beta2*(abs(N)+10)**(1./n)*(u**2 + v**2 + 1e-1)**((1./n - 1.)/2.)*u
tau_by = -beta2*(abs(N)+10)**(1./n)*(u**2 + v**2 + 1e-1)**((1./n - 1.)/2.)*v


R_u_body = (-phi_x.dx(0)*2*eta*H*(2*u.dx(0) + v.dx(1)) - phi_x.dx(1)*eta*H*(u.dx(1) + v.dx(0)) + phi_x*tau_bx + phi_x.dx(0)*rho_i*g*(H**2 - B**2)/2. - phi_x*rho_i*g*(B+H)*B.dx(0))*df.dx 
R_u_boundary = (phi_x*2*eta*H*(2*u.dx(0) + v.dx(1))*nhat[0] + phi_x*eta*H*(u.dx(1) + v.dx(0))*nhat[1])*df.ds(subdomain_data=edgefunction)(3) - (phi_x*rho_i*g*(H**2 - B**2)/2*nhat[0])*df.ds

R_v_body = (-phi_y.dx(0)*eta*H*(u.dx(1) + v.dx(0)) - phi_y.dx(1)*2*eta*H*(u.dx(0) + 2*v.dx(1)) + phi_y*tau_by + phi_y.dx(1)*rho_i*g*(H**2 - B**2)/2. - phi_y*rho_i*g*(B+H)*B.dx(1))*df.dx 
R_v_boundary = (phi_y*eta*H*(u.dx(1) + v.dx(0))*nhat[0] + phi_y*2*eta*H*(u.dx(0) + 2*v.dx(1))*nhat[1])*df.ds(subdomain_data=edgefunction)(3) - (phi_y*rho_i*g*(H**2 - B**2)/2*nhat[1])*df.ds

H_avg = 0.5*(H('+') + H('-'))
H_jump = H('+')*nhat('+') + H('-')*nhat('-')

xsi_jump = (xsi('-')*nhat('-') + xsi('+')*nhat('+'))

uu = df.as_vector([u,v])
un = df.dot(uu,nhat)
un_avg = 0.5*(abs(un)('+') + abs(un)('-'))

#Upwind
uH = uu('+')*H_avg + 0.5*abs(un_avg)*H_jump

R_H = ((H-H0)/dt - adot)*xsi*df.dx + df.dot(uH,xsi_jump)*df.dS + xsi*df.dot(uu*H,nhat)*df.ds(subdomain_data=edgefunction)(1)

R = R_u_body + R_u_boundary + R_v_body + R_v_boundary + R_H
J = df.derivative(R,U,dU)

# Nonlinear Problem
problem = df.NonlinearVariationalProblem(R,U,J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3
solver.parameters['newton_solver']['error_on_nonconvergence'] = False
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['maximum_iterations'] = 20
solver.parameters['newton_solver']['report'] = True

assigner_inv = df.FunctionAssigner([Q_cg,Q_cg,Q_dg],V)
assigner     = df.FunctionAssigner(V,[Q_cg,Q_cg,Q_dg])

t = 0.0
t_end = 200

ux_file = df.File('ux.pvd')
uy_file = df.File('uy.pvd')
H_file = df.File('H.pvd')

assigner.assign(U,[u0,v0,H0])
while t<t_end:
 
    solver.solve()
    assigner_inv.assign([u0,v0,H0],U)

    ux_file << u0
    uy_file << v0
    H_file << H0

    t+=dt_float


#docker run --rm -it -v "$PWD":/app -w /app --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable 
             
