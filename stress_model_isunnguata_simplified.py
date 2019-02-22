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

dt_float = 1e-3
dt = df.Constant(dt_float)

mesh = df.Mesh('data/isunnguata_sermia.xml')
nhat = df.FacetNormal(mesh)

E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
Q_cg = df.FunctionSpace(mesh,E_cg)

E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)

E = df.MixedElement([E_cg,E_cg])
V = df.FunctionSpace(mesh,E)

B = df.Function(Q_cg,'data/Bed/B_c.xml')
H = df.Function(Q_cg,'data/Thk/H_c.xml')

edgefunction = df.MeshFunction('size_t',mesh,1)

mesh.init()
for f in df.facets(mesh):
    if f.exterior():
        print('here')
        edgefunction[f] = 2
        if H(f.midpoint().x(),f.midpoint().y())<200:
            edgefunction[f]=1

u_b = df.Function(Q_dg,'data/Vel_x/u_d.xml')
v_b = df.Function(Q_dg,'data/Vel_y/v_d.xml')

U = df.Function(V)
Lambda = df.Function(V)
Phi = df.TestFunction(V)
dU = df.TrialFunction(V)

u,v = df.split(U)
lamda_x,lamda_y = df.split(Lambda)

phi = df.TestFunction(Q_cg)

u0 = df.Function(Q_cg)
v0 = df.Function(Q_cg)

Be = df.Constant(215443.)
eta = Be/2.*(u.dx(0)**2 + v.dx(1)**2 + u.dx(0)*v.dx(1) + 0.25*(u.dx(1) + v.dx(0))**2 + 1e-5)**((1-n)/(2*n))
beta2 = df.Function(Q_cg)
beta2.vector()[:] = 300

S = B+H

N = 0.2*H*rho_i*g

tau_bx = -beta2*u#(abs(N)+10)**(1./n)*(u**2 + v**2 + 1e-1)**((1./n - 1.)/2.)*u
tau_by = -beta2*v#(abs(N)+10)**(1./n)*(u**2 + v**2 + 1e-1)**((1./n - 1.)/2.)*v


R_u_body = (-lamda_x.dx(0)*2*eta*H*(2*u.dx(0) + v.dx(1)) - lamda_x.dx(1)*eta*H*(u.dx(1) + v.dx(0)) + lamda_x*tau_bx - lamda_x*rho_i*g*H*S.dx(0))*df.dx# phi_x.dx(0)*rho_i*g*(H**2 - B**2)/2. - phi_x*rho_i*g*(B+H)*B.dx(0))*df.dx 
R_u_boundary = (lamda_x*2*eta*H*(2*u.dx(0) + v.dx(1))*nhat[0] + lamda_x*eta*H*(u.dx(1) + v.dx(0))*nhat[1])*df.ds(subdomain_data=edgefunction)(2)# - (phi_x*rho_i*g*(H**2 - B**2)/2*nhat[0])*df.ds

R_v_body = (-lamda_y.dx(0)*eta*H*(u.dx(1) + v.dx(0)) - lamda_y.dx(1)*2*eta*H*(u.dx(0) + 2*v.dx(1)) + lamda_y*tau_by - lamda_y*rho_i*g*H*S.dx(1))*df.dx# phi_y.dx(1)*rho_i*g*(H**2 - B**2)/2. - phi_y*rho_i*g*(B+H)*B.dx(1))*df.dx 
R_v_boundary = (lamda_y*eta*H*(u.dx(1) + v.dx(0))*nhat[0] + lamda_y*2*eta*H*(u.dx(0) + 2*v.dx(1))*nhat[1])*df.ds(subdomain_data=edgefunction)(2)# - (phi_y*rho_i*g*(H**2 - B**2)/2*nhat[1])*df.ds

ones = df.Function(Q_cg)
ones.vector()[:] = 1
area = df.assemble(ones*df.dx)
gamma = df.Constant(100000)

misfit = ((u-u_b)**2 + (v-v_b)**2)*df.dx 
prior = gamma*(beta2.dx(0)**2 + beta2.dx(1)**2)*df.dx

F = R_u_body + R_u_boundary + R_v_body + R_v_boundary + misfit + prior
R = df.derivative(F,Lambda,Phi)
J = df.derivative(R,U,dU)

import ufl
R_adjoint = df.derivative(F,U,Phi)
R_adjoint_l = ufl.replace(R_adjoint,{Lambda:dU})
A_adjoint = df.lhs(R_adjoint_l)
b_adjoint = df.rhs(R_adjoint_l)

gradient = df.derivative(F,beta2,phi)


# Nonlinear Problem
problem = df.NonlinearVariationalProblem(R,U,J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relaxation_parameter'] = 1.0
solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3
solver.parameters['newton_solver']['error_on_nonconvergence'] = False
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['maximum_iterations'] = 20
solver.parameters['newton_solver']['report'] = True

def _I_fun(b):
    beta2.vector().set_local(b)
    solver.solve()
    I = df.assemble(misfit)
    return I

def _J_fun(b):
    beta2.vector().set_local(b)
#    solver.solve()
    df.solve(A_adjoint == b_adjoint,Lambda)
    dg = df.assemble(gradient)
    return dg.get_local()


from scipy.optimize import fmin_l_bfgs_b

step = 100
b = beta2.vector().get_local()
for i in range(1000):
    print(np.sqrt(_I_fun(b)/area))
    grad = _J_fun(b)
    grad/=np.linalg.norm(grad)
    b -= step*grad

#fmin_l_bfgs_b(_I_fun,beta2.vector().get_local(),fprime=_J_fun)
"""
assigner_inv = df.FunctionAssigner([Q_cg,Q_cg],V)
assigner     = df.FunctionAssigner(V,[Q_cg,Q_cg])

t = 0.0
t_end = 200

ux_file = df.File('ux.pvd')
uy_file = df.File('uy.pvd')

assigner.assign(U,[u0,v0])
 
solver.solve()
assigner_inv.assign([u0,v0],U)

ux_file << u0
uy_file << v0
"""

#docker run --rm -it -v "$PWD":/app -w /app --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable 
             
