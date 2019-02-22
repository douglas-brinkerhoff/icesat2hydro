import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['allow_extrapolation'] = True

spy = 60**2*24*365
L = 100000.
n = 3.0
g = 9.81

rho_i = 917.
rho_w = 1000.0

h_r = df.Constant(0.2)
l_r = df.Constant(5.0)
A = df.Constant(2./9.*3.5e-17)

alpha = df.Constant(5./4.)
beta = df.Constant(3./2.)

k = df.Constant(.01*spy)

e_v = df.Constant(1e-4)

dt_float = 1e-2
dt = df.Constant(dt_float)

mesh = df.Mesh('data/isunnguata_sermia.xml')
nhat = df.FacetNormal(mesh)


E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
Q_cg = df.FunctionSpace(mesh,E_cg)

E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)

E = df.MixedElement([E_dg,E_dg,E_dg,E_dg])
V = df.FunctionSpace(mesh,E)

class B_ex(df.UserExpression):
    def eval(self,values,x):
        values[0] = 0#2*np.cos(6*np.pi*x[0]/L) + 1e-4*x[0]

B = df.Function(Q_dg,'data/Bed/B_d.xml')
B_cg = df.Function(Q_cg,'data/Bed/B_c.xml')

class H_ex(df.UserExpression):
    def eval(self,values,x):
        values[0] = 6*(np.sqrt(x[0]+5e3) - np.sqrt(5e3)) + 1.

H = df.Function(Q_dg,'data/Thk/H_d.xml')
H_cg = df.Function(Q_cg,'data/Thk/H_c.xml')

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)
def Softplus(a,b,alpha=1): return Max(a,b) + 1./alpha*df.ln(1 + df.exp(-abs(a-b)*alpha))

edgefunction = df.MeshFunction('size_t',mesh,1)

mesh.init()
for f in df.facets(mesh):
    if f.exterior():
        print('here')
        edgefunction[f] = 2
        if H_cg(f.midpoint().x(),f.midpoint().y())<200:
            edgefunction[f]=1

u_b = df.Function(Q_dg,'data/Vel_x/u_d.xml')
v_b = df.Function(Q_dg,'data/Vel_y/v_d.xml')

ub = df.sqrt(u_b**2 + v_b**2 + 1e-3)
#ub = df.Constant(1e-6*spy)

U = df.Function(V)
Phi = df.TestFunction(V)
dU = df.TrialFunction(V)

h_w,h,phi_x,phi_y = df.split(U)
xsi,chi,w_x,w_y = df.split(Phi)

h0_w = df.Function(Q_dg)
h0 = df.Function(Q_dg)
phi0_x = df.Function(Q_dg)
phi0_y = df.Function(Q_dg)

du = df.TrialFunction(Q_dg)

theta = df.Constant(1.0)
h_mid = theta*h_w + (1-theta)*h0_w

P_0 = rho_i*g*H
phi_m = rho_w*g*B
phi = rho_w*g*B + rho_w*g*Softplus(h_mid, 1./e_v*(h_mid - h) + h,alpha=0.01)
P_w = rho_w*g*Softplus(h_mid, 1./e_v*(h_mid - h) + h,alpha=0.01)
N = P_0 - P_w

m = df.Function(Q_dg,'data/SMB/smb_d.xml')
m.vector()[:] = -m.vector()[:]/1000.
m0 = df.project(Max(m,0),Q_dg)
m = df.Function(Q_dg)

df.File('m.pvd') << m

h0_w.vector()[:] = 0.21
h0.vector()[:] = 0.2
phi0_x.vector()[:] = 0

grad_phi = df.as_vector([phi_x,phi_y])
w_vec = df.as_vector([w_x,w_y])

phi_avg = 0.5*(grad_phi('+') + grad_phi('-'))
phi_jump = df.dot(grad_phi('+'),nhat('+')) + df.dot(grad_phi('-'),nhat('-'))

h_w_avg = 0.5*(Min(h_mid('+'),h('+')) + Min(h_mid('-'),h('-')))
h_w_jump = Min(h_mid,h)('+')*nhat('+') + Min(h_mid,h)('-')*nhat('-')

xsi_jump = (xsi('-')*nhat('-') + xsi('+')*nhat('+'))
w_jump = df.dot(nhat('+'),w_vec('+')) + df.dot(nhat('-'),w_vec('-'))

u = -k*(Min(h_mid,h)**2 + 1e-3)**((alpha-1)/2.)*(df.dot(grad_phi,grad_phi) + 1e0)**(beta/2.-1)*grad_phi

un = df.dot(u,nhat)
un_avg = 0.5*(abs(un)('+') + abs(un)('-'))

#Upwind
uH = df.avg(u*Min(h_mid,h)) + 0.5*abs(un_avg)*h_w_jump

R_hw = ((h_w-h0_w)/dt - m)*xsi*df.dx + df.dot(uH,xsi_jump)*df.dS + xsi*df.dot(u*Min(h_mid,h),nhat)*df.ds(subdomain_data=edgefunction)(1)


R_gradphi = df.dot(w_vec,grad_phi)*df.dx - w_jump*df.avg(phi)*df.dS - df.dot(w_vec,nhat)*phi*df.ds(subdomain_data=edgefunction)(2) - df.dot(w_vec,nhat)*rho_w*g*(B_cg+H_cg)*df.ds(subdomain_data=edgefunction)(1) 

O = Max(ub*(h_r - h)/l_r,1e-3)
C = A*h*N**n

R_h = ((h - h0)/dt - O + C)*chi*df.dx

R = R_hw + R_gradphi + R_h
J = df.derivative(R,U,dU)

# Nonlinear Problem
problem = df.NonlinearVariationalProblem(R,U,J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relaxation_parameter'] = 1.0
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
solver.parameters['newton_solver']['error_on_nonconvergence'] = False
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['maximum_iterations'] = 10
solver.parameters['newton_solver']['report'] = True

assigner_inv = df.FunctionAssigner([Q_dg,Q_dg,Q_dg,Q_dg],V)
assigner     = df.FunctionAssigner(V,[Q_dg,Q_dg,Q_dg,Q_dg])

df.File('k01_phi0001_steady.xml') >> U
assigner_inv.assign([h0_w,h0,phi0_x,phi0_y],U)

#assigner.assign(U,[h0_w,h0,phi0_x,phi0_y])
t = 0.0
t_end = 20

hw_file = df.File('results/hw.pvd')
h_file = df.File('results/h.pvd')
phifile = df.File('results/phi.pvd')
fracfile = df.File('results/frac.pvd')
Nfile = df.File('results/N.pvd')
phihat = df.Function(Q_dg)
frachat = df.Function(Q_dg)
Nhat = df.Function(Q_dg)

while t<t_end:
    m.vector()[:] = (1+0.5*np.sin(2*np.pi*t))*m0.vector()[:] + 1e-1
    #m.vector()[:] = m0.vector()[:] + 1e-1
 
    solver.solve()
    assigner_inv.assign([h0_w,h0,phi0_x,phi0_y],U)

    t+=dt_float
    print(df.assemble(h0_w*df.dx),h0.vector().min())
    
    fracfrac = df.project(P_w/P_0,Q_dg)
    phiphi = df.project(phi,Q_dg)
    phihat.vector()[:] = phiphi.vector()[:]
    frachat.vector()[:] = fracfrac.vector()[:]
    NN = df.project(N,Q_dg)
    Nhat.vector()[:] = NN.vector()[:]
    hw_file << (h0_w,t)
    h_file << (h0,t)
    fracfile << (frachat,t)
    phifile << (phihat,t)
    Nfile << (Nhat,t)


#docker run --rm -it -v "$PWD":/app -w /app --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable 
             
