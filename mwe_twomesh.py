"""
MWE which creates separate meshes for the intra/extracellular domains
"""


from fenics import *
from mshr import *

# Define geometry, create meshes, mixed function space

Omega = Rectangle(Point(0, 0), Point(1, 1))
Omega_i = Rectangle(Point(0.2, 0.2), Point(0.4, 0.4))
Omega_e = Omega - Omega_i

mesh_i = generate_mesh(Omega_i, 256)
mesh_e = generate_mesh(Omega_e, 256)

# The following is from https://github.com/cdaversin/mixed-dimensional-examples/blob/v2019.1/Poisson/poisson_lm_3D.py but doesn't work in fenics 2019.1
# I = FunctionSpace(mesh_i, "Lagrange", 1)
# E = FunctionSpace(mesh_e, "Lagrange", 1)
# IE = MixedFunctionSpace(I, E)

I = FiniteElement("Lagrange", mesh_i.ufl_cell(), 1)
E = FiniteElement("Lagrange", mesh_e.ufl_cell(), 1)
IE = FunctionSpace(mesh, I*E)

import ipdb; ipdb.set_trace()

# Mark boundaries of Omega_e (subdomains of \partial \Omega_e; \Omega_i is all one domain)
INTERFACE = 3
LEFTEDGE = 4
BOTTOMEDGE = 5

boundaries = MeshFunction("size_t", mesh_e, 1)
for f in facets(mesh_e):
    p0 = Vertex(mesh_e, f.entities(0)[0])
    p1 = Vertex(mesh_e, f.entities(0)[1])
    x0, y0 = p0.x(0), p0.x(1)
    x1, y1 = p1.x(0), p1.x(1)
    on_vert_edge = lambda x, y: (near(x, .2) or near(x, .4)) and y > .2 and y < .4
    on_horiz_edge = lambda x, y: (near(y, .2) or near(y, .4)) and x > .2 and x < .4
    on_edge = lambda x, y: on_vert_edge(x, y) or on_horiz_edge(x, y)
    if on_edge(x0, y0) and on_edge(x1, y1):
        boundaries[f] = INTERFACE
    elif x0 < DOLFIN_EPS and x1 < DOLFIN_EPS:
        boundaries[f] = LEFTEDGE
    elif y0 < DOLFIN_EPS and y1 < DOLFIN_EPS:
        boundaries[f] = BOTTOMEDGE

# # TEST
# # Define a function of ones on the mesh:
# V = FunctionSpace(mesh_e, "CG", 1)
# f = Function(V)
# f.vector()[:] = 1.

# # Integrate over the membrane
# dstest = ds(subdomain_data=boundaries)
# perim = assemble(f*dstest(INTERFACE))
# print("Perimeter from integrating 1 on the interface: {} (expected 0.8)".format(perim))
# END TEST


# Define variational form

# Test functions
Vi = Function(I)
Ve = Function(E)
vi = TestFunction(I)
ve = TestFunction(E)

# Measures
# dx same for E and I, but define here just to be safe
dx_i = Measure("dx")(domain=mesh_i)
dx_e = Measure("dx")(domain=mesh_e)
ds = Measure("ds")(domain=mesh_e, subdomain_data=boundaries)
# dS not used (no integration over internal facets)

# Normal vectors to facets
ni = FacetNormal(mesh_i)
ne = FacetNormal(mesh_e)


LHS = inner(grad(Vi), grad(vi)) * dx_i # Poisson
LHS += inner(grad(Ve), grad(ve)) * dx_e # Poisson
# LHS += (inner(ni, grad(Vi)) * vi - inner(ne, grad(Ve)) * ve) * ds(INTERFACE) # continuous current
# LHS += grad(Vi) * vi * ds
# LHS += grad(Ve) * ve * ds


solve(LHS == 0, Ve, [DirichletBC(E, 1.0, boundaries, LEFTEDGE),
                     DirichletBC(E, 2.0, boundaries, BOTTOMEDGE)])













# Define constants for marking subdomains/boundaries
OMEGA_E = 1
OMEGA_I = 2
INTERFACE = 3
LEFTEDGE = 4
BOTTOMEDGE = 5

# Create domain and mark subdomains
Omega = Rectangle(Point(0, 0), Point(1, 1))
Omega_i = Rectangle(Point(0.2, 0.2), Point(0.4, 0.4))
Omega_e = Omega - Omega_i
mesh_i = generate_mesh(Omega_i, 128)
mesh_e = generate_mesh(Omega_e, 128)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, 1)
for f in facets(mesh):
    p0 = Vertex(mesh, f.entities(0)[0])
    p1 = Vertex(mesh, f.entities(0)[1])
    x0, y0 = p0.x(0), p0.x(1)
    x1, y1 = p1.x(0), p1.x(1)
    on_vert_edge = lambda x, y: (near(x, .2) or near(x, .4)) and y > .2 and y < .4
    on_horiz_edge = lambda x, y: (near(y, .2) or near(y, .4)) and x > .2 and x < .4
    on_edge = lambda x, y: on_vert_edge(x, y) or on_horiz_edge(x, y)
    if on_edge(x0, y0) and on_edge(x1, y1):
        boundaries[f] = INTERFACE
    elif x0 < DOLFIN_EPS and x1 < DOLFIN_EPS:
        boundaries[f] = LEFTEDGE
    elif y0 < DOLFIN_EPS and y1 < DOLFIN_EPS:
        boundaries[f] = BOTTOMEDGE

# TEST
# Define a function of ones on the mesh:
# V = FunctionSpace(mesh, "CG", 1)
# f = Function(V)
# f.vector()[:] = 1.

# Integrate over the internal boundary \partial \Omega_i
# dStest = dS(subdomain_data=boundaries)
# perim = assemble(f*dStest(INTERFACE))
# print("Perimeter from integrating 1 on the interface: {} (expected 0.8)".format(perim))
# END TEST

V = FunctionSpace(mesh, "Lagrange", 1)
# V = FunctionSpace(mesh, "DG", 2)

dx = Measure("dx")(subdomain_data=subdomains)
dS = Measure("dS")(subdomain_data=boundaries)
ds = Measure("ds")(subdomain_data=boundaries)

Theta = Function(V)
v = TestFunction(V)

n = FacetNormal(mesh)

LHS = inner(grad(Theta), grad(v)) * dx(OMEGA_E)
LHS += inner(grad(Theta), grad(v)) * dx(OMEGA_I)
LHS += inner(n('-'), grad(Theta('-'))) * v('-') * dS(INTERFACE)
LHS -= inner(n('+'), grad(Theta('+'))) * v('+') * dS(INTERFACE)
Vm = 0.5
# LHS += (Theta('+')*v('+') - Theta('-')*v('-') - Vm*v('-')) * dS(INTERFACE)
# LHS += 1.0 * v * ds(LEFTEDGE) # This works fine if you remove the leftedge DirichletBC

# sol = Function(V)
# solve(LHS == 0, Theta, solver_parameters={"newton_solver": {"absolute_tolerance": 6e-2}})
solve(LHS == 0, Theta, [DirichletBC(V, 1.0, boundaries, LEFTEDGE),
                        DirichletBC(V, 2.0, boundaries, BOTTOMEDGE)]
)

from vtkplotter.dolfin import plot
plot(Theta)
import ipdb; ipdb.set_trace()

