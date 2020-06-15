"""
MWE which creates separate meshes for the intra/extracellular domains,
and attempts to use MultiMesh[Function[Space]] to specify the form

https://fenicsproject.discourse.group/t/integration-on-part-of-multimesh/2453/4

Hi Cécile, I'm interested in doing something like this, but trying to assess whether your mixed-dim framework is what I need. I describe my problem here: https://fenicsproject.discourse.group/t/prescribe-discontinuity-on-internal-boundary/3365/4. Essentially, the only part where I need to assemble a form that integrates variables on distinct meshes is to restrict the discontinuity in the solution across the boundary between the two meshes to a specified value.

I am seeing some classes called "MultiMesh", "MultiMeshFunctionSpace", etc, which sound like they might 
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
I = FunctionSpace(mesh_i, "Lagrange", 1)
E = FunctionSpace(mesh_e, "Lagrange", 1)
IE = MixedFunctionSpace(I, E)

# I = FiniteElement("Lagrange", mesh_i.ufl_cell(), 1)
# E = FiniteElement("Lagrange", mesh_e.ufl_cell(), 1)
# IE = FunctionSpace(mesh_e, I*E) # Not sure if this will work since it uses mesh_e

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
    # TODO: we are now cutting corners - check that both endpoints are on any edge,
    # not necessarily the same edge
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
# Vi = Function(I)
# Ve = Function(E)
# vi = TestFunction(I)
# ve = TestFunction(E)
# Vi, Ve = TrialFunction(IE) # TODO: Try this?
V = Function(IE)
Vi, Ve = V.sub(0), V.sub(1)
vi, ve = TestFunction(IE)

# Measures
# dx should be same for E and I (the form arguments are only from one mesh,
# so the mesh is implicit), but define here just to be safe
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


soln = Function(IE)
solve(LHS == 0, soln, [DirichletBC(IE, 1.0, boundaries, LEFTEDGE),
                       DirichletBC(IE, 2.0, boundaries, BOTTOMEDGE)])
Vi, Ve = soln.sub(0), soln.sub(1)


from vtkplotter.dolfin import plot
plot(Ve)
import ipdb; ipdb.set_trace()

