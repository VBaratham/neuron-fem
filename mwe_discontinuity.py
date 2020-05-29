"""
Here is a model problem for what I want to do. Consider a unit square $\Omega$ with a smaller square $\Omega_i$ fully contained in $\Omega$. Let $\Omega_e = \Omega \setminus \Omega_i$ be the region inside the unit square but outside the smaller square. We want to solve Poisson's equation in $\Omega_e$ and $\Omega_i$:
$$
- \nabla ^2 V_i = 0 \text{ in } \Omega_i
\\
- \nabla ^2 V_e = 0 \text{ in } \Omega_e
$$
Subject to homogeneous (for now) Neumann boundary conditions on the outer square boundary $\partial \Omega$

I want the normal derivative to be smooth across $\partial \Omega_i$:
$$
\mathbf{n} \cdot \nabla V_i = - \mathbf{n} \cdot \nabla V_e \text{ on } \partial \Omega_i
$$
I also want to specify the discontinuity in $V$ across the boundary:
$$
V_e - V_i = V_m \text{ on } \partial \Omega_i
$$
for a given function $V_m$ defined on $\partial \Omega_i$.

I believe the weak form of this problem (not including that last condition on the discontinuity across $\partial \Omega_i$) is:
$$
\int_{\Omega_i} \nabla V_i \cdot \nabla v_i \, dx
+ \int_{\Omega_e} \nabla V_e \cdot \nabla v_e \, dx
+ \int_{\partial \Omega_i} (\nabla V_e \cdot v_e - \nabla V_i \cdot v_i) ds = 0
$$

I'm not sure how to implement this in fenics. One approach would be to construct separate meshes for $\Omega_i$ and $\Omega_e$, but I don't know how to create a mixed space over two distinct meshes (and from what I've read, it doesn't seem like this is possible in fenics). Is there some way to get this to work, or is there another approach that might be suitable for this problem? I'm grateful for any advice that might help point me in the right direction!
"""


from fenics import *
from mshr import *

# Omega = Rectangle(Point(0, 0), Point(1, 1))
# Omega_i = Rectangle(Point(0.2, 0.2), Point(0.4, 0.4))
# Omega_e = Omega - Omega_i

# mesh_i = generate_mesh(Omega_i, 64)
# mesh_e = generate_mesh(Omega_e, 64)

# # I = FiniteElement("Lagrange", mesh_i.ufl_cell(), 1)
# # E = FiniteElement("Lagrange", mesh_e.ufl_cell(), 1)
# I = FunctionSpace(mesh_i, "Lagrange", 1)
# E = FunctionSpace(mesh_e, "Lagrange", 1)
# import ipdb; ipdb.set_trace()
# Vi = TrialFunction(I)
# Ve = TrialFunction(E)
# vi = TestFunction(I)
# ve = TestFunction(E)

# LHS = inner(grad(Vi), grad(vi)) * dx
# LHS += inner(grad(Ve), grad(ve)) * dx
# LHS += grad(Vi) * vi * ds
# LHS += grad(Ve) * ve * ds



# class Interior(SubDomain):
#     def inside(self, pt, on_boundary):
#         x, y = pt
#         if x > .2 and x < .4 and y > .2 and y < .4:
#             return True
#         else:
#             return False
# domains = MeshFunction('size_t', mesh, 2) # all 0 initially
# Interior().mark(domains, 1)

# class InteriorBoundary(SubDomain):
#     def inside(self, pt, on_boundary):
#         x, y = pt
#         if x > .2 and x < .4 and (near(y, .2) or near(y, .4)):
#             return True
#         if y > .2 and y < .4 and (near(x, .2) or near(x, .4)):
#             return True
#         return False
# boundary_parts = MeshFunction("size_t", mesh, 1)
# InteriorBoundary().mark(boundary_parts, 1)

# Define constants for marking subdomains/boundaries
OMEGA_E = 1
OMEGA_I = 2
INTERFACE = 3

Omega = Rectangle(Point(0, 0), Point(1, 1))
Omega_i = Rectangle(Point(0.2, 0.2), Point(0.4, 0.4))
Omega.set_subdomain(OMEGA_E, Omega - Omega_i)
Omega.set_subdomain(OMEGA_I, Omega_i)
mesh = generate_mesh(Omega, 256)
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

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

# TEST
# Define a function of ones on the mesh:
# V = FunctionSpace(mesh, "CG", 1)
# f = Function(V)
# f.vector()[:] = 1.

# dStest = dS(subdomain_data=boundaries)
# perim = assemble(f*dStest(INTERFACE))
# print("Perimeter from integrating 1 on the interface: {} (expected 0.8)".format(perim))
# END TEST

V = FunctionSpace(mesh, "Lagrange", 1)

dx = Measure("dx")(subdomain_data=subdomains)
dS = Measure("dS")(subdomain_data=boundaries)

Theta = Function(V)
v = TestFunction(V)

n = FacetNormal(mesh)

LHS = inner(grad(Theta), grad(v)) * dx(OMEGA_E)
LHS += inner(grad(Theta), grad(v)) * dx(OMEGA_I)
LHS += inner(n('-'), grad(Theta('-'))) * v('-') * dS(INTERFACE)
LHS -= inner(n('+'), grad(Theta('+'))) * v('+') * dS(INTERFACE)

# sol = Function(V)
solve(LHS == 0, Theta)


import ipdb; ipdb.set_trace()
