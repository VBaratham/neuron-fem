"""
Proof-of-concept for FEM modeling of extracellular potentials using FEniCS.
Solve the inhomogeneous Poisson equation for a cylinder emitting current in a box.
Only the potential outside the cylinder is computed.

Variational formulation of the problem in the extracellular domain is
given by eq (A.1) in Agudelo-Toro and Neef, 2013:
https://iopscience.iop.org/article/10.1088/1741-2560/10/2/026019#jne440081eqn11
with:
  - I_N = 0 (no currents through the box bounding the solution domain)
  - I_m given by NEURON (cable equation solution for membrane potentials;
    \Gamma is the cell membrane). But for this proof of concept, the values
    are taken to be a constant
  - \rho_e = 0 (no extracellular current sources)
"""
import matplotlib
matplotlib.use("TkAgg")
import fenics as fe
from mshr import Box, Cylinder, generate_mesh, Rectangle, Circle
import matplotlib.pyplot as plt

BOX_SIZE = 100
MESH_PTS = 1024
CONDUCTIVITY = 0.3 # S/m
CURRENT = 0.5 # idk what units

# Define solution domain
# box = Box(fe.Point(0, 0, 0), fe.Point(BOX_SIZE, BOX_SIZE, BOX_SIZE))
# cylinder = Cylinder(fe.Point(50, 50, 10), fe.Point(50, 50, 30), 5.0, 5.0)

# DEBUG
box = Rectangle(fe.Point(0, 0), fe.Point(BOX_SIZE, BOX_SIZE))
cylinder = Circle(fe.Point(50,10), 5.0)
# END DEBUG

domain = box - cylinder

# Generate and display the mesh
mesh = generate_mesh(domain, MESH_PTS)
# fe.plot(mesh, "3D Mesh")
# plt.show()

# Variables defined below are named exactly as in eq (A.1) (see comment at top of file)

# Create (scalar) function space for the solution domain. Try 'P' as the element family?
Omega = fe.FunctionSpace(mesh, 'CG', 1)

# Define a function for the boundary term.
# The function should return `CURRENT` on the cylinder, and 0 on the box
# See https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html#defining-subdomains-for-different-materials
# for defining subdomains for multiple Neumann conditions
# Here, we use the "Expressions" technique: https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html#using-expressions-to-define-subdomains
class BoundaryVals(fe.UserExpression):
    def eval(self, value, x):
        """
        Set value[0] to 0 if x is on the box,
        or to CURRENT otherwise
        """
        # for edge in (0, BOX_SIZE):
        #     for coord in x:
        #         if fe.near(coord, edge):
        #             value[0] = 0
        #             return
        # rad_sq = (x[0] - 50)**2 + (x[1] - 10)**2
        # print(x, rad_sq)
        # value[0] = CURRENT
        rad_sq = (x[0] - 50)**2 + (x[1] - 10)**2
        if fe.near(rad_sq, 25):
            value[0] = CURRENT
        else:
            value[0] = 0
boundary = BoundaryVals(degree=1)
# boundary = fe.Expression('(x[0] - 50)*(x[0] - 50) + (x[1] - 10)*(x[1] - 10) - 25 < tol ? cur : 0', degree=1, tol=1e-7, cur=CURRENT)

# Define the variational problem
sigma = CONDUCTIVITY
Theta = fe.Function(Omega)
v = fe.TestFunction(Omega)
LHS = sigma * fe.inner(fe.grad(Theta), fe.grad(v)) * fe.dx - boundary * v * fe.ds

# Solve the variational problem
# we use 0 as the RHS because
#  - There are no sources (\rho = 0)
#  - The Neumann condition on \partial\Theta_N is combined with the I_m term on the LHS
fe.solve(LHS == 0, Theta)

# Plot solution
fe.plot(Theta, "Solution")
plt.show()

