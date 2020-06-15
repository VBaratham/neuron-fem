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
  - IMPORTANT: An added Lagrange term to impose the constraint that the integral
    of \Theta_e be 0 in the solution domain. This guarantees uniqueness of the
    solution, but it's not clear to me what it means exactly
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import fenics as fe
from mshr import Box, Cylinder, generate_mesh, Rectangle, Circle

BOX_SIZE = 1
CYL_X, CYL_Y, CYL_Z1, CYL_Z2 = 0.5, 0.5, 0.2, 0.6
CYL_R = 0.1
MESH_PTS = 40
CONDUCTIVITY = 30 # S/m
CURRENT = 0.01 # idk what units

# Define solution domain
box = Box(fe.Point(0, 0, 0), fe.Point(BOX_SIZE, BOX_SIZE, BOX_SIZE))
cylinder = Cylinder(fe.Point(CYL_X, CYL_Y, CYL_Z1), fe.Point(CYL_X, CYL_Y, CYL_Z2), CYL_R, CYL_R)

# DEBUG
# box = Rectangle(fe.Point(0, 0), fe.Point(BOX_SIZE, BOX_SIZE))
# cylinder = Circle(fe.Point(CYL_X, CYL_Y), CYL_R)
# END DEBUG

domain = box - cylinder

# Generate and display the mesh
mesh = generate_mesh(domain, MESH_PTS)
# fe.plot(mesh, "3D Mesh")
# plt.show()

# Variables defined below are named exactly as in eq (A.1) (see comment at top of file)

# Create (scalar) function space for the solution domain. Try 'P' as the element family?
Omega = fe.FunctionSpace(mesh, 'CG', 2)

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
        rad_sq = (x[0] - CYL_X)**2 + (x[1] - CYL_Y)**2
        if fe.near(rad_sq, CYL_R**2):
            value[0] = CURRENT
        else:
            value[0] = 0
boundary = BoundaryVals(degree=2)
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
fe.solve(LHS == 0, Theta, solver_parameters={"newton_solver": {"absolute_tolerance": 6.5e-7}})

# Plot solution
# fe.plot(Theta, "Solution")
# plt.show()
import ipdb; ipdb.set_trace()

