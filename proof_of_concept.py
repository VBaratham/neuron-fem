"""
Proof-of-concept for FEM modeling of extracellular potentials using FEniCS
Solve the inhomogeneous Poisson equation for a cylinder emitting sine wave current in a box

Variational formulation of the problem in the extracellular domain is
given by eq (A.1) in Agudelo-Toro and Neef, 2013:
https://iopscience.iop.org/article/10.1088/1741-2560/10/2/026019#jne440081eqn11
with:
  - I_N = 0 (no currents through the box bounding the solution domain)
  - I_m given by NEURON (cable equation solution for membrane potentials;
    \Gamma is the cell membrane)
  - \rho_e = 0 (no extracellular current sources)
"""
import matplotlib
matplotlib.use("TkAgg")
from fenics import Point, plot
from mshr import Box, Cylinder, generate_mesh
import matplotlib.pyplot as plt

MESH_PTS = 64

# Define solution domain
box = Box(Point(0, 0, 0), Point(100, 100, 100))
cylinder = Cylinder(Point(50, 50, 10), Point(50, 50, 30), 5.0, 5.0)
domain = box - cylinder

# Generate and display the mesh
mesh = generate_mesh(domain, MESH_PTS)
plot(mesh, "3D Mesh")
plt.show()
