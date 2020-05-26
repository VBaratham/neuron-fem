"""
Proof of concept for solving Poisson separately in the
intracellular and extracellular domains, manually imposing
the membrane potential boundary condition
"""

import fenics as fe
from mshr import generate_mesh, Rectangle as Rect
from vtkplotter.dolfin import plot

from segments2d import Simulation

BOX_SIZE    = 1.0
MESH_PTS    = 64

class Segment(object):
    """
    Represents either the passive or active segment.
    For the active segment, the mesh will not be used
    """
    def __init__(self, left, bottom, right, top, mesh_pts=32):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.seg_coords = (left, bottom, right, top)
        self.mesh_pts = mesh_pets

        self.rect = Rect(fe.Point(left, bottom), fe.Point(right, top))
        self.mesh = generate_mesh(self.rect, self.mesh_pts)

    def is_near(self, pt):
        x, y = pt
        if (fe.near(y, self.bottom) or fe.near(y, self.top)) \
           and (x > self.left and x < self.right):
            return True
        else:
            return False

    def assign_v(self, v):
        pass

    def run(self):
        """
        Solve Poisson inside this segment, given Dirichlet BC passed in via assign_v()
        """
        pass
        

class TwoSegCapacitiveSimulation(object):
    def _create_e_domain(self):
        box = Rect(fe.Point(0, 0), fe.Point(BOX_SIZE, BOX_SIZE))
        self.active_seg = Segment(0.3, 0.3, 0.7, 0.35)
        self.passive_seg = Segment(0.3, 0.5, 0.7, 0.55)

        self.domain = box - self.active_seg.rect - self.passive_seg.rect

        return self.domain

    def _create_boundary_expression(self):
        pass
    
    def run(self):
        e_domain = self._create_e_domain()
        mesh = generate_mesh(e_domain, MESH_PTS)

        self._create_boundary_expression()

        Omega = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R = fe.FiniteElement("Real", mesh.ufl_cell(), 0)
        W = fe.FunctionSpace(mesh, Omega*R)
        Theta, c = fe.TrialFunction(W)
        v, d = fe.TestFunctions(W)
        sigma = 0.3

        LHS = (sigma * fe.inner(fe.grad(Theta), fe.grad(v)) + c*v + Theta*d) * fe.dx
        RHS = self.boundary_exp * v * fe.ds
        
        # TODO: solve Poisson in extracellular space

        # Set the potential just inside the membrane to V_e (just computed)
        # minus V_m (from prev timestep)
        self.passive_seg.set_v()

        # Solve for the potential inside the passive cell (V_i)
        self.passive_seg.run()

        # Compute the new V_m by subtracting V_e - V_i
        pass

    
