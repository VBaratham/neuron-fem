"""
2D FEM model of a segment embedded within other segments, all lying horizontally
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import fenics as fe
from mshr import generate_mesh, Rectangle as Rect
from vtkplotter.dolfin import plot

BOX_SIZE    = 1.0

SEG_LEN     = 0.21
SEG_GAP_X   = 0.04  # Gap between adjacent segments
SEG_HT      = 0.04  # Segment thickness
SEG_GAP_Y   = 0.02  # Gap between adjacent layers
SEG_OFFSET  = 0.08  # Distance between left edges of segment and the one above

DIPOLE      = True  # T: Use source with 2 opposing currents. F: 1 current

MESH_PTS    = 128

class Rectangle(Rect):
    def __init__(self, left, bottom, right, top):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.seg_coords = (left, bottom, right, top)

        self.rect = Rect(fe.Point(left, bottom), fe.Point(right, top))

        def is_near(self, pt):
            x, y = x
            if (fe.near(y, self.bottom) or fe.near(y, self.top)) \
               and (x > self.left and x < self.right):
                return True
            else:
                return False


class Simulation(object):
    def __init__(self, freq, active_seg_idx=(3, 2), conductivity=1.0, current=10.0):
        self.freq = freq
        self.active_seg_idx = active_seg_idx
        self.conductivity = conductivity
        self.current = current
        
    def _create_segments(self):
        self.segs = {}
        offset = 0 # horiz offset of first segment in the current layer
        # TODO: currently only works w/ up to 13 layers. Not sure why
        # for i, bottomedge in enumerate(np.arange(SEG_GAP_Y, BOX_SIZE, SEG_GAP_Y+SEG_HT)):
        for i, bottomedge in enumerate(np.arange(6*SEG_GAP_Y, BOX_SIZE, SEG_GAP_Y+SEG_HT)[:12]):
            # bottomedge = y coord of bottom of segments in current layer
            for j, leftedge in enumerate(np.arange(offset, BOX_SIZE, SEG_LEN+SEG_GAP_X)):
                # leftedge = x coord of leftmost point of current segment
                topedge = bottomedge + SEG_HT
                rightedge = leftedge + SEG_LEN
                self.segs[i, j] = Rectangle(leftedge, bottomedge, rightedge, topedge)

                # Save spatial coordinates of active segment
                if (i, j) == self.active_seg_idx:
                    self.active_seg_coords = (leftedge, bottomedge, rightedge, topedge)
                    print(self.active_seg_coords)

                # Increment offset for next layer
                offset -= SEG_OFFSET
                if offset < -SEG_LEN: # First segment will not reach the box
                    offset += SEG_LEN+SEG_GAP_X

        print(len(self.segs))

    def _create_domain(self):
        self.domain = Rect(fe.Point(0, 0), fe.Point(BOX_SIZE, BOX_SIZE))
        self._create_segments()
        for seg in self.segs.values():
            self.domain -= seg.rect
        return self.domain

    def _create_boundary_expression(self):
        left, bottom, right, top = self.active_seg_coords
        Im = self.current
        class BoundaryVals(fe.UserExpression):
            def eval(self, value, pt):
                """
                Set value[0] to self.current if x is on the active segment,
                or to 0 otherwise
                """
                x, y = pt
                if (fe.near(y, bottom) or fe.near(y, top)) and x > left and x < right:
                    value[0] = Im
                else:
                    value[0] = 0
        self.boundary_exp = BoundaryVals(degree=2)
                    
    def run(self):
        domain = self._create_domain()
        mesh = generate_mesh(domain, MESH_PTS)
        
        # fe.plot(mesh)
        # plt.show()

        self._create_boundary_expression()
        
        Omega = fe.FunctionSpace(mesh, 'CG', 2)
        sigma = self.conductivity
        Theta = fe.Function(Omega)
        v = fe.TestFunction(Omega)
        LHS = sigma * fe.inner(fe.grad(Theta), fe.grad(v)) * fe.dx - self.boundary_exp * v * fe.ds

        fe.solve(LHS == 0, Theta, solver_parameters={"newton_solver": {"absolute_tolerance": 1e-4}})

        fe.plot(Theta, "solution")
        plt.show()

class OneSegSimulation(Simulation):
    def _create_segments(self):
        super(OneSegSimulation, self)._create_segments()
        i, j = self.active_seg_idx
        self.segs = {(i, j): self.segs[i, j]}

class ConstrainedSimulation(Simulation):
    """
    Added constraint that the integral of V vanishes (guarantee uniqueness w/ pure Neumann BC)
    """
    def run(self):
        domain = self._create_domain()
        mesh = generate_mesh(domain, MESH_PTS)
        
        # fe.plot(mesh)
        # plt.show()

        self._create_boundary_expression()
        
        Omega = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        R = fe.FiniteElement("Real", mesh.ufl_cell(), 0)
        W = fe.FunctionSpace(mesh, Omega*R)
        Theta, c = fe.TrialFunction(W)
        v, d = fe.TestFunctions(W)
        sigma = self.conductivity
        
        LHS = (sigma * fe.inner(fe.grad(Theta), fe.grad(v)) + c*v + Theta*d) * fe.dx
        RHS = self.boundary_exp * v * fe.ds

        w = fe.Function(W)
        fe.solve(LHS == RHS, w)
        Theta, c = w.split()
        print(c(0, 0))

        # fe.plot(Theta, "solution", mode='color', vmin=-3, vmax=3)
        # plt.show()

        plot(fe.interpolate(Theta, fe.FunctionSpace(mesh, Omega)), mode='color')
        plt.show()

class OneSegConstrainedSimulation(ConstrainedSimulation, OneSegSimulation):
    pass

if __name__ == '__main__':
    sim = ConstrainedSimulation(freq=1.0)
    sim.run()

    
