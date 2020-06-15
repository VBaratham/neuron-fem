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
    def __init__(self, left, bottom, right, top, mesh_pts=0):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.seg_coords = (left, bottom, right, top)
        self.mesh_pts = mesh_pts

        self.rect = Rect(fe.Point(left, bottom), fe.Point(right, top))
        if self.mesh_pts > 0:
            self.mesh = generate_mesh(self.rect, self.mesh_pts)

    def is_near(self, pt):
        x, y = pt
        if (fe.near(y, self.bottom) or fe.near(y, self.top)) \
           and (x > self.left and x < self.right):
            return True
        elif (fe.near(y, self.left) or fe.near(y, self.right)) \
           and (x > self.bottom and x < self.top):
            return True
        else:
            return False

    def set_v(self, V_e, V_m):
        """
        Given V_e and V_m (both fe.Function), construct a new fe.UserExpression
        representing V_i at the membrane

        V_e = extracellular potential
        V_m = membrane potential
        (both fe.Function)
        """
        _self = self
        class BoundaryVals(fe.UserExpression):
            def eval(self, value, pt):
                if _self.is_near(pt):
                    value[0] = V_m(*pt) + V_e(*pt)
                else:
                    value[0] = 0
        self.V_i = BoundaryVals(degree=2)

    def run(self):
        """
        Solve Poisson inside this segment, given Dirichlet BC passed in via assign_v()
        """
        # TODO: solve for V_i
        self.V_i = None # This should be a fe.Function
        pass
        

class ExtracellularSimulation(object):
    def __init__(self, active_seg_coords, passive_seg_coords, cm=.1, rm=1, dt=0.0001):
        self.active_seg_coords = active_seg_coords
        self.passive_seg_coords = passive_seg_coords
        self.active_seg_current = 1.0 # constant across entire seg
        self.passive_seg_im = fe.Constant(0) # initial value
        self.passive_seg_vm = fe.Constant(-80) # initial value
        self.dt = dt
        self.cm = cm
        self.rm = rm
    
    def _create_e_domain(self):
        box = Rect(fe.Point(0, 0), fe.Point(BOX_SIZE, BOX_SIZE))

        left, bottom, right, top = self.active_seg_coords
        # self.active_seg = Segment(0.3, 0.3, 0.7, 0.35)
        self.active_seg = Segment(left, bottom, right, top, mesh_pts=0)

        left, bottom, right, top = self.passive_seg_coords
        # self.passive_seg = Segment(0.3, 0.5, 0.7, 0.55)
        self.passive_seg = Segment(left, bottom, right, top)

        self.domain = box - self.active_seg.rect - self.passive_seg.rect

        return self.domain

    def _create_boundary_expression(self):
        _self = self
        class BoundaryVals(fe.UserExpression):
            def eval(self, value, pt):
                """
                Set value[0] to self.active_seg_current, or
                             to the value from self.passive_seg_im, or
                             0 if not on either boundary
                """
                if _self.passive_seg.is_near(pt):
                    value[0] = _self.passive_seg_im(*pt)
                elif _self.active_seg.is_near(pt):
                    value[0] = _self.active_seg_current
                else:
                    value[0] = 0
        self.boundary_exp = BoundaryVals(degree=2)
                    
    def run(self):
        e_domain = self._create_e_domain()
        mesh = generate_mesh(e_domain, MESH_PTS)

        self._create_boundary_expression()

        Omega_e = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        Omega_i = fe.FiniteElement("Lagrange", self.passive_seg.mesh.ufl_cell(), 1)
        Omega = fe.FunctionSpace(mesh, Omega_e*Omega_i)
        Theta_e, Theta_i = fe.TrialFunction(Omega)
        v_e, v_i = fe.TestFunctions(Omega)
        sigma_e, sigma_i = 0.3, 0.4

        LHS = sigma_e * fe.inner(fe.grad(Theta_e), fe.grad(v_e)) * fe.dx # poisson
        LHS += sigma_i * fe.inner(fe.grad(Theta_i), fe.grad(v_i)) * fe.dx # poisson
        LHS -= sigma_e * fe.grad(Theta_e) * v_e * fe.ds # current
        LHS += sigma_i * fe.grad(Theta_i) * v_i * fe.ds # current
        RHS = self.boundary_exp * v_e * fe.ds # source term
        
        # TODO: solve Poisson in extracellular space
        w = fe.Function(Omega)
        fe.solve(LHS == RHS, w)
        Theta_e, Theta_i = w.split()

        plot(fe.interpolate(Theta, fe.FunctionSpace(mesh, Omega)), mode='color')
        plt.show()
        
        # Set the potential just inside the membrane to Ve (just computed)
        # minus V_m (from prev timestep)
        self.passive_seg.set_v(Theta, self.passive_seg_vm)

        # Solve for the potential and current inside the passive cell
        self.passive_seg.run()

        # Use Im to compute a new Vm for the next timestep, eq (8)
        self.passive_seg_vm = self.passive_seg_vm + self.dt / self.cm * (
            self.passive_seg_im - self.passive_seg_vm / self.rm)


class TimeSimulation(object):
    """
    Run a simulation over time
    """

    def __init__(self, t_sim=1000, v_init=-80, i_init=0, dvdt=0):
        self.extracell = ExtracellularSimulation((.3, .3, .7, .35), (.3, .5, .7, .55))

    def run(self):
        self.extracell.run()
        

if __name__ == '__main__':
    sim = TimeSimulation()
    sim.run()
