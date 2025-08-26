from mpi4py import MPI
from dolfinx import fem, mesh # type: ignore
from dolfinx.fem import petsc # type: ignore
import ufl
from basix.ufl import element
from petsc4py import PETSc # type: ignore

import numpy as np
from typing import Optional

from vessel_models.elastic_vessel import ElasticVessel


class ElasticDGVessel(ElasticVessel):

    method_type = "DG"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.LB_fem = None
        self.RB_fem = None

    def create_mesh(self, h: float):
        n = int(self.L / h)
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, n, (0, self.L))

    def create_mesh_tags(self):
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim - 1)

        facet_imap = self.mesh.topology.index_map(tdim - 1)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts

        indices = np.arange(num_facets)
        values = np.zeros(num_facets, dtype=np.intc)

        left_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
        right_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], self.L))

        boundary_id = {"Gamma_L": 1, "Gamma_R": 2}
        values[left_facets] = boundary_id["Gamma_L"]
        values[right_facets] = boundary_id["Gamma_R"]

        self.msh_tags = mesh.meshtags(self.mesh, tdim - 1, indices, values)

    def create_fem_space(self):
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element("DG", self.mesh.topology.cell_name(), 1, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

        self.u = fem.Function(self.V)  # Current time step solution
        self.u_n = fem.Function(self.V)  # Previous

    def set_initial_condition(self):
        if self.V is None or self.u_n is None:
            raise ValueError("Function space not created. Call create_fem_space() first.")
        
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

    def max_eigval(self, u: fem.Function):
        eigval1 = self.alpha * (u[1] / u[0]) + self.c_alpha_ufl(u)
        eigval2 = self.alpha * (u[1] / u[0]) - self.c_alpha_ufl(u)

        return ufl.max_value(
            ufl.conditional(ufl.ge(eigval1, 0), eigval1, -eigval1),
            ufl.conditional(ufl.ge(eigval2, 0), eigval2, -eigval2)
        )

    def LxF(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u('+')), self.max_eigval(u('-')))

        # Lax-Friedrichs numerical flux
        flux_avg = 0.5 * (self.F(u('+')) + self.F(u('-'))) # type: ignore
        jump = u('+') - u('-')
        return flux_avg - 0.5 * lambda_max * jump # type: ignore

    def LxF_bound_L(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.LB_fem))

        # Lax-Friedrichs numerical flux at left boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.LB_fem)) # type: ignore
        jump = u - self.LB_fem
        return flux_avg - 0.5 * lambda_max * jump # type: ignore
    
    def LxF_bound_R(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.RB_fem))

        # Lax-Friedrichs numerical flux at right boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.RB_fem)) # type: ignore
        jump = self.RB_fem - u
        return flux_avg - 0.5 * lambda_max * jump # type: ignore

    def set_variational_problem(self, dt: float):
        
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.msh_tags)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(self.u_n, v) * ufl.dx
        L -= dt * ufl.inner(self.B(self.u_n), v) * ufl.dx # type: ignore
        L += dt * ufl.inner(self.F(self.u_n), v.dx(0)) * ufl.dx # type: ignore
        L += dt * ufl.inner(self.LxF(self.u_n), ufl.jump(v)) * ufl.dS # type: ignore

        L += dt * ufl.inner(self.LxF_bound_L(self.u_n), v) * ds(1) # type: ignore
        L += dt * ufl.inner(self.LxF_bound_R(self.u_n), v) * ds(2) # type: ignore

        self.bilinear_form = fem.form(a)
        self.linear_form = fem.form(L)

    def update_BCs(self, LB: Optional[np.ndarray] = None, RB: Optional[np.ndarray] = None):
        # Ensure we have valid boundary values
        left_bc = LB if LB is not None else getattr(self, 'LB', None)
        right_bc = RB if RB is not None else getattr(self, 'RB', None)
        
        if left_bc is None or right_bc is None:
            raise ValueError("Boundary conditions not properly initialized")
        
        if self.LB_fem is None:
            self.LB_fem = fem.Constant(self.mesh, left_bc)
        else:
            self.LB_fem.value = left_bc

        if self.RB_fem is None:
            self.RB_fem = fem.Constant(self.mesh, right_bc)
        else:
            self.RB_fem.value = right_bc

    def assemble_solver(self):
        if self.mesh is None or self.V is None:
            raise ValueError("Mesh or function space not created. Call create_mesh() and create_fem_space() first.")
        
        if self.bilinear_form is None or self.linear_form is None:
            raise ValueError("Variational problem not fully defined. Call set_variational_problem() and update_BCs() first.")

        self.A = petsc.assemble_matrix(self.bilinear_form)
        self.A.assemble()

        self.rhs = petsc.create_vector(self.linear_form)

        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def setup(self, h: float, dt: float):
        self.create_mesh(h)
        self.create_mesh_tags()
        self.create_fem_space()
        self.set_initial_condition()  # Move this before update_BCs
        self.update_BCs()
        self.set_variational_problem(dt)
        self.assemble_solver()

        self.h = h
        self.dt = dt

    def solve(self):
        if self.solver is None or self.rhs is None:
            raise ValueError("Solver not assembled. Call assemble_solver() first.")

        with self.rhs.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs, self.linear_form)
        self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.solver.solve(self.rhs, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array[:]

        if np.any(np.isnan(self.u.x.array)):
            raise ValueError("NaN values encountered in the solution.")

    def add_solution(self, t: float):
        if self.mesh is None or self.V is None:
            raise ValueError("Mesh or function space not created. Call create_mesh() and create_fem_space() first.")

        if self.u is None:
            raise ValueError("No solution available. Call solve() first.")
        
        comm = self.mesh.comm
        rank = comm.Get_rank()

        u = self.u

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 2)

        self.last_sol = global_sol

        if (rank == 0) and (t - self.last_saved_time) >= 0.001:
            self.solutions["t"].append(t)
            self.solutions["A"].append(global_sol[:, 0])
            self.solutions["Q"].append(global_sol[:, 1])

            mid_index = len(global_sol) // 2
            self.middlepoints["A"].append(global_sol[mid_index, 0])
            self.middlepoints["Q"].append(global_sol[mid_index, 1])

            self.last_saved_time = t


if __name__ == "__main__":
    data = {
        "id": 0,
        "length": 1,
        "initial_area": 0.126,
        "beta_coeff": 0.060606e7,
        "left_bound": "inflow",
        "right_bound": "branch"
    }

    vessel = ElasticDGVessel(**data)
    vessel.create_mesh(h=0.01)
    
    tdim = vessel.mesh.topology.dim
    print(vessel.mesh.topology.create_entities(tdim - 1))