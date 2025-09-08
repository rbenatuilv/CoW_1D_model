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

        self.diff_const = 10.0

    def create_mesh(self, h: float):
        n = int(self.L / h)
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, n, (0, self.L))
        self.h = h

    def create_mesh_tags(self):
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim - 1)

        facet_imap = self.mesh.topology.index_map(tdim - 1)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts

        indices = np.arange(num_facets)
        values = np.zeros(num_facets)

        left_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
        right_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], self.L))

        boundary_id = {"Gamma_L": 1, "Gamma_R": 2}
        values[left_facets] = boundary_id["Gamma_L"]
        values[right_facets] = boundary_id["Gamma_R"]

        self.msh_tags = mesh.meshtags(self.mesh, tdim - 1, indices, values)

    def create_fem_space(self):
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element("DG", self.mesh.topology.cell_name(), 0, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

        self.u = fem.Function(self.V)  # Current time step solution
        self.v = fem.Function(self.V)  # auxiliary solution

        self.u_n = fem.Function(self.V)  # Previous
        self.v_n = fem.Function(self.V)  # auxiliary previous solution

    def set_initial_condition(self):
        if self.V is None or self.u_n is None:
            raise ValueError("Function space not created. Call create_fem_space() first.")
        
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)
        self.v_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

        comm = MPI.COMM_WORLD

        u = self.u_n

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 2)

        self.last_sol = global_sol

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
        flux_avg = ufl.avg(self.F(u)) # type: ignore
        jump = ufl.jump(u)
        return flux_avg - 0.5 * self.diff_const * lambda_max * jump # type: ignore

    def LxF_bound_L(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.LB_fem))

        # Lax-Friedrichs numerical flux at left boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.LB_fem)) # type: ignore
        jump = u - self.LB_fem
        return flux_avg - 0.5 * self.diff_const * lambda_max * jump # type: ignore

    def LxF_bound_R(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.RB_fem))

        # Lax-Friedrichs numerical flux at right boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.RB_fem)) # type: ignore
        jump = self.RB_fem - u
        return flux_avg - 0.5 * self.diff_const * lambda_max * jump # type: ignore
    
    def dU_dz(self, u: np.ndarray):
        area = u[:, 0]
        flux = u[:, 1]

        h = self.L / float(len(area) // 2)

        dA = []
        for i in range(0, len(area)-1, 2):
            derv = (area[i+1] - area[i]) / h
            dA.append(derv)

        dQ = []
        for i in range(0, len(flux)-1, 2):
            derv = (flux[i+1] - flux[i]) / h
            dQ.append(derv)
        
        return np.stack([dA, dQ], axis=-1)

    def HxF(self, u: fem.Function, v: fem.Function):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.msh_tags)

        L = -ufl.inner(self.B(u), v) * ufl.dx # type: ignore
        L += ufl.inner(self.F(u), v.dx(0)) * ufl.dx # type: ignore
        L += ufl.inner(self.LxF(u), (ufl.jump(v))) * ufl.dS # type: ignore
        L += ufl.inner(self.LxF_bound_L(u), v) * ds(1) # type: ignore
        L += ufl.inner(self.LxF_bound_R(u), -v) * ds(2) # type: ignore

        return L

    def set_variational_problem(self, dt: float):
        u1 = ufl.TrialFunction(self.V)
        v1 = ufl.TestFunction(self.V)

        a1 = ufl.inner(u1, v1) * ufl.dx
        L1 = ufl.inner(self.u_n, v1) * ufl.dx
        L1 += dt * self.HxF(self.u_n, v1) # type: ignore

        self.bilinear_form_1 = fem.form(a1)
        self.linear_form_1 = fem.form(L1)

        u2 = ufl.TrialFunction(self.V)
        v2 = ufl.TestFunction(self.V)

        a2 = ufl.inner(u2, v2) * ufl.dx
        L2 = 0.5 * ufl.inner(self.u_n, v2) * ufl.dx # type: ignore
        L2 += 0.5 * ufl.inner(self.v_n, v2) * ufl.dx # type: ignore
        L2 += 0.5 * dt * self.HxF(self.v_n, v2) # type: ignore

        self.bilinear_form_2 = fem.form(a2)
        self.linear_form_2 = fem.form(L2)

    def update_BCs(self, LB: Optional[np.ndarray] = None, RB: Optional[np.ndarray] = None):
        # Ensure we have valid boundary values
        left_bc = LB if LB is not None else self.LB
        right_bc = RB if RB is not None else self.RB
        
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
        
        self.A1 = petsc.assemble_matrix(self.bilinear_form_1)
        self.A1.assemble()

        self.rhs_1 = petsc.create_vector(self.linear_form_1)

        self.solver_1 = PETSc.KSP().create(self.mesh.comm)
        self.solver_1.setOperators(self.A1)
        self.solver_1.setType(PETSc.KSP.Type.PREONLY)
        self.solver_1.getPC().setType(PETSc.PC.Type.LU)

        self.A2 = petsc.assemble_matrix(self.bilinear_form_2)
        self.A2.assemble()

        self.rhs_2 = petsc.create_vector(self.linear_form_2)

        self.solver_2 = PETSc.KSP().create(self.mesh.comm)
        self.solver_2.setOperators(self.A2)
        self.solver_2.setType(PETSc.KSP.Type.PREONLY)
        self.solver_2.getPC().setType(PETSc.PC.Type.LU)

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

        with self.rhs_1.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs_1, self.linear_form_1)
        self.rhs_1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.solver_1.solve(self.rhs_1, self.v.x.petsc_vec)
        self.v.x.scatter_forward()
        self.v_n.x.array[:] = self.v.x.array[:]

        with self.rhs_2.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs_2, self.linear_form_2)
        self.rhs_2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) 
        
        self.solver_2.solve(self.rhs_2, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array[:]

        if np.any(np.isnan(self.u.x.array)):
            raise ValueError("NaN values encountered in the solution.")

    def get_max_cfl_dt(self):
        """Calculate maximum stable time step based on CFL condition"""
        # Get maximum wave speed (eigenvalue)
        c_max = np.sqrt(self.beta / (2 * self.rho * self.A0))  # Approximate wave speed
        return 0.5 * self.h / c_max  # CFL < 0.5 for stability


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

        if (rank == 0) and (t - self.last_saved_time) >= 1e-5:
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
    h = 2 * 0.03125
    vessel.setup(h, dt=1e-5)
    print("Max CFL dt:", vessel.get_max_cfl_dt())

    