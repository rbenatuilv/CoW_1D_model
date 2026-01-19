from mpi4py import MPI
from dolfinx import fem, mesh # type: ignore
from dolfinx.fem import petsc # type: ignore
import ufl
from basix.ufl import element
from petsc4py import PETSc # type: ignore

import numpy as np
from typing import Optional

from vessel_models.elastic_vessel import ElasticVessel


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class ElasticCGVessel(ElasticVessel):

    method_type = "CG"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mesh = None
        self.V = None
        self.dofs_L = None
        self.dofs_R = None
        self.bcs = None

        self.u = None  # Current time step solution
        self.u_n = None  # Previous time step solution

        self.bilinear_form = None
        self.linear_form = None
        self.A = None
        self.rhs = None
        self.solver = None

    def create_mesh(self, h: float):
        n = int(self.L / h)

        self.mesh = mesh.create_interval(MPI.COMM_WORLD, n, (0, self.L))

    def create_fem_space(self):
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element("Lagrange", self.mesh.topology.cell_name(), 1, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

        self.u = fem.Function(self.V)  # Current time step solution
        self.u_n = fem.Function(self.V)  # Previous time step solution

    def set_boundary_dofs(self):
        if self.V is None:
            raise ValueError("Function space not created. Call create_fem_space() first.")
        
        self.dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        self.dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.L))

    def update_BCs(self, LB: Optional[np.ndarray] = None, RB: Optional[np.ndarray] = None):
        # self.LB = LB if LB is not None else self.LB
        # self.RB = RB if RB is not None else self.RB

        bc_L = fem.dirichletbc(self.LB, self.dofs_L, self.V)
        bc_R = fem.dirichletbc(self.RB, self.dofs_R, self.V)

        self.bcs = [bc_L, bc_R]

    def set_initial_condition(self):
        if self.V is None or self.u_n is None:
            raise ValueError("Function space not created. Call create_fem_space() first.")
        
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

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

    def set_variational_problem(self, dt: float):
        if self.u_n is None:
            raise ValueError("Initial condition not set. Call set_initial_condition() first.")
        
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(self.u_n, v) * ufl.dx
        L += dt * ufl.inner(self.FLW(self.u_n, dt), v.dx(0)) * ufl.dx # type: ignore
        L += (dt ** 2 / 2) * ufl.inner(ufl.dot(self.dB_dU(self.u_n), self.F(self.u_n).dx(0)), v) * ufl.dx # type: ignore
        L -= (dt ** 2 / 2) * ufl.inner(ufl.dot(self.H(self.u_n), self.F(self.u_n).dx(0)), v.dx(0)) * ufl.dx # type: ignore
        L -= dt * ufl.inner(self.BLW(self.u_n, dt), v) * ufl.dx # type: ignore

        self.bilinear_form = fem.form(a)
        self.linear_form = fem.form(L)

    def assemble_solver(self):
        if self.mesh is None or self.V is None:
            raise ValueError("Mesh or function space not created. Call create_mesh() and create_fem_space() first.")
        
        if self.bilinear_form is None or self.linear_form is None or self.bcs is None:
            raise ValueError("Variational problem not fully defined. Call set_variational_problem() and update_BCs() first.")

        self.A = petsc.assemble_matrix(self.bilinear_form, bcs=self.bcs)
        self.A.assemble()

        self.rhs = petsc.create_vector(self.linear_form)

        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def setup(self, h: float, dt: float):
        self.create_mesh(h)
        self.create_fem_space()
        self.set_boundary_dofs()
        self.update_BCs()
        self.set_initial_condition()
        self.set_variational_problem(dt)
        self.assemble_solver()

        self.h = h
        self.dt = dt

    def dU_dz(self, u: np.ndarray) -> np.ndarray:
        area = u[:, 0]
        flux = u[:, 1]

        # print("Last solution area in dU_dz:", area)
        # input("Press Enter to continue...")

        # print("Last solution flux in dU_dz:", flux)
        # input("Press Enter to continue...")

        # Assume uniform grid along z:
        z = np.linspace(0, self.L, len(area))

        dA_dz = np.gradient(area, z)
        dQ_dz = np.gradient(flux, z)

        return np.stack([dA_dz, dQ_dz], axis=1)  # shape (n, 2)

    def solve(self):
        if self.solver is None or self.rhs is None or self.u is None or self.u_n is None:
            raise ValueError("Solver not assembled. Call assemble_solver() first.")

        with self.rhs.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs, self.linear_form)

        petsc.apply_lifting(self.rhs, [self.bilinear_form], bcs=[self.bcs])
        self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self.rhs, self.bcs)

        self.solver.solve(self.rhs, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array[:]

        # print(f"Solution on rank {rank} for vessel {self.id}: {self.u.x.array}")
        # input("Press Enter to continue...")

        if np.any(np.isnan(self.u.x.array)):
            raise ValueError(f"NaN values encountered in the solution of vessel ID {self.id}.")

        
        
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

        self.last_sol = global_sol.copy()

        if (rank == 0) and (t - self.last_saved_time) >= 0.001:
            self.solutions["t"].append(t)
            self.solutions["A"].append(global_sol[:, 0])
            self.solutions["Q"].append(global_sol[:, 1])

            mid_index = len(global_sol) // 2
            self.middlepoints["A"].append(global_sol[mid_index, 0])
            self.middlepoints["Q"].append(global_sol[mid_index, 1])

            self.last_saved_time = t
