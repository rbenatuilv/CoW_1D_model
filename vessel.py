from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import petsc
import ufl
from basix.ufl import element
from petsc4py import PETSc

import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import os

from dolfinx.cpp.graph import AdjacencyList_int32

def seq_graph_partitioner(comm, nparts, dual_graph, redistribute):
    print("dual_graph:")
    print(dual_graph)
    local_n   = len(dual_graph)
    counts    = comm.allgather(local_n)
    offset    = sum(counts[:comm.rank])
    global_id = offset + np.arange(local_n, dtype=np.int64)

    base, r   = divmod(sum(counts), nparts)
    owners    = np.concatenate([
        np.full(base + (i < r), i, dtype=np.int32)
        for i in range(nparts)
    ])

    return AdjacencyList_int32(owners[global_id])


class Blood:

    DYNAMIC_VISCOSITY = 0.045  # Poise (g/(cm.s))
    DENSITY = 1.050  # g/cm^3

    @property
    def mu(self):
        return self.DYNAMIC_VISCOSITY
    
    @property
    def rho(self):
        return self.DENSITY


class BloodVessel:

    GAMMA_PROFILE = 2
    POISSON_RATIO = 0.5 # Assuming incompressible material

    blood = Blood()

    def __init__(
        self, id: int, longitude: float, initial_area: float, 
        beta_coeff: float,
        left_bound: Literal["branch", "inflow", "outflow"] = "inflow",
        right_bound: Literal["branch", "inflow", "outflow"] = "outflow"
    ):
        
        self.id = id

        self.long = longitude
        self.A0 = initial_area 

        self.alpha = (self.GAMMA_PROFILE + 2) / (self.GAMMA_PROFILE + 1)
        # self.beta = np.sqrt(np.pi) * young_mod * wall_thick / (1 - self.POISSON_RATIO ** 2)
        self.beta = beta_coeff


        self.Kr = 2 * (self.GAMMA_PROFILE + 2) * np.pi * self.blood.mu / self.blood.rho

        self.LB_type = left_bound
        self.RB_type = right_bound

        self.LB = np.array([self.A0, 0], dtype=default_scalar_type)
        self.RB = np.array([self.A0, 0], dtype=default_scalar_type)
        self.bcs = []

        self.dofs_L = None
        self.dofs_R = None

        self.V = None
        self.Vprim = None
        self.n_dofs = 0
        self.n_dofs_prim = 0

        self.u_n = None  # Initial condition
        self.u = None

        self.bilinear = None
        self.linear = None
        self.A = None
        self.rhs = None
        self.solver = None

        self.mesh = None

        self.middlepoints = {
            "area": [],
            "flux": []
        }

        self.solutions = {
            "area": [],
            "flux": []
        }

        self.last_solution = {
            "area": None,
            "flux": None
        }

    def create_mesh(self, h: float):
        N = int(self.long / h)
        self.N = N
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, N, (0, self.long)) # , partitioner=seq_graph_partitioner

    def create_fem_space(self, element_type: str = "Lagrange"):
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element(element_type, self.mesh.topology.cell_name(), 1, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

        self.n_dofs = self.V.dofmap.index_map.size_global

        elem = element("DG", self.mesh.topology.cell_name(), 0, shape=(2, ))
        self.Vprim = fem.functionspace(self.mesh, elem)

        self.n_dofs_prim = self.Vprim.dofmap.index_map.size_global

    def set_boundary_dofs(self):
        if self.V is None:
            raise ValueError("Function space not set. Call create_fem_space() first.")
        local_range = self.V.dofmap.index_map.local_range
        self.dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        self.dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.long))
        
        # self.dofs_L = self.dofs_L[(self.dofs_L >= local_range[0]) & (self.dofs_L < local_range[1])]
        # self.dofs_R = self.dofs_R[(self.dofs_R >= local_range[0]) & (self.dofs_R < local_range[1])]

    def set_boundary_conditions(self):
        assert self.V is not None, "Function space not set. Call set_fem_space() first."

        bc_L = fem.dirichletbc(self.LB, self.dofs_L, self.V)
        bc_R = fem.dirichletbc(self.RB, self.dofs_R, self.V)
        self.bcs = [bc_L, bc_R]
        print(f"Boundary conditions set: {self.LB} --- y --- {self.RB}", flush=True)
        
        # # Apply area BC at left, flux BC at left
        # bc_L_area = fem.dirichletbc(self.LB[0], self.dofs_L, self.V.sub(0))
        # bc_L_flux = fem.dirichletbc(self.LB[1], self.dofs_L, self.V.sub(1))

        # # Apply area BC at right, flux BC at right
        # bc_R_area = fem.dirichletbc(self.RB[0], self.dofs_R, self.V.sub(0))
        # bc_R_flux = fem.dirichletbc(self.RB[1], self.dofs_R, self.V.sub(1))

        # self.bcs = [bc_L_area, bc_L_flux, bc_R_area, bc_R_flux]
        # if self.id == "1" and self.mesh.comm.rank == 0:
        #     print(f"Boundary conditions set: LB={self.LB}, RB={self.RB}", flush=True)

    def set_initial_conditions(self):
        assert self.V is not None, "Function space not set. Call create_fem_space() first."

        self.u_n = fem.Function(self.V)
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

        self.add_solution(self.u_n)


    def add_solution4(self, u: fem.Function):
        """
        Stores the vessel solution by gathering data on rank 0,
        correctly handling distributed DOFs and ghost cells.
        """
        comm = self.mesh.comm
        rank = comm.Get_rank()

        # Get the global indices of the owned DOFs on this rank
        dofmap = self.V.dofmap
        local_size = dofmap.index_map.local_range[1] - dofmap.index_map.local_range[0]
        owned_local_indices = np.arange(local_size, dtype=np.int32)
        owned_global_indices = dofmap.index_map.local_to_global(owned_local_indices)
        
        # Get all local DOFs (owned + ghosts) and their corresponding global indices
        all_local_indices = np.arange(dofmap.index_map.size_local, dtype=np.int32)
        all_global_indices = dofmap.index_map.local_to_global(all_local_indices)

        # Get the coordinates for all local DOFs
        all_local_coords = self.V.tabulate_dof_coordinates()[:, 0]
        
        # Get the values for the owned DOFs using their global indices
        # This is a safe way to get values regardless of local indexing.
        owned_values = u.x.petsc_vec.getValues(owned_global_indices.astype(np.int32))
        
        # Now, find the coordinates that correspond to the owned DOFs
        owned_coords = all_local_coords[np.isin(all_global_indices, owned_global_indices)]

        # Combine the coordinates and values into a single array
        local_data = np.vstack((owned_coords, owned_values)).T

        if rank == 0:
            all_data = comm.gather(local_data, root=0)
            global_data = np.concatenate(all_data)
            sorted_data = global_data[np.argsort(global_data[:, 0])]
            
            n_dofs = sorted_data.shape[0]
            n_components = self.V.dofmap.index_map_bs
            solution_components = sorted_data[:, 1].reshape(n_dofs // n_components, n_components)

            self.last_solution["area"] = solution_components[:, 0]
            self.last_solution["flux"] = solution_components[:, 1]
    def add_solution(self, u: fem.Function):
        """
        Stores the vessel solution by gathering data on rank 0,
        correctly handling distributed DOFs and ghost cells.
        """
        comm = self.mesh.comm
        rank = comm.Get_rank()

        local_size = self.V.dofmap.index_map.local_range[1] - self.V.dofmap.index_map.local_range[0]
        owned_dof_indices_local = np.arange(local_size, dtype=np.int32)
        owned_dof_indices_global = self.V.dofmap.index_map.local_to_global(owned_dof_indices_local)



        # Get the local DOFs and their physical coordinates (including ghosts)
        all_local_coords = self.V.tabulate_dof_coordinates()[:, 0]
        print(f"rank {rank}: ",all_local_coords, flush=True)
        
        # Get the local DOF indices that are owned by this rank.
        # We use a numpy array to match the PETSc API.
        print(f"rank {rank}: ",self.V.dofmap.index_map.local_range, flush=True)
        print(f"rank {rank}: ",owned_dof_indices_local, flush=True)
        # Get the values for the owned DOFs
        owned_values = u.x.petsc_vec.getValues(owned_dof_indices_global.astype(np.int32))
        
        # Get the coordinates for the owned DOFs by using the local-to-global mapping
        # to find which local coordinates correspond to the owned DOFs
        # This ensures owned_coords and owned_values have the same size.
        owned_coords = all_local_coords[owned_dof_indices_local]

        # Combine the local coordinates and solution values into a single array.
        local_data = np.vstack((owned_coords, owned_values)).T

        if rank == 0:
            # Gather the data from all ranks.
            all_data = comm.gather(local_data, root=0)
            
            # Concatenate all gathered arrays into a single global array.
            global_data = np.concatenate(all_data)
            
            # Sort the data based on the x-coordinate to get a physically ordered solution.
            sorted_data = global_data[np.argsort(global_data[:, 0])]
            
            # Separate into area and flux vectors.
            n_dofs = sorted_data.shape[0]
            n_components = self.V.dofmap.index_map_bs
            solution_components = sorted_data[:, 1].reshape(n_dofs // n_components, n_components)

            self.last_solution["area"] = solution_components[:, 0]
            self.last_solution["flux"] = solution_components[:, 1]

    def add_solution3(self, u: fem.Function, save_all: bool = True, debug: bool = True):
        """
        Store the vessel solution, with rank 0 holding the full global vector
        for inflow/outflow BC computation.
        """
        local_range = self.V.dofmap.index_map.local_range
        x = self.mesh.geometry.x[local_range[0]:local_range[1], 0]  # 1D mesh, take first column
        print(x, flush=True)
        comm = self.mesh.comm
        local_coords = x.copy()
        all_coords = comm.allgather(local_coords)
        global_coords = np.concatenate(all_coords)

        with u.x.petsc_vec.localForm() as lf:
            owned_array = lf.array.copy()
        vec = u.x.petsc_vec
        size = vec.getSize()

        if self.id == "1" and debug:
            # print("----- add_solution called -----")
            print(f"Owned array on rank {self.mesh.comm.Get_rank()}: {owned_array}", flush=True)


        start, end = vec.getOwnershipRange()
        local_n = end - start
        # --- Gather lengths to compute displacements ---
        owned_values = vec.getValues(range(start, end))
        all_lengths = comm.allgather(local_n)
        displs = np.cumsum([0] + all_lengths[:-1])
        global_array = None
        if comm.rank == 0:
            global_array = np.empty(sum(all_lengths), dtype=owned_values.dtype)
            
        # Gather all owned values to rank 0
        comm.Gatherv(sendbuf=owned_values,
                recvbuf=(global_array, all_lengths, displs, MPI.DOUBLE),
                root=0)
    
        if comm.rank == 0:
            n_total = len(global_array) // 2
            self.last_solution["area"] = global_array[0::2].copy()
            self.last_solution["flux"] = global_array[1::2].copy()

        # if comm.rank == 0:
        #     # n = size // 2  # assuming 2-component system
        #     # self.last_solution["area"] = global_array[0::2].copy()
        #     # self.last_solution["flux"] = global_array[1::2].copy()
        #     # Sort by coordinates
        #     print(global_array.size, flush=True)
        #     n_nodes = int(global_array.size / 2)
        #     global_array_reshaped = global_array.reshape(n_nodes, 2)
        #     sort_idx = np.argsort(global_coords)
        #     print(f"Sort idx: {sort_idx}", flush=True)
            
        #     sorted_global_array = global_array_reshaped[sort_idx, 0]  # area
        #     sorted_global_array_flux = global_array_reshaped[sort_idx, 1]  # flux
        #     self.last_solution["area"] = sorted_global_array.copy()
        #     self.last_solution["flux"] = sorted_global_array_flux.copy()
        
        # if comm.rank == 0:
        #     # The number of nodes is half the global array size
        #     n_nodes = global_array.size // 2
        #     global_array_reshaped = global_array.reshape(n_nodes, 2)
        #     # Get the coordinates for these nodes
        #     # Gather all node coordinates from all ranks
        #     all_coords = comm.allgather(self.mesh.geometry.x[:, 0])
        #     global_coords = np.concatenate(all_coords)
        #     # Remove duplicates and sort
        #     global_coords_unique = np.unique(global_coords)
        #     # Now sort the solution by coordinates
        #     sort_idx = np.argsort(global_coords_unique)
        #     sorted_area = global_array_reshaped[sort_idx, 0]
        #     sorted_flux = global_array_reshaped[sort_idx, 1]
        #     self.last_solution["area"] = sorted_area.copy()
        #     self.last_solution["flux"] = sorted_flux.copy()
        # Optionally save all timesteps
        if save_all and comm.rank == 0:
            if "area_all" not in self.solutions:
                self.solutions["area_all"] = []
                self.solutions["flux_all"] = []
            if comm.rank == 0:
                self.solutions["area_all"].append(self.last_solution["area"].copy())
                self.solutions["flux_all"].append(self.last_solution["flux"].copy())

    def add_solution2(self, u: fem.Function, save_all: bool = True):
        assert u.ufl_shape == (2, ), "Solution must be a vector of size 2."

        comm = self.mesh.comm
        rank = comm.Get_rank()

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 2)

        # 4) Store the global solution in the last_solution attribute
        self.last_solution["area"] = global_sol[:, 0]
        self.last_solution["flux"] = global_sol[:, 1]

        # 5) Append the global solution to the solutions dictionary if save_all is True
        if save_all and rank == 0:
            self.solutions["area"].append(global_sol[:, 0])
            self.solutions["flux"].append(global_sol[:, 1])

            self.middlepoints["area"].append(global_sol[len(global_sol) // 2, 0])
            self.middlepoints["flux"].append(global_sol[len(global_sol) // 2, 1])


    def set_variational_problem(self, dt: float):
        assert self.V is not None, "Function space not set. Call create_fem_space() first."
        assert self.bcs, "Boundary conditions not set. Call set_boundary_conditions() first."

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(self.u_n, v) * ufl.dx
        L += dt * ufl.inner(self.FLW(self.u_n, dt), v.dx(0)) * ufl.dx
        L += (dt ** 2 / 2) * ufl.inner(ufl.dot(self.dB_dU(self.u_n), self.F(self.u_n).dx(0)), v) * ufl.dx
        L -= (dt ** 2 / 2) * ufl.inner(ufl.dot(self.H(self.u_n), self.F(self.u_n).dx(0)), v.dx(0)) * ufl.dx
        L -= dt * ufl.inner(self.BLW(self.u_n, dt), v) * ufl.dx

        self.bilinear = fem.form(a)
        self.linear = fem.form(L)

        self.A = petsc.assemble_matrix(self.bilinear, bcs=self.bcs)
        self.A.assemble()
        
        self.rhs = petsc.create_vector(self.linear)

        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

        self.u = fem.Function(self.V)
        self.u.x.array[:] = self.u_n.x.array

    def initial_setup(self, h: float, dt: float):
        self.create_mesh(h)
        self.create_fem_space()
        self.set_boundary_dofs()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_variational_problem(dt)

    def save_middlepoint_plot(self, T: float, quantity: Literal["area", "flux"], filename: str):

        assert quantity in self.solutions, f"Invalid quantity: {quantity}. Available: {list(self.solutions.keys())}"

        data = self.middlepoints[quantity]
        if not data:
            raise ValueError(f"No solutions available for {quantity}.")

        unit = "cm^2" if quantity == "area" else "cm^3/s"

        middle_point_sol = np.array(data)
        x_values = np.linspace(0, T, len(middle_point_sol))
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, middle_point_sol, color='blue', label=f'Middle Point {quantity.capitalize()}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'{quantity.capitalize()} ({unit})')
        plt.title(f'Middle point {quantity.capitalize()} over time for Vessel {self.id}')
        plt.grid()
        plt.legend()
        plt.savefig(filename, dpi=300)

    def save_solution(self, dirname: str):

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, f"vessel_{self.id}_solutions.npz")

        # Save a pkl file with the solutions
        with open(filename, 'wb') as f:
            np.savez(f, area=np.array(self.solutions["area"]), flux=np.array(self.solutions["flux"]))

    ####### Methods for the variational problem #######

    def B(self, U: fem.Function): 
        assert U.ufl_shape == (2, )

        return ufl.as_vector([
            0,
            self.Kr * (U[1] / U[0])
        ])
    
    def BLW(self, U: fem.Function, dt: float):
        assert U.ufl_shape == (2, )

        return self.B(U) - (dt / 2) * ufl.dot(self.dB_dU(U), self.B(U))
    
    def dB_dU(self, U: fem.Function):
        assert U.ufl_shape == (2, )

        return ufl.as_tensor([
            [0, 0],
            [- self.Kr * (U[1] / U[0] ** 2), self.Kr / U[0]]
        ])
    
    def c2(self, U: fem.Function):
        assert U.ufl_shape == (2, )

        return (self.beta / (2 * self.blood.rho * self.A0)) * U[0] ** 0.5
    
    def C(self, U: fem.Function):
        assert U.ufl_shape == (2, )

        return (self.beta / (3 * self.blood.rho)) * (U[0] ** 1.5 / self.A0 - self.A0 ** 0.5)

    def F(self, U: fem.Function):
        assert U.ufl_shape == (2, )

        return ufl.as_vector([
            U[1],
            self.alpha * U[1] ** 2 / U[0] + self.C(U)
        ])
    
    def FLW(self, U: fem.Function, dt: float):
        assert U.ufl_shape == (2, )

        return self.F(U) - (dt / 2) * ufl.dot(self.H(U), self.B(U))
    
    def H(self, U: fem.Function):
        assert U.ufl_shape == (2, )

        return ufl.as_tensor([
            [0, 1],
            [self.c2(U) - self.alpha * (U[1] / U[0]) ** 2, 2 * self.alpha * (U[1] / U[0])]
        ])
    
    
    ########### Numpy methods for BC problem #############

    def B_np(self, U: np.ndarray):
        assert U.shape == (2, )

        return np.array([
            0,
            self.Kr * (U[1] / U[0])
        ])
    
    def c2_np(self, U: np.ndarray):
        assert U.shape == (2, )

        return (self.beta / (2 * self.blood.rho * self.A0)) * np.sqrt(U[0])

    def c_alpha(self, U: np.ndarray):
        assert U.shape == (2, )

        return np.sqrt(self.c2_np(U) + self.alpha * (self.alpha - 1) * (U[1] / U[0]) ** 2)
    
    def CC(self, U: np.ndarray, dU: np.ndarray, dt: float):
        assert U.shape == (2, )

        return U - dt * self.H_np(U) @ dU - dt * self.B_np(U)

    def H_np(self, U: np.ndarray):
        assert U.shape == (2, )

        return np.array([
            [0, 1],
            [self.c2_np(U) - self.alpha * (U[1] / U[0]) ** 2, 2 * self.alpha * (U[1] / U[0])]
        ])
    
    def I1(self, U: np.ndarray):
        assert U.shape == (2, )

        return np.array([
            self.c_alpha(U) - self.alpha * (U[1] / U[0]),
            1
        ])
    
    def I2(self, U: np.ndarray):
        assert U.shape == (2, )
        return np.array([
            - self.c_alpha(U) - self.alpha * (U[1] / U[0]),
            1
        ])
    
    def project(self, fun: fem.Function, V):

        u, v = ufl.TestFunction(V), ufl.TrialFunction(V)
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(fun, v) * ufl.dx
        problem = petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()

        return problem.u

    def dU_dz(self):
        area = self.last_solution["area"]
        flux = self.last_solution["flux"]

        # Assume uniform grid along z:
        z = np.linspace(0, self.long, len(area))  # or use actual coordinates if nonuniform

        dA_dz = np.gradient(area, z)
        dQ_dz = np.gradient(flux, z)

        return np.stack([dA_dz, dQ_dz], axis=1)  # shape (n, 2)
    
    def Y0(self, U: np.ndarray, dU: np.ndarray, dt: float):
        assert U.shape == (2, )

        return np.array([
            self.I2(U) @ self.CC(U, dU, dt),
            self.I1(U) @ (U - dt * self.B_np(U))
        ])
    
    def YL(self, U: np.ndarray, dU: np.ndarray, dt: float):
        assert U.shape == (2, )

        return np.array([
            self.I1(U) @ self.CC(U, dU, dt),
            self.I2(U) @ (U - dt * self.B_np(U))
        ])

    def W0(self, U: np.ndarray):
        assert U.shape == (2, )

        return (1 / (2 * self.c_alpha(U))) * np.array([
            [-1, 1],
            [self.c_alpha(U) - self.alpha * (U[1] / U[0]), self.c_alpha(U) + self.alpha * (U[1] / U[0])],
        ])

    def WL(self, U: np.ndarray):
        assert U.shape == (2, )

        return (1 / (2 * self.c_alpha(U))) * np.array([
            [1, -1],
            [self.c_alpha(U) + self.alpha * (U[1] / U[0]), self.c_alpha(U) - self.alpha * (U[1] / U[0])]
        ])
    
    #### Methods for branching ####

    def P(self, U: np.ndarray):
        assert U.shape == (2, )
        a, q = U

        return self.beta * ((np.sqrt(a) - self.A0 ** 0.5) / self.A0) + 0.5 * (q / a) ** 2
    
    def dP_dU(self, U: np.ndarray):
        assert U.shape == (2, )
        a, q = U

        dP_da = self.beta / (2 * self.A0 * np.sqrt(a)) - (q ** 2) / (a ** 3)
        dP_dq = q / a ** 2

        return np.array([dP_da, dP_dq])

    def f_branch(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        assert U.shape == (2, )

        a, q = U

        return np.sign(q) * gamma * (q / a) ** 2 * np.sqrt(2 * (1 - np.cos(theta)))
    
    def df_dU(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        assert U.shape == (2, )
        a, q = U

        dfdU_a = -2 * gamma * ((q ** 2) / (a ** 3)) * np.sqrt(2 * (1 - np.cos(theta)))
        dfdU_q = 2 * gamma * q / a ** 2 * np.sqrt(2 * (1 - np.cos(theta)))

        return np.array([dfdU_a, dfdU_q])

    def CCR(self, dt: float):
        uR = self.RB
        dU_dz_R = self.dU_dz()[-1]

        return self.CC(uR, dU_dz_R, dt)
    
    def CCL(self, dt: float):
        uL = self.LB
        dU_dz_L = self.dU_dz()[0]

        return self.CC(uL, dU_dz_L, dt)



class VesselSystem:
    def __init__(self, vessels_data: dict, bifurcations_data: dict):
        self.vessels = {}
        self.bifurcations = bifurcations_data

        for id, data in vessels_data.items():
            vessel = BloodVessel(id=id, **data)
            self.vessels[id] = vessel

    def setup(self, h: float, dt: float):
        for vessel in self.vessels.values():
            vessel.initial_setup(h, dt)

        MPI.COMM_WORLD.barrier()  # Ensure all processes are synchronized before proceeding

    def set_inflows(self, inflows: dict[int, callable]):
        self.inflows = inflows


if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx import mesh
    from basix.ufl import element

    L = 1.0
    n = 10

    # Create a 1D mesh with n intervals
    domain = mesh.create_interval(MPI.COMM_WORLD, n, (0, L))

    # Create a function space on the mesh. Note that the element is
    # a vector-valued Lagrange element of degree 1, with 2 components
    # (for the two components of the vector field).
    elem = element("Lagrange", domain.topology.cell_name(), 1, shape=(2, ))
    V = fem.functionspace(domain, elem)

    U = fem.Function(V)

    blood_vessel = BloodVessel(1, 1, 1, 1)
    F = blood_vessel.F(U)
    


    

