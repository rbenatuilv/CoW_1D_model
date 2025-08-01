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


class Blood:
    """
    Class to save blood properties.
    Properties:
    DYNAMIC_VISCOSITY: 0.045 Poise (g/(cm.s))
    DENSITY: 1.050 g/cm^3
    These values are taken from the literature and are typical for human blood.
    """

    DYNAMIC_VISCOSITY = 0.045  # Poise (g/(cm.s))
    DENSITY = 1.050  # g/cm^3

    @property
    def mu(self):
        return self.DYNAMIC_VISCOSITY
    
    @property
    def rho(self):
        return self.DENSITY


class BloodVessel:
    """
    Class that simulates a 1D blood vessel.
    """

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
        self.n_dofs = 0

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
        """Create a 1D mesh for the blood vessel."""

        N = int(self.long / h)
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, N, (0, self.long))

    def create_fem_space(self, element_type: str = "Lagrange"):
        """Create a finite element function space for the blood vessel."""

        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element(element_type, self.mesh.topology.cell_name(), 1, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

        self.n_dofs = self.V.dofmap.index_map.size_global

    def set_boundary_dofs(self):
        """Set the degrees of freedom for the left and right boundaries."""

        if self.V is None:
            raise ValueError("Function space not set. Call create_fem_space() first.")

        self.dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        self.dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.long))

    def set_boundary_conditions(self):
        """
        Set the boundary conditions for the left and right boundaries.
        """

        assert self.V is not None, "Function space not set. Call set_fem_space() first."

        bc_L = fem.dirichletbc(self.LB, self.dofs_L, self.V)
        bc_R = fem.dirichletbc(self.RB, self.dofs_R, self.V)
        self.bcs = [bc_L, bc_R]

    def set_initial_conditions(self):
        """Set the initial conditions for the blood vessel."""

        assert self.V is not None, "Function space not set. Call create_fem_space() first."

        self.u_n = fem.Function(self.V)
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

        self.add_solution(self.u_n)

    def add_solution(self, u: fem.Function, save_all: bool = True):
        """
        Add a solution to the vessel. `save_all` determines if all solutions are saved
        to the solutions dictionary or only the last one.
        """

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


    def set_variational_problem(self, dt: float, method: Literal["CG", "DG"] = "CG"):
        """
        Set up the variational problem for the blood vessel.
        This method creates the bilinear and linear forms, assembles the matrix and vector,
        and sets up the solver.
        """

        assert self.V is not None, "Function space not set. Call create_fem_space() first."
        assert self.bcs, "Boundary conditions not set. Call set_boundary_conditions() first."

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        if method == "CG":
            a = ufl.inner(u, v) * ufl.dx
            L = ufl.inner(self.u_n, v) * ufl.dx
            L += dt * ufl.inner(self.FLW(self.u_n, dt), v.dx(0)) * ufl.dx
            L += (dt ** 2 / 2) * ufl.inner(ufl.dot(self.dB_dU(self.u_n), self.F(self.u_n).dx(0)), v) * ufl.dx
            L -= (dt ** 2 / 2) * ufl.inner(ufl.dot(self.H(self.u_n), self.F(self.u_n).dx(0)), v.dx(0)) * ufl.dx
            L -= dt * ufl.inner(self.BLW(self.u_n, dt), v) * ufl.dx
        
        elif method == "DG":

            flux_int = self.numflux(self.u_n('+'), self.u_n('-'), 10)

            a = ufl.inner(u, v) * ufl.dx
            L = ufl.inner(self.u_n, v) * ufl.dx
            L += dt * ufl.inner(self.F(self.u_n, dt), v.dx(0)) * ufl.dx
            L -= dt * ufl.inner(self.B(self.u_n), v) * ufl.dx
            L -= dt * (flux_int * v('+') - flux_int * v('-')) * ufl.dS


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

    def initial_setup(self, h: float, dt: float, method: Literal["CG", "DG"] = "CG"):
        """
        Initial setup for the blood vessel.
        This method creates the mesh, sets up the finite element space, boundary conditions,
        initial conditions, and the variational problem.
        """

        self.create_mesh(h, method)
        self.create_fem_space()
        self.set_boundary_dofs()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_variational_problem(dt, method)

    def save_middlepoint_plot(self, T: float, quantity: Literal["area", "flux"], filename: str):
        """
        Save a plot of the middle point solution for the specified quantity (area or flux).
        Args:
            T (float): Total time for the simulation.
            quantity (str): The quantity to plot ("area" or "flux").
            filename (str): The filename to save the plot.
        """

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
        """
        Save the solutions of the vessel to a file.
        Args:
            dirname (str): Directory where the solutions will be saved.
        """

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
    
    def numflux(self, u_minus, u_plus, alpha):
        return 0.5*(self.F(u_minus) + self.F(u_plus)) - 0.5 * alpha * (u_plus - u_minus)
    
    
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
        z = np.linspace(0, self.long, len(area))

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

        return self.beta * ((np.sqrt(a) - self.A0 ** 0.5) / self.A0) + 0.5 * self.blood.rho * (q / a) ** 2
    
    def dP_dU(self, U: np.ndarray):
        assert U.shape == (2, )
        a, q = U

        dP_da = self.beta / (2 * self.A0 * np.sqrt(a)) - self.blood.rho * (q ** 2) / (a ** 3)
        dP_dq = self.blood.rho * q / a ** 2

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
    """
    Class that represents a system of blood vessels and bifurcations.
    It contains methods to initialize the vessels, set up the system, and manage inflows.
    """

    def __init__(self, vessels_data: dict, bifurcations_data: dict):
        self.vessels = {}
        self.bifurcations = bifurcations_data

        for id, data in vessels_data.items():
            vessel = BloodVessel(id=id, **data)
            self.vessels[id] = vessel

    def setup(self, h: float, dt: float):
        """
        Set up the system of vessels.
        This method initializes each vessel with the given mesh size `h` and time step `dt`.
        It creates the mesh, sets up the finite element space, boundary conditions,
        initial conditions, and the variational problem for each vessel.
        """

        for vessel in self.vessels.values():
            vessel.initial_setup(h, dt)

        MPI.COMM_WORLD.barrier()  # Ensure all processes are synchronized before proceeding

    def set_inflows(self, inflows: dict[int, callable]):
        """
        Set the inflows for the vessels.
        Args:
            inflows (dict[int, callable]): A dictionary where keys are vessel IDs and values are functions
                                            that define the inflow conditions for each vessel.
        """

        self.inflows = inflows
