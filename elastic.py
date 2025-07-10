from vesselV2 import Vessel

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import petsc
import ufl
from basix.ufl import element
from petsc4py import PETSc

import numpy as np


class ElasticVessel(Vessel):
    """
    Class representing an elastic vessel.
    Inherits from the Vessel class.
    """
    def __init__(
        self, *args, **kwargs
    ):
        """
        Initialize the elastic vessel with its properties.
        Calls the parent class constructor.
        """
        super().__init__(*args, **kwargs)

        # Vessel boundary values
        self.LB = np.array([self.A0, 0], dtype=default_scalar_type)
        self.RB = np.array([self.A0, 0], dtype=default_scalar_type)


    def create_fem_space(self, element_type: str = "Lagrange"):
        """
        Create a finite element function space for the elastic vessel.
        Parameters:
        element_type (str): Type of finite element to use (default is "Lagrange").
        """

        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element(element_type, self.mesh.topology.cell_name(), 1, shape=(2, ))
        self.V = fem.functionspace(self.mesh, elem)

    def add_solution(self, u: fem.Function, save_all: bool = False):
        """
        Add a solution to the elastic vessel.
        Parameters:
        u (fem.Function): The solution function to be added.
        save_all (bool): Whether to save all solutions or just the last one.
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

    def set_variational_problem(self, dt: float):
        """
        Set the variational problem for the elastic vessel.
        Parameters:
        dt (float): Time step size for the simulation.
        """
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
        z = np.linspace(0, self.L, len(area))

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

    def P(self, U: np.ndarray, *args, **kwargs):
        assert U.shape == (2, )
        a, q = U

        return self.beta * ((np.sqrt(a) - self.A0 ** 0.5) / self.A0) + 0.5 * self.blood.rho * (q / a) ** 2
    
    def dP_dU(self, U: np.ndarray, *args, **kwargs):
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