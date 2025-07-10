from vesselV2 import Vessel

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import petsc
import ufl
from basix.ufl import element
from petsc4py import PETSc

import numpy as np


class ViscoelasticVessel(Vessel):
    """
    Class representing a viscoelastic vessel.
    Inherits from the Vessel class.
    """

    VISCOELASTIC_PARAM = 3 * 1e5
    SMOOTH_MUSCLE_FRAC = 0.1

    def __init__(self, *args, eps: float = 1e-3, **kwargs):
        """
        Initialize the viscoelastic vessel with its properties.
        Calls the parent class constructor.
        """
        super().__init__(*args, **kwargs)

        self.eps = eps  # Relaxation parameter

        # Vessel boundary values
        self.LB = np.array([self.A0, 0, 0], dtype=default_scalar_type)
        self.RB = np.array([self.A0, 0, 0], dtype=default_scalar_type)

        self.radius = np.sqrt(self.A0 / np.pi) 
        self.thick = 0.1 * self.radius  # Assuming a constant thickness for simplicity

        self.gamma_visc = self.VISCOELASTIC_PARAM * self.SMOOTH_MUSCLE_FRAC
        self.Gamma = (2/3) * np.sqrt(np.pi) * self.gamma_visc * self.thick

        self.middlepoints = {
            "area": [],
            "flux": [],
            "xi": []
        }

        self.solutions = {
            "area": [],
            "flux": [],
            "xi": []
        }

        self.last_solution = {
            "area": None,
            "flux": None,
            "xi": None
        }


    def create_fem_space(self, element_type: str = "Lagrange"):
        """
        Create a finite element function space for the viscoelastic vessel.
        Parameters:
        element_type (str): Type of finite element to use (default is "Lagrange").
        """

        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element(element_type, self.mesh.topology.cell_name(), 1, shape=(3, ))
        self.V = fem.functionspace(self.mesh, elem)

    def add_solution(self, u: fem.Function, save_all: bool = False):
        """
        Add a solution to the viscoelastic vessel.
        Parameters:
        u (fem.Function): The solution function to be added.
        save_all (bool): Whether to save all solutions or just the last one.
        """
        assert u.ufl_shape == (3, ), "Solution must be a vector of size 2."

        comm = self.mesh.comm
        rank = comm.Get_rank()

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        uxi_loc = u.sub(2).collapse().x.array    # xi component
        local_sol = np.stack([uA_loc, uQ_loc, uxi_loc], axis=-1)  # shape (n_local, 3)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 3)

        # 4) Store the global solution in the last_solution attribute
        self.last_solution["area"] = global_sol[:, 0]
        self.last_solution["flux"] = global_sol[:, 1]
        self.last_solution["xi"] = global_sol[:, 2]

        # 5) Append the global solution to the solutions dictionary if save_all is True
        if save_all and rank == 0:
            self.solutions["area"].append(global_sol[:, 0])
            self.solutions["flux"].append(global_sol[:, 1])
            self.solutions["xi"].append(global_sol[:, 2])

            self.middlepoints["area"].append(global_sol[len(global_sol) // 2, 0])
            self.middlepoints["flux"].append(global_sol[len(global_sol) // 2, 1])
            self.middlepoints["xi"].append(global_sol[len(global_sol) // 2, 2])


    def set_variational_problem(self, dt):
        """
        Set the variational problem for the viscoelastic vessel.
        This method defines the weak form of the equations governing the viscoelastic behavior.
        """
        if self.V is None:
            raise ValueError("Finite element space not created. Call create_fem_space() first.")

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        ####
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(self.u_n, v) * ufl.dx
        L += dt * ufl.inner(self.S(self.u_n) + self.HdxU(self.u_n), v) * ufl.dx
        L += (dt**2 / 2) * ufl.inner(self.dS_dU(self.u_n) * (self.S(self.u_n) + self.HdxU(self.u_n)), v) * ufl.dx
        
        L += (dt**2 / 2) * ufl.inner(
            ufl.dot(
                ufl.dot(self.dH_dU(self.u_n), self.S(self.u_n) + self.HdxU(self.u_n)),
                self.u_n.dx(0)
            ),
            v
        ) * ufl.dx

        L -= (dt**2 / 2) * ufl.inner(
            ufl.dot(
                ufl.dot(self.dH_dU(self.u_n), self.u_n.dx(0)),
                self.S(self.u_n) + self.HdxU(self.u_n)
            ),
            v
        ) * ufl.dx
        
        L -= (dt**2 / 2) * ufl.inner(self.H(self.u_n) * (self.S(self.u_n) + self.HdxU(self.u_n)), v.dx(0)) * ufl.dx
        ####

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


    ########### Methods for setting up the variational problem ###########

    def phi(self, u: fem.Function):
        """
        Calculate the phi function for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return self.beta * (ufl.sqrt(u[0]) - np.sqrt(self.A0)) / self.A0
    
    def dphi_dA(self, u: fem.Function):
        """
        Derivative of the phi function with respect to area.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return self.beta / (2 * self.A0 * ufl.sqrt(u[0]))
    
    def d2_phi_dA2(self, u: fem.Function):
        """
        Second derivative of the phi function with respect to area.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return -self.beta / (4 * self.A0 * u[0] ** (3/2))
    
    def psi(self, u: fem.Function):
        """
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return self.Gamma / (self.A0 * ufl.sqrt(u[0]))
    
    def dpsi_dA(self, u: fem.Function):
        """
        Derivative of the psi function with respect to area.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return -self.Gamma / (2 * self.A0 * u[0] ** (3/2))
    
    def vel(self, u: fem.Function):
        """
        Calculate the velocity of the fluid in the vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return u[1] / u[0]
    
    def c2(self, u: fem.Function):
        """
        Calculate the c2 coefficient for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return (u[0] / self.blood.rho) * self.dphi_dA(u)
    
    def dc2_dA(self, u: fem.Function):
        """
        Derivative of the c2 coefficient with respect to area.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return (1 / self.blood.rho) * self.dphi_dA(u) + (u[0] / self.blood.rho) * self.d2_phi_dA2(u)
    
    def alpha_gamma(self, u: fem.Function):
        """
        Calculate the alpha gamma term for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return self.psi(u) * u[2] / self.blood.rho
    
    def dalpha_gamma_dA(self, u: fem.Function):
        """
        Derivative of the alpha gamma term with respect to area.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return (u[2] / self.blood.rho) * self.dpsi_dA(u)
    
    def force(self, u: fem.Function):
        """
        Calculate the force term for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return self.Kr * self.vel(u)
    
    def H(self, u: fem.Function):
        """
        Calculate the H matrix for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return ufl.as_tensor([
            [0, 1, 0],
            [self.c2(u) - self.alpha * self.vel(u) ** 2 + self.alpha_gamma(u) / 2, 2 * self.alpha * self.vel(u), -(u[0] / self.blood.rho) * self.psi(u)],
            [0, -1 / self.eps, 0]
        ])
    
    def S(self, u: fem.Function):
        """
        Calculate the S vector for the viscoelastic vessel.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return ufl.as_vector([
            0, 
            self.force(u),
            (1 / self.eps) * u[2]
        ])
    
    def dS_dU(self, u: fem.Function):
        """
        Calculate the derivative of the S vector with respect to the solution vector.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return ufl.as_tensor([
            [0, 0, 0],
            [- self.Kr * (u[1] / u[0] ** 2), self.Kr / u[0], 0],
            [0, 0, 1 / self.eps]
        ])

    def HdxU(self, u: fem.Function):
        """
        Calculate the H matrix multiplied by the derivative of the solution vector.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return ufl.dot(self.H(u), u.dx(0))

    def dH_dU(self, u: fem.Function):
        """
        Calculate the derivative of the H matrix with respect to the solution vector.
        """
        assert u.ufl_shape == (3, ), "Input function must have shape (3,)."

        return ufl.as_tensor([
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],

            [
                [
                    self.dc2_dA(u) + 2 * self.alpha * self.vel(u) * (u[1] / u[0] ** 2) + self.dalpha_gamma_dA(u) / 2,
                    -2 * self.alpha * self.vel(u) / u[0],
                    self.psi(u) / (2 * self.blood.rho)
                ],
                [
                    -2 * self.alpha * (u[1] / u[0] ** 2),
                    2 * self.alpha / u[0],
                    0
                ],
                [
                    -self.psi(u) / self.blood.rho - self.dpsi_dA(u) * u[0] / self.blood.rho,
                    0,
                    0,
                ],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            
        ])
    
    ########### Numpy methods for BC problem #############

    def S_np(self, U: np.ndarray):
        """
        Calculate the B vector for the viscoelastic vessel in numpy format.
        """
        return np.array([
            0,
            self.Kr * (U[1] / U[0]),
            (1 / self.eps) * U[2]
        ])
    
    def c2_np(self, U: np.ndarray):
        """
        Calculate the c2 coefficient for the viscoelastic vessel in numpy format.
        """
        return (np.sqrt(U[0]) / self.blood.rho) * (self.beta / (2 * self.A0))
    
    def phi_np(self, U: np.ndarray):
        """
        Calculate the phi function for the viscoelastic vessel in numpy format.
        """
        return self.beta * (np.sqrt(U[0]) - np.sqrt(self.A0)) / self.A0
    
    def dphi_dA_np(self, U: np.ndarray):
        """
        Derivative of the phi function with respect to area in numpy format.
        """
        return self.beta / (2 * self.A0 * np.sqrt(U[0]))

    def psi_np(self, U: np.ndarray):
        """
        Calculate the psi function for the viscoelastic vessel in numpy format.
        """
        return self.Gamma / (self.A0 * np.sqrt(U[0]))

    def alpha_gamma_np(self, U: np.ndarray):
        """
        Calculate the alpha gamma term for the viscoelastic vessel in numpy format.
        """
        return self.psi_np(U) * U[2] / self.blood.rho

    def H_np(self, U: np.ndarray):
        """
        Calculate the H matrix for the viscoelastic vessel in numpy format.
        """

        return np.array([
            [0, 1, 0],
            [self.c2_np(U) - self.alpha * (U[1] / U[0]) ** 2 + self.alpha_gamma_np(U) / 2, 
             2 * self.alpha * (U[1] / U[0]), 
             -(U[0] / self.blood.rho) * self.psi_np(U)],
            [0, -1 / self.eps, 0]
        ])
    
    def dU_dz(self):
        area = self.last_solution["area"]
        flux = self.last_solution["flux"]
        xi = self.last_solution["xi"]

        z = np.linspace(0, self.L, len(area))
        dA_dz = np.gradient(area, z)
        dQ_dz = np.gradient(flux, z)
        dxi_dz = np.gradient(xi, z)

        return np.stack([dA_dz, dQ_dz, dxi_dz], axis=1)  # shape (n, 3)
    
    def omega(self, u: np.ndarray):
        return (self.psi_np(u) * u[0]) / (self.blood.rho * self.eps) + self.alpha_gamma_np(u) / 2
    
    def c_tilde(self, u: np.ndarray):
        """
        Calculate the c_tilde coefficient for the viscoelastic vessel in numpy format.
        """
        return np.sqrt(self.c2_np(u) + self.omega(u))

    def CC(self, u: np.ndarray, du: np.ndarray, dt: float):
        """
        Calculate the CC matrix for the viscoelastic vessel in numpy format.
        This matrix is used to compute the next state of the system.
        """
        return u - dt * np.dot(self.H_np(u), du) - dt * self.S_np(u)

    def CCR(self, dt: float):
        """
        Calculate the CC matrix for the viscoelastic vessel in numpy format.
        """
        uR = self.RB
        dU_dz_R = self.dU_dz()[-1]

        return uR - dt * np.dot(self.H_np(uR), dU_dz_R) - dt * self.S_np(uR)
    
    def CCL(self, dt: float):
        """
        Calculate the CC matrix for the viscoelastic vessel in numpy format.
        """
        uL = self.LB
        dU_dz_L = self.dU_dz()[0]

        return uL - dt * np.dot(self.H_np(uL), dU_dz_L) - dt * self.S_np(uL)
    
    def I1(self, u: np.ndarray):
        """
        Calculate the I1 eigenvector for the viscoelastic vessel in numpy format.
        """
        return np.array([
            1,
            u[1] / u[0] - self.c_tilde(u),
            0
        ])
    
    def I2(self, u: np.ndarray):
        """
        Calculate the I2 eigenvector for the viscoelastic vessel in numpy format.
        """
        return np.array([
            1,
            u[1] / u[0] + self.c_tilde(u),
            0
        ])
    
    def xiR(self, dt: float):
        """
        Calculate the xiR value for the viscoelastic vessel in numpy format.
        """
        uR = self.RB
        dU_dz_R = self.dU_dz()[-1]
        dQ_dz_R = dU_dz_R[1]

        return uR[2] + (dt / self.eps) * (dQ_dz_R - uR[2])
    
    def xiL(self, dt: float):
        """
        Calculate the xiL value for the viscoelastic vessel in numpy format.
        """
        uL = self.LB
        dU_dz_L = self.dU_dz()[0]
        dQ_dz_L = dU_dz_L[1]

        return uL[2] + (dt / self.eps) * (dQ_dz_L - uL[2])
    
    ########### Methods for branching ###########

    def P(self, u: np.ndarray, bound_side: str, dt: float):

        if bound_side == "right":
            xi = self.xiR(dt)
        elif bound_side == "left":
            xi = self.xiL(dt)

        return self.phi_np(u) - self.psi_np(u) * xi + (1 / 2) * self.blood.rho * (u[1] / u[0]) ** 2 

    def dP_dU(self, u: np.ndarray, bound_side: str, dt: float):
        """
        Calculate the derivative of the pressure with respect to the solution vector.
        """
        if bound_side == "right":
            xi = self.xiR(dt)
        elif bound_side == "left":
            xi = self.xiL(dt)

        dP_da = self.dphi_dA_np(u) + xi * self.psi_np(u) / (2 * u[0]) - self.blood.rho * (u[1] ** 2) / (u[0] ** 3)
        dP_dq = self.blood.rho * u[1] / (u[0] ** 2)

        return np.array([dP_da, dP_dq, -self.psi_np(u)])
        
    def f_branch(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        a, q = U
        return np.sign(q) * gamma * (q / a) ** 2 * np.sqrt(2 * (1 - np.cos(theta)))
    
    def df_dU(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        a, q = U

        dfdU_a = -2 * gamma * ((q ** 2) / (a ** 3)) * np.sqrt(2 * (1 - np.cos(theta)))
        dfdU_q = 2 * gamma * q / a ** 2 * np.sqrt(2 * (1 - np.cos(theta)))

        return np.array([dfdU_a, dfdU_q])
