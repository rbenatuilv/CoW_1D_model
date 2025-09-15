from mpi4py import MPI # type: ignore
from dolfinx import fem # type: ignore
import ufl
import numpy as np
from .vessel import BloodVessel


class ElasticVessel(BloodVessel):

    model_type = "Elastic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ###### UFL Forms and functions ######

    def B(self, u: fem.Function):
        return ufl.as_vector([
            0,
            self.Kr * (u[1] / u[0])
        ])
    
    def dB_dU(self, u: fem.Function):
        return ufl.as_tensor([
            [0, 0],
            [-self.Kr * (u[1] / (u[0]**2)), self.Kr / u[0]]
        ])
    
    def BLW(self, u: fem.Function, dt: float):
        return self.B(u) - (dt / 2) * ufl.dot(self.dB_dU(u), self.B(u)) # type: ignore
    
    def c_alpha_ufl(self, u: fem.Function):
        return ufl.sqrt(self.c2(u) + self.alpha * (self.alpha - 1) * (u[1] / u[0]) ** 2)

    def c2(self, u: fem.Function):
        return (self.beta / (2 * self.rho * self.A0)) * u[0] ** 0.5
    
    def C(self, U: fem.Function):
        return (self.beta / (3 * self.rho)) * (U[0] ** 1.5 / self.A0 - self.A0 ** 0.5)

    def F(self, U: fem.Function):
        return ufl.as_vector([
            U[1],
            self.alpha * U[1] ** 2 / U[0] + self.C(U)
        ])
    
    def FLW(self, U: fem.Function, dt: float):
        return self.F(U) - (dt / 2) * ufl.dot(self.H(U), self.B(U)) # type: ignore
    
    def H(self, U: fem.Function):
        return ufl.as_tensor([
            [0, 1],
            [self.c2(U) - self.alpha * (U[1] / U[0]) ** 2, 2 * self.alpha * (U[1] / U[0])]
        ])
    
    ###### Numpy Functions for BC and branching ######

    def B_np(self, U: np.ndarray):
        return np.array([
            0,
            self.Kr * (U[1] / U[0])
        ])
    
    def c2_np(self, U: np.ndarray):
        return (self.beta / (2 * self.rho * self.A0)) * np.sqrt(U[0])

    def c_alpha(self, U: np.ndarray):
        return np.sqrt(self.c2_np(U) + self.alpha * (self.alpha - 1) * (U[1] / U[0]) ** 2)
    
    def CC(self, U: np.ndarray, dU: np.ndarray, dt: float):
        return U - dt * self.H_np(U) @ dU - dt * self.B_np(U)
    
    def CCL(self, dt: float):
        uL = self.LB
        dU_dz_L = self.dU_dz(self.last_sol)[0] # type: ignore
        # print(f"dU_dz_L for vessel {self.id}: {dU_dz_L}")
        # input("Press Enter to continue...")

        return self.CC(uL, dU_dz_L, dt)
    
    def CCR(self, dt: float):
        uR = self.RB
        dU_dz_R = self.dU_dz(self.last_sol)[-1] # type: ignore
        # print(f"dU_dz_R for vessel {self.id}: {dU_dz_R}")
        # input("Press Enter to continue...")

        return self.CC(uR, dU_dz_R, dt)

    def f_branch(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        a, q = U
        return np.sign(q) * gamma * (q / a) ** 2 * np.sqrt(2 * (1 - np.cos(theta)))
    
    def df_dU(self, U: np.ndarray, theta: float, gamma: float = 2.0):
        a, q = U

        dfdU_a = -2 * gamma * ((q ** 2) / (a ** 3)) * np.sqrt(2 * (1 - np.cos(theta)))
        dfdU_q = 2 * gamma * q / a ** 2 * np.sqrt(2 * (1 - np.cos(theta)))

        return np.array([dfdU_a, dfdU_q])

    def H_np(self, U: np.ndarray):
        return np.array([
            [0, 1],
            [self.c2_np(U) - self.alpha * (U[1] / U[0]) ** 2, 2 * self.alpha * (U[1] / U[0])]
        ])
    
    def I1(self, U: np.ndarray):
        return np.array([
            self.c_alpha(U) - self.alpha * (U[1] / U[0]),
            1
        ])
    
    def I2(self, U: np.ndarray):
        return np.array([
            - self.c_alpha(U) - self.alpha * (U[1] / U[0]),
            1
        ])
    
    def P(self, U: np.ndarray):
        a, q = U
        return self.beta * ((np.sqrt(a) - self.A0 ** 0.5) / self.A0) + 0.5 * self.rho * (q / a) ** 2
    
    def dP_dU(self, U: np.ndarray):
        a, q = U

        dP_da = self.beta / (2 * self.A0 * np.sqrt(a)) - self.rho * (q ** 2) / (a ** 3)
        dP_dq = self.rho * q / a ** 2

        return np.array([dP_da, dP_dq])

    def dU_dz(self, u: np.ndarray):
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
