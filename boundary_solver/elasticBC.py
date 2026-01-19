from mpi4py import MPI # type: ignore
from dolfinx import default_scalar_type # type: ignore
import numpy as np

from vessel_models.elastic_vessel import ElasticVessel


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class ElasticBCSolver:

    def solve_inflow_BC(self, vessel: ElasticVessel, t: float, dt: float):
        if "inflow" not in [vessel.LB_type, vessel.RB_type]:
            return

        if vessel.inflow is None:
            raise ValueError(f"Inflow profile not set for the vessel {vessel.id}")

        sol = vessel.last_sol
        if sol is None:
            raise ValueError(f"No previous solution found for the vessel {vessel.id}.")
        
        du_dz = vessel.dU_dz(sol)

        if vessel.LB_type == "inflow":
            uL = sol[0]
            du_dz_L = du_dz[0]

            q = vessel.inflow(t) * vessel.A0
            i2 = vessel.I2(uL)
            A = (i2 @ vessel.CC(uL, du_dz_L, dt) - i2[1] * q) / (i2[0])

            vessel.LB = np.array([A, q], dtype=default_scalar_type)

        if vessel.RB_type == "inflow":
            uR = sol[-1]
            du_dz_R = du_dz[-1]

            q = -vessel.inflow(t) * vessel.A0
            i1 = vessel.I1(uR)
            A = (i1 @ vessel.CC(uR, du_dz_R, dt) - i1[1] * q) / (i1[0])

            vessel.RB = np.array([A, q], dtype=default_scalar_type)

    def solve_outflow_BC(self, vessel: ElasticVessel, dt: float):
        if "outflow" not in [vessel.LB_type, vessel.RB_type]:
            return 
    
        sol = vessel.last_sol
        if sol is None:
            raise ValueError(f"No previous solution found for the vessel {vessel.id}.")
        
        du_dz = vessel.dU_dz(sol)

        if vessel.LB_type == "outflow":
            uL = sol[0]
            du_dz_L = du_dz[0]

            A = vessel.A0 
            i2 = vessel.I2(uL)
            q = (i2 @ vessel.CC(uL, du_dz_L, dt) - i2[0] * A) / (i2[1])

            vessel.LB = np.array([A, q], dtype=default_scalar_type)

        if vessel.RB_type == "outflow":
            uR = sol[-1]
            du_dz_R = du_dz[-1]

            A = vessel.A0
            i1 = vessel.I1(uR)
            q = (i1 @ vessel.CC(uR, du_dz_R, dt) - i1[0] * A) / (i1[1])

            vessel.RB = np.array([A, q], dtype=default_scalar_type)

    def create_newton(
        self, vessels: list[ElasticVessel], branch: dict, 
        dt: float, gamma: float = 2.0
    ):

        v1, v2, v3 = vessels
        th2, th3 = branch["angles"][1:]

        i1 = v1.I1(v1.RB) if branch["positions"][0] == "right" else v1.I2(v1.LB)
        i2 = v2.I1(v2.RB) if branch["positions"][1] == "right" else v2.I2(v2.LB)
        i3 = v3.I1(v3.RB) if branch["positions"][2] == "right" else v3.I2(v3.LB)

        CC1 = v1.CCR(dt) if branch["positions"][0] == "right" else v1.CCL(dt)
        CC2 = v2.CCR(dt) if branch["positions"][1] == "right" else v2.CCL(dt)
        CC3 = v3.CCR(dt) if branch["positions"][2] == "right" else v3.CCL(dt)

        positions = np.array([
            1 if pos == "right" else -1
            for pos in branch["positions"]
        ])

        def N(U):
            u1 = U[:2]
            u2 = U[2:4]
            u3 = U[4:6]

            return np.array([
                np.dot(
                    np.array([u1[1], u2[1], u3[1]]),
                    positions
                ),

                v1.P(u1) - v2.P(u2) - v2.f_branch(u2, th2, gamma),
                v1.P(u1) - v3.P(u3) - v3.f_branch(u3, th3, gamma),
                np.dot(i1, u1) - np.dot(i1, CC1),
                np.dot(i2, u2) - np.dot(i2, CC2),
                np.dot(i3, u3) - np.dot(i3, CC3)
            ])
        
        def J(U):
            j = np.zeros((6, 6))

            u1 = U[:2]
            u2 = U[2:4]
            u3 = U[4:6]

            j[0, 1] = positions[0]
            j[0, 3] = positions[1]
            j[0, 5] = positions[2]

            dP1 = v1.dP_dU(u1)
            dP2 = v2.dP_dU(u2)
            dP3 = v3.dP_dU(u3)

            df2 = v2.df_dU(u2, th2, gamma)
            df3 = v3.df_dU(u3, th3, gamma)

            j[1, 0] = dP1[0]
            j[1, 1] = dP1[1]
            j[1, 2] = -dP2[0] - df2[0]
            j[1, 3] = -dP2[1] - df2[1]

            j[2, 0] = dP1[0]
            j[2, 1] = dP1[1]
            j[2, 4] = -dP3[0] - df3[0]
            j[2, 5] = -dP3[1] - df3[1]

            j[3, 0] = i1[0]
            j[3, 1] = i1[1]

            j[4, 2] = i2[0]
            j[4, 3] = i2[1]

            j[5, 4] = i3[0]
            j[5, 5] = i3[1]

            return j

        u1 = v1.LB if branch["positions"][0] == "left" else v1.RB
        u2 = v2.LB if branch["positions"][1] == "left" else v2.RB
        u3 = v3.LB if branch["positions"][2] == "left" else v3.RB

        U0 = np.array([u1[0], u1[1], u2[0], u2[1], u3[0], u3[1]])

        return N, J, U0
    
    @staticmethod
    def branch_conv_criterion(u0, u_prev, u_curr, tol):
        """
        Check convergence criterion for the branch solution.
        This function checks if the relative change in area and flux
        between the previous and current states is below a specified tolerance.
        Args:
            u0     : initial state (shape [6])
            u_prev : previous state (shape [6])
            u_curr : current state   (shape [6])
            tol    : tolerance for convergence criterion
        """
        # Compute relative change normalized by current values
        eps = 1e-12
        relative_change = np.abs(u_curr - u_prev) / (np.abs(u_curr) + eps)
        
        # Check if maximum relative change is below tolerance
        return np.max(relative_change) < tol

    def solve_branch(
        self, vessels: list[ElasticVessel], branch: dict, dt: float, 
        tol: float = 1e-8, max_iter: int = 1000, omega: float = 1.0
    ):
        N, J, U0 = self.create_newton(vessels, branch, dt)

        converged = False
        u_prev = U0.copy()
        u_curr = None  # Initialize to handle edge cases

        for i in range(max_iter):
            residual = N(u_prev)
            residual_norm = np.linalg.norm(residual)
            
            # Check convergence based on residual norm (primary criterion)
            if residual_norm < tol:
                converged = True
                u_curr = u_prev
                break
            
            delta_u = np.linalg.solve(J(u_prev), -residual)
            u_curr = u_prev + omega * delta_u
            
            # Check for NaN or Inf
            if np.isnan(u_curr).any() or np.isinf(u_curr).any():
                raise RuntimeError("Branch Newton solver diverged (NaN or Inf encountered)")

            # Check convergence based on relative change (secondary criterion)
            if self.branch_conv_criterion(U0, u_prev, u_curr, tol):
                converged = True
                break

            u_prev = u_curr

        if not converged:
            # Handle case where no iterations were performed
            if u_curr is None:
                raise RuntimeError(f"Branch Newton solver failed: max_iter={max_iter} is invalid")
            
            raise RuntimeError(f"Branch Newton solver failed to converge within {max_iter} iterations. Final residual: {np.linalg.norm(N(u_curr))}")

        for i, vessel in enumerate(vessels):
            if branch["positions"][i] == "left":
                vessel.LB = np.array(u_curr[i*2:i*2+2], dtype=default_scalar_type) # type: ignore
                
            else:
                vessel.RB = np.array(u_curr[i*2:i*2+2], dtype=default_scalar_type) # type: ignore 
                