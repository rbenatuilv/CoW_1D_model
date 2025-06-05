import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import petsc
from petsc4py import PETSc
from vessel import BloodVessel, VesselSystem
from tqdm import tqdm
import os
from mpi4py import MPI
from profiler.profiler import profile_this
from typing import Literal


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class VascularSolver:
    def __init__(self, h: float, dt: float):
        self.h = h
        self.dt = dt
        self.system = None

    def set_system(self, system: VesselSystem):
        self.system = system
        self.system.setup(h=self.h, dt=self.dt)

    @profile_this
    def solve_interior(self, vessel: BloodVessel, store_solution: bool = True):
        
        with vessel.rhs.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(vessel.rhs, vessel.linear)

        petsc.apply_lifting(vessel.rhs, [vessel.bilinear], [vessel.bcs])
        vessel.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(vessel.rhs, vessel.bcs)

        vessel.solver.solve(vessel.rhs, vessel.u.x.petsc_vec)
        vessel.u.x.scatter_forward()
        vessel.u_n.x.array[:] = vessel.u.x.array

        if np.isnan(vessel.u.x.array).any():
            raise RuntimeError(f"NaN values found in the solution of vessel {vessel.id}. "
                               "Check the boundary conditions or the initial conditions.")

        vessel.add_solution(vessel.u, save_all=store_solution)

    @profile_this
    def solve_inflow_BC(self, vessel: BloodVessel, t: float):
        if "inflow" not in [vessel.LB_type, vessel.RB_type]:
            return
        
        vessel_A = vessel.last_solution["area"]
        vessel_Q = vessel.last_solution["flux"]

        sol = np.column_stack((vessel_A, vessel_Q)).reshape((len(vessel_A), 2))

        inflow_func = self.system.inflows[vessel.id]

        if vessel.LB_type == "inflow":

            uL = sol[0]
            du_dz_L = vessel.dU_dz()[0]

            q = inflow_func(t) * vessel.A0
            i2 = vessel.I2(uL)
            A = (i2 @ vessel.CC(uL, du_dz_L, self.dt) - i2[1] * q) / (i2[0] + 1e-12)

            vessel.LB = np.array([A, q], dtype=default_scalar_type)

        else:
            uR = sol[-1]
            du_dz_R = vessel.dU_dz()[-1]
            
            q = -inflow_func(t) * vessel.A0
            i1 = vessel.I1(uR)
            A = (i1 @ vessel.CC(uR, du_dz_R, self.dt) - i1[1] * q) / (i1[0] + 1e-12)

            vessel.RB = np.array([A, q], dtype=default_scalar_type)

    @profile_this
    def solve_outflow_BC(self, vessel: BloodVessel):
        if "outflow" not in [vessel.LB_type, vessel.RB_type]:
            return
        
        vessel_A = vessel.last_solution["area"]
        vessel_Q = vessel.last_solution["flux"]

        sol = np.column_stack((vessel_A, vessel_Q)).reshape((len(vessel_A), 2))

        if vessel.LB_type == "outflow":
            uL = sol[0]
            du_dz_L = vessel.dU_dz()[0]
            vessel.LB = vessel.W0(uL) @ vessel.Y0(uL, du_dz_L, self.dt)

        if vessel.RB_type == "outflow":
            uR = sol[-1]
            du_dz_R = vessel.dU_dz()[-1]
            vessel.RB = vessel.WL(uR) @ vessel.YL(uR, du_dz_R, self.dt)


    def create_newton(self, bifurcation: dict, gamma: float = 2.0):
        v1, v2, v3 = [self.system.vessels[vessel_id] 
                      for vessel_id in bifurcation["branches"]]

        th2, th3 = bifurcation["angles"][1:]

        i1 = v1.I1(v1.RB) if bifurcation["positions"][0] == "right" else v1.I2(v1.LB)
        i2 = v2.I1(v2.RB) if bifurcation["positions"][1] == "right" else v2.I2(v2.LB)
        i3 = v3.I1(v3.RB) if bifurcation["positions"][2] == "right" else v3.I2(v3.LB)

        CC1 = v1.CCR(self.dt) if bifurcation["positions"][0] == "right" else v1.CCL(self.dt)
        CC2 = v2.CCR(self.dt) if bifurcation["positions"][1] == "right" else v2.CCL(self.dt)
        CC3 = v3.CCR(self.dt) if bifurcation["positions"][2] == "right" else v3.CCL(self.dt)

        positions = np.array([
            1 if pos == "right" else -1
            for pos in bifurcation["positions"]
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

        u1 = v1.LB if bifurcation["positions"][0] == "left" else v1.RB
        u2 = v2.LB if bifurcation["positions"][1] == "left" else v2.RB
        u3 = v3.LB if bifurcation["positions"][2] == "left" else v3.RB

        U0 = np.array([u1[0], u1[1], u2[0], u2[1], u3[0], u3[1]])

        return N, J, U0

    
    @staticmethod
    def branch_conv_criterion(u0, u_prev, u_curr, tol):
        """
        u0     : initial state (shape [6])
        u_prev : previous state (shape [6])
        u_curr : current state   (shape [6])
        tol    : tolerance for convergence criterion
        """
        # Separate area and flux components
        A0, Q0 = u0[::2], u0[1::2]
        A_prev, Q_prev = u_prev[::2], u_prev[1::2]
        A_curr, Q_curr = u_curr[::2], u_curr[1::2]

        # Avoid division by zero
        eps = 1e-12
        relative_A = np.abs(A_curr - A_prev) / (np.abs(A0) + eps)
        relative_Q = np.abs(Q_curr - Q_prev) / (np.abs(Q0) + eps)

        total_error = np.sum(relative_A + relative_Q)
        return total_error < tol

    @profile_this
    def solve_branches(self, tol: float = 1e-5, max_iter: int = 100, gamma: float = 2.0):
        for bid in self.system.bifurcations:
            bif = self.system.bifurcations[bid]
            
            N, J, U0 = self.create_newton(bif, gamma=gamma)

            converged = False
            u_prev = U0.copy()
            for i in range(max_iter):
                u_curr = u_prev + np.linalg.solve(J(u_prev), -N(u_prev))

                if self.branch_conv_criterion(U0, u_prev, u_curr, tol):
                    converged = True
                    break

                u_prev = u_curr

            if not converged:
                if np.isnan(u_curr).any():
                    raise RuntimeError("Convergence failed due to NaN values in the solution.")
                
            
            vessels = [self.system.vessels[vessel_id]
                            for vessel_id in bif["branches"]]
            
            for i, vessel in enumerate(vessels):
                if bif["positions"][i] == "left":
                    vessel.LB = np.array(u_curr[i*2:i*2+2], dtype=default_scalar_type)
                else:
                    vessel.RB = np.array(u_curr[i*2:i*2+2], dtype=default_scalar_type)

    def _broadcast_updated_BCs(self):
        """
        After rank 0 has run solve_inflow_BC, solve_outflow_BC and solve_branches(),
        each vessel’s `vessel.LB` and `vessel.RB` live only on rank 0. Here we pack
        them into length–2 arrays and Bcast them so that every rank ends up with the
        same boundary arrays.
        """
        for vessel_id, vessel in self.system.vessels.items():
            # Prepare two 2‐entry buffers for LB and RB
            lb_buf = np.zeros(2, dtype=default_scalar_type)
            rb_buf = np.zeros(2, dtype=default_scalar_type)

            if rank == 0:
                # On rank 0, load the “true” BCs into the buffers.
                if hasattr(vessel, "LB"):
                    lb_buf[:] = vessel.LB
                if hasattr(vessel, "RB"):
                    rb_buf[:] = vessel.RB

            # Broadcast from rank 0 → all ranks:
            comm.Bcast(lb_buf, root=0)
            comm.Bcast(rb_buf, root=0)

            # Overwrite each vessel’s LB/RB on every rank:
            vessel.LB = lb_buf.copy()
            vessel.RB = rb_buf.copy()

        comm.Barrier()

    def solve(self, T: float, gamma_bif: float = 2.0):
        """
        Time-march for t from 0 → T, in steps of self.dt. Only `solve_interior`
        is MPI-parallel; all boundary logic happens on rank 0 and then is Bcast.
        """
        n_steps = int(T / self.dt)

        # Only rank 0 gets a tqdm bar
        if rank == 0:
            iterator = tqdm(range(n_steps), desc="Solving Vascular System", unit="step")
        else:
            iterator = range(n_steps)

        t = 0.0
        i = 0

        save_on = n_steps // int(T / 1e-3) # Save every 1 ms

        for _ in iterator:
            i += 1
            t += self.dt

            store_solution = (i % save_on == 0)

            # 1) EVERY RANK does each vessel's interior solve (parallel PETSc calls)
            for vessel in self.system.vessels.values():
                self.solve_interior(vessel, store_solution=store_solution)
                # At this point, vessel.add_solution(u) has stored a global array
                # in vessel.solutions["area"] and ["flux"] on every rank.

            # 2) Sync so that all ranks finish interior solves before BC logic:
            comm.Barrier()

            # 3) On rank 0 only: compute inflow & outflow BC for each vessel
            if rank == 0:
                for vessel in self.system.vessels.values():
                    self.solve_inflow_BC(vessel, t)
                    self.solve_outflow_BC(vessel)

                # 4) Then on rank 0 run the bifurcation coupling
                self.solve_branches(gamma=gamma_bif)

            # 5) Broadcast the newly computed LB/RB from rank 0 → all ranks
            self._broadcast_updated_BCs()

            # 6) EVERY RANK now applies the BCs to its local DOLFINx objects:
            for vessel in self.system.vessels.values():
                vessel.set_boundary_conditions()

            comm.Barrier()  # Ensure all ranks are synchronized before the next step

        # Optional final sync:
        comm.Barrier()

    def create_results_directory(self, mode: Literal["main", "test"] = "main"):
        path = os.path.join("results")
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, mode)
        if not os.path.exists(path):
            os.mkdir(path)

        plots_path = os.path.join(path, "plots")
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)

        area_path = os.path.join(plots_path, "area")
        if not os.path.exists(area_path):
            os.mkdir(area_path)
        flux_path = os.path.join(plots_path, "flux")
        if not os.path.exists(flux_path):
            os.mkdir(flux_path)

        data_path = os.path.join(path, "data")
        if not os.path.exists(data_path):
            os.mkdir(data_path)

    def plot_solutions(self, T: float, mode: Literal["main", "test"] = "main"):
        if rank != 0:
            return

        if not self.system:
            raise RuntimeError("System not set. Please set the system before plotting solutions.")
        
        self.create_results_directory(mode)

        path = os.path.join("results", mode, "plots")
        area_path = os.path.join(path, "area")
        flux_path = os.path.join(path, "flux")

        for vessel in tqdm(self.system.vessels.values(), desc="Plotting Solutions", unit="vessel"):
            vessel.save_middlepoint_plot(T, "area", os.path.join(area_path, f"vessel_{vessel.id}.png"))
            vessel.save_middlepoint_plot(T, "flux", os.path.join(flux_path, f"vessel_{vessel.id}.png"))

        print(f"Plots saved to {path}")

    def save_solutions(self, mode: Literal["main", "test"] = "main"):
        if rank != 0:
            return

        if not self.system:
            raise RuntimeError("System not set. Please set the system before saving solutions.")
        
        self.create_results_directory(mode)

        path = os.path.join("results", mode, "data")

        for vessel in tqdm(self.system.vessels.values(), desc="Saving Solutions", unit="vessel"):
            vessel.save_solution(path)

        print(f"Solutions saved to {path}")


if __name__ == "__main__":
    pass
    