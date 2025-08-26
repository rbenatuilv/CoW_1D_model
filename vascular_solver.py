from mpi4py import MPI
from tqdm import tqdm
from dolfinx import default_scalar_type  # type: ignore
import numpy as np
import os
from typing import Literal

from vascular_net import VascularNetwork
from boundary_solver.elasticBC import ElasticBCSolver


class VascularSolver:
    def __init__(self, network: VascularNetwork, model: str = "Elastic", method: str = "CG"):
        self.network = network
        self.model = model
        self.method = method

        self.bc_solver = ElasticBCSolver()

    def setup(self, h: float, dt: float):
        self.network.setup_network(h=h, dt=dt, model=self.model, method=self.method)
        self.dt = dt
        self.h = h

    def _broadcast_updated_BCs(self):
        """
        After rank 0 has run solve_inflow_BC, solve_outflow_BC and solve_branches(),
        each vessel’s `vessel.LB` and `vessel.RB` live only on rank 0. Here we pack
        them into length–2 arrays and Bcast them so that every rank ends up with the
        same boundary arrays.
        """
        comm = MPI.COMM_WORLD
        rank = comm.rank

        for vessel_id, vessel in self.network.vessels.items():
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

    def solve(self, t_end: float):
        comm = MPI.COMM_WORLD
        rank = comm.rank

        time_steps = int(t_end / self.dt)
         
        if rank == 0:
            iterator = tqdm(range(time_steps), desc="Solving Vascular Network", unit="step")
        else:
            iterator = range(time_steps)

        t = 0.0
        for _ in iterator:
            t += self.dt

            for vessel in self.network.vessels.values():
                vessel.solve()
                vessel.add_solution(t)

            comm.Barrier()

            if rank == 0:
                for vessel in self.network.vessels.values():
                    self.bc_solver.solve_inflow_BC(vessel, t, self.dt)
                    self.bc_solver.solve_outflow_BC(vessel, self.dt)

                for bid in self.network.bifurcations:
                    bif = self.network.bifurcations[bid]

                    vessels = [self.network.vessels[v_id] for v_id in bif["branches"]]
                    self.bc_solver.solve_branch(vessels, bif, dt=self.dt)

            self._broadcast_updated_BCs()

            for vessel in self.network.vessels.values():
                vessel.update_BCs()

            comm.Barrier()

    def create_results_directory(self, mode: Literal["main", "test"] = "main"):
        """
        Create the results directory structure for storing plots and data.
        Args:
            mode (str): The mode of operation, either "main" or "test".
        """

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
        """
        Plot the solutions of the vascular system for each vessel.
        Args:
            T (float): Total time to plot the solutions for.
            mode (str): The mode of operation, either "main" or "test".
        """
        comm = MPI.COMM_WORLD
        rank = comm.rank

        if rank != 0:
            return

        self.create_results_directory(mode)

        path = os.path.join("results", mode, "plots")
        area_path = os.path.join(path, "area")
        flux_path = os.path.join(path, "flux")

        for vessel in tqdm(self.network.vessels.values(), desc="Plotting Solutions", unit="vessel"):
            vessel.save_middlepoint_plot("A", os.path.join(area_path, f"vessel_{vessel.id}.png"))
            vessel.save_middlepoint_plot("Q", os.path.join(flux_path, f"vessel_{vessel.id}.png"))

        print(f"Plots saved to {path}")

    def save_solutions(self, mode: Literal["main", "test"] = "main"):
        """
        Save the solutions of the vascular system for each vessel.
        Args:
            mode (str): The mode of operation, either "main" or "test".
        """
        comm = MPI.COMM_WORLD
        rank = comm.rank

        if rank != 0:
            return

        if not self.network:
            raise RuntimeError("System not set. Please set the system before saving solutions.")
        
        self.create_results_directory(mode)

        path = os.path.join("results", mode, "data")

        for vessel in tqdm(self.network.vessels.values(), desc="Saving Solutions", unit="vessel"):
            vessel.save_solution(path)

        print(f"Solutions saved to {path}")