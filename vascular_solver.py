from mpi4py import MPI
from tqdm import tqdm
from contextlib import nullcontext
from dolfinx import default_scalar_type  # type: ignore
import numpy as np
import os
from typing import Literal

from dolfinx import fem, mesh # type: ignore
import ufl

from vascular_net import VascularNetwork
from boundary_solver.elasticBC import ElasticBCSolver


class VascularSolver:
    def __init__(self, network: VascularNetwork, model: str = "Elastic", method: str = "CG", num_flux: str = "HLL", name: str | None = None):
        self.network = network
        self.model = model
        self.method = method
        self.num_flux = num_flux
        self.name = name

        self.bc_solver = ElasticBCSolver()

    def setup(self, h: float, dt: float):
        self.network.setup_network(h=h, dt=dt, model=self.model, method=self.method, num_flux=self.num_flux)
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

        for vessel in self.network.vessels.values():
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

    def max_eigval(self, u: fem.Function):
        eigval1 = self.network.alpha * (u[1] / u[0]) + self.network.c_alpha_ufl(u)
        eigval2 = self.network.alpha * (u[1] / u[0]) - self.network.c_alpha_ufl(u)

        # return 1000

        return ufl.max_value(
            ufl.conditional(ufl.ge(eigval1, 0), eigval1, -eigval1),
            ufl.conditional(ufl.ge(eigval2, 0), eigval2, -eigval2)
        )
    

    
    def calculate_dt(self, enforce: Literal["CG", "DG", "min"]="min"):
        if enforce == "CG":
            C = np.sqrt(3)/3
        elif enforce == "DG":
            C = 0.5
        else:
            C = np.minimum(np.sqrt(3)/3, 0.5)
        max_lambda = -1*np.inf
        ### IT works only for DG
        for vessel in self.network.vessels.values():
            max_lambda = np.maximum(vessel.max_lambda_u_n(), max_lambda)
        return C*self.h/max_lambda

    def solve(self, t_end: float):
        comm = MPI.COMM_WORLD
        rank = comm.rank
         
        progress_context = tqdm(total=t_end, desc="Solving Vascular Network", unit="s") if rank == 0 else nullcontext()
        t = 0.0
        with progress_context as pbar:
            while t<t_end:
                dt = self.calculate_dt(enforce=self.method)
                self.dt = min(dt, t_end-t)
                t += self.dt

                for vessel in self.network.vessels.values():
                    vessel.dt.value = self.dt
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
                if rank==0:
                    pbar.update(dt)

    def create_results_directory(self, T: float, mode: Literal["main", "test", "test_single"] = "main", 
        method: Literal["CG", "DG"] = "CG", num_flux: Literal["LxF", "HLL"] = "LxF"):
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
        
        path = os.path.join(path, method)
        if not os.path.exists(path):
            os.mkdir(path)
        
        if method == "DG":
            path = os.path.join(path, num_flux)
            if not os.path.exists(path):
                os.mkdir(path)

        name = f"{self.name}_h{self.h}_dt{self.dt}_T{T}" if self.name is not None else f"{self.model}_{self.method}_h{self.h}_dt{self.dt}_T{T}"

        path = os.path.join(path, name)
        if not os.path.exists(path):
            os.mkdir(path)
        

        self.results_path = path

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

    def plot_solutions(self, T: float, mode: Literal["main", "test", "test_single"] = "main", 
        method: Literal["CG", "DG"] = "CG", num_flux: Literal["LxF", "HLL"] = "LxF"):
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

        self.create_results_directory(T, mode, method, num_flux)

        path = os.path.join(self.results_path, "plots")

        area_path = os.path.join(path, "area")
        flux_path = os.path.join(path, "flux")

        for vessel in tqdm(self.network.vessels.values(), desc="Plotting Solutions", unit="vessel"):
            vessel.save_middlepoint_plot("A", os.path.join(area_path, f"vessel_{vessel.id}.png"))
            vessel.save_middlepoint_plot("Q", os.path.join(flux_path, f"vessel_{vessel.id}.png"))

        print(f"Plots saved to {self.results_path}")

    def save_solutions(self, T: float, mode: Literal["main", "test", "test_single"] = "main", 
        method: Literal["CG", "DG"] = "CG", num_flux: Literal["LxF", "HLL"] = "LxF"):
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
        
        self.create_results_directory(T, mode, method, num_flux)

        path = os.path.join(self.results_path, "data")

        for vessel in tqdm(self.network.vessels.values(), desc="Saving Solutions", unit="vessel"):
            vessel.save_solution(path)

        print(f"Solutions saved to {path}")