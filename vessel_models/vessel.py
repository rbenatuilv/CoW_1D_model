from mpi4py import MPI
import dolfinx.fem as fem  # type: ignore
from dolfinx import default_scalar_type  # type: ignore

import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import os 


class Blood:
    DYNAMIC_VISCOSITY = 0.045  # Poise = g/(cm·s)
    DENSITY = 1.050  # g/cm^3

    @property
    def mu(self):
        return self.DYNAMIC_VISCOSITY
    
    @property
    def rho(self):
        return self.DENSITY
    

class BloodVessel(Blood):

    GAMMA_PROFILE = 2
    POISSON_RATIO = 0.5

    def __init__(
        self, id: str, length: float, initial_area: float,
        beta_coeff: float,
        left_bound: Literal["inflow", "outflow", "branch"] = "inflow",
        right_bound: Literal["inflow", "outflow", "branch"] = "outflow"
    ):

        self.id = id
        self.L = length
        self.A0 = initial_area

        self.alpha = (self.GAMMA_PROFILE + 2) / (self.GAMMA_PROFILE + 1)  # Coriolis coefficient
        self.beta = beta_coeff
        self.Kr = 2 * (self.GAMMA_PROFILE + 2) * np.pi * self.mu / self.rho

        self.LB_type = left_bound
        self.RB_type = right_bound

        self.LB = np.array([self.A0, 0.0], dtype=default_scalar_type)
        self.RB = np.array([self.A0, 0.0], dtype=default_scalar_type)

        self.inflow = None  # To be set later if needed

        self.solutions = {
            "t": [],
            "A": [],
            "Q": [],
        }

        self.middlepoints = {
            "A": [],
            "Q": []
        }

        self.last_saved_time = 0.0
        self.last_sol = None # To be updated during the simulation

    def add_solution(self, t: float, save_array: bool):
        raise NotImplementedError("Method add_solution() must be implemented in subclasses.")

    def save_middlepoint_plot(self, quantity: Literal["A", "Q"], filename: str):
        """
        Save a plot of the middle point solution for the specified quantity (area or flux).
        Args:
            T (float): Total time for the simulation.
            quantity (str): The quantity to plot ("A" or "Q").
            filename (str): The filename to save the plot.
        """

        assert quantity in self.solutions, f"Invalid quantity: {quantity}. Available: {list(self.solutions.keys())}"

        data = self.middlepoints[quantity]
        if not data:
            raise ValueError(f"No solutions available for {quantity}.")

        unit = "cm^2" if quantity == "area" else "cm^3/s"

        middle_point_sol = np.array(data)
        x_values = self.solutions["t"]

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

        # Extract and convert one at a time, removing the original to free RAM
        # Note: np.savez accepts a filename directly, no need for `with open(...)`
        time_arr = np.array(self.solutions.pop("t"))
        area_arr = np.array(self.solutions.pop("A"))
        flux_arr = np.array(self.solutions.pop("Q"))

        np.savez(filename, area=area_arr, flux=flux_arr, time=time_arr)
        
        # Optional: force garbage collection here if saving multiple vessels
        import gc
        gc.collect()