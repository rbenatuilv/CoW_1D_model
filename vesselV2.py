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

from abc import ABC, abstractmethod


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


class Vessel(ABC):
    """
    Abstract base class for a vessel.
    """

    GAMMA_PROFILE = 2
    blood = Blood()

    def __init__(
        self, id: str, longitude: float, initial_area: float,
        beta_coeff: float, 
        left_bound: Literal["branch", "inlet", "outlet"] = "inlet",
        right_bound: Literal["branch", "inlet", "outlet"] = "outlet"
    ):
        """
        Initialize the vessel with its properties.

        Parameters:
        id (str): Unique identifier for the vessel.
        length (float): Length of the vessel in cm.
        initial_area (float): Initial cross-sectional area of the vessel in cm^2.
        beta_coeff (float): Vessel wall stiffness coefficient in g/s^2.
        left_bound (Literal["branch", "inlet", "outlet"]): Type of left boundary condition.
        right_bound (Literal["branch", "inlet", "outlet"]): Type of right boundary condition.
        """

        # Basic properties
        self.id = id
        self.L = longitude  # Length of the vessel in cm
        self.A0 = initial_area

        self.LB_type = left_bound
        self.RB_type = right_bound

        self.alpha = (self.GAMMA_PROFILE + 2) / (self.GAMMA_PROFILE + 1)
        self.beta = beta_coeff
        self.Kr = 2 * (self.GAMMA_PROFILE + 2) * np.pi * self.blood.mu / self.blood.rho

        # Mesh and function space
        self.mesh = None
        self.V = None
        self.n_dofs = 0

        # Boundary conditions
        self.LB = None # Left boundary condition value
        self.RB = None # Right boundary condition value

        self.dofs_L = None  # Degrees of freedom for left boundary
        self.dofs_R = None  # Degrees of freedom for right boundary
        self.bcs = []  # List of boundary conditions

        # Variational problem
        self.bilinear = None
        self.linear = None
        self.A = None
        self.rhs = None
        self.solver = None

        # Initial conditions
        self.u_n = None  # Function for initial conditions

        # Solution storage
        self.u = None  # Current solution function

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

    def create_mesh(self, mesh_size: float = 0.1):
        """
        Create a 1D mesh for the vessel.

        Parameters:
        mesh_size (float): Size of the mesh elements in cm.
        """

        N = int(self.L / mesh_size)
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, N, (0, self.L))

    @abstractmethod
    def create_fem_space(self, element_type: str = "Lagrange"):
        """
        Abstract method to create a finite element function space.
        Parameters:
        element_type (str): Type of finite element to use (e.g., "Lagrange", "CG").
        """

        raise NotImplementedError("Subclasses must implement this method.")


    def set_boundary_dofs(self):
        """Set the degrees of freedom for the left and right boundaries."""

        if self.V is None:
            raise ValueError("Function space not set. Call create_fem_space() first.")

        self.dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        self.dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.L))

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

    @abstractmethod
    def add_solution(self, u: fem.Function, save_all: bool = False):
        """
        Abstract method to add a solution to the vessel.
        Parameters:
        u (fem.Function): The solution function to be added.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def set_variational_problem(self, dt: float):
        """
        Abstract method to set the variational problem for the vessel.
        Parameters:
        dt (float): Time step size for the simulation.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def initial_setup(self, mesh_size: float, dt: float):
        """
        Perform the initial setup for the vessel.
        This includes creating the mesh, function space, boundary conditions, and initial conditions.

        Parameters:
        mesh_size (float): Size of the mesh elements in cm.
        dt (float): Time step size for the simulation.
        """

        self.create_mesh(mesh_size)
        self.create_fem_space()
        self.set_boundary_dofs()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_variational_problem(dt)

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


    @abstractmethod
    def dU_dz(self):
        """
        Abstract method to compute the derivative of the solution with respect to z.
        This is typically used in the variational problem.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def I1(self, u: np.ndarray):
        """
        Abstract method to compute the first invariant of the solution.
        This is typically used in the variational problem.
        Args:
            u (np.ndarray): The solution vector.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def I2(self, u: np.ndarray):
        """
        Abstract method to compute the second invariant of the solution.
        This is typically used in the variational problem.
        Args:
            u (np.ndarray): The solution vector.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def CC(self, u: np.ndarray, du: np.ndarray, dt: float):
        """
        Abstract method to compute the compatibility condition.
        This is typically used in the variational problem.
        Args:
            u (np.ndarray): The solution vector.
            du (np.ndarray): The derivative of the solution vector.
            dt (float): Time step size.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def CCR(self, dt: float):
        """
        Abstract method to compute the compatibility condition for the right boundary.
        This is typically used in the variational problem.
        Args:
            dt (float): Time step size.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def CCL(self, dt: float):
        """
        Abstract method to compute the compatibility condition for the left boundary.
        This is typically used in the variational problem.
        Args:
            dt (float): Time step size.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def P(self, u: np.ndarray):
        """
        Abstract method to compute the pressure in the vessel.
        Args:
            u (np.ndarray): The solution vector.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def dP_dU(self, u: np.ndarray):
        """
        Abstract method to compute the derivative of the pressure with respect to the solution.
        Args:
            u (np.ndarray): The solution vector.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def f_branch(self, u: np.ndarray, theta: float, gamma: float = 2.0):
        """
        Abstract method to compute the branching function.
        Args:
            u (np.ndarray): The solution vector.
            theta (float): Angle of branching.
            gamma (float): Branching coefficient.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def df_dU(self, u: np.ndarray, theta: float, gamma: float = 2.0):
        """
        Abstract method to compute the derivative of the branching function with respect to the solution.
        Args:
            u (np.ndarray): The solution vector.
            theta (float): Angle of branching.
            gamma (float): Branching coefficient.
        """

        raise NotImplementedError("Subclasses must implement this method.")
    
