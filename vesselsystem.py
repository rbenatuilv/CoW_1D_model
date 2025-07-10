from mpi4py import MPI
from typing import Literal

from elastic import ElasticVessel
from viscoelastic import ViscoelasticVessel


class VesselSystem:
    """
    Class that represents a system of blood vessels and bifurcations.
    It contains methods to initialize the vessels, set up the system, and manage inflows.
    """

    def __init__(self, vessels_data: dict, bifurcations_data: dict):
        
        self.vessels_data = vessels_data
        self.bifurcations = bifurcations_data
        self.vessels = {}
        self.inflows = {}

    def setup(self, h: float, dt: float, vessel_type: Literal["elastic", "viscoelastic"] = "elastic"):
        """
        Set up the system of vessels.
        This method initializes each vessel with the given mesh size `h` and time step `dt`.
        It creates the mesh, sets up the finite element space, boundary conditions,
        initial conditions, and the variational problem for each vessel.
        """

        for id, data in self.vessels_data.items():

            if vessel_type == "elastic":
                vessel = ElasticVessel(id=id, **data)
            elif vessel_type == "viscoelastic":
                vessel = ViscoelasticVessel(id=id, **data)
            
            self.vessels[id] = vessel

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
