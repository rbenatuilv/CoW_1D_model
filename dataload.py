import numpy as np
import os
import json
from scipy.interpolate import interp1d
from typing import Literal


class VascularDataLoader:
    """
    A class to load vascular data for the simulation.
    It loads vessel data, bifurcation data, and inflow data from specified paths.
    The paths can be specified or default paths will be used based on the mode.
    """

    def __init__(
        self,
        mode: Literal["main", "test", "test_single", "sin", "sin_single"] = "main",
        vessel_data_path: str = "",
        bif_data_path: str = "",
        inflows_dir_path: str = "",
    ):
        
        default_dir = "input_data"
        if mode not in ["main", "test", "test_single", "sin", "sin_single"]:
            raise ValueError("Mode must be either 'main' or 'test'.")
        if mode == "test":
            default_dir = os.path.join(default_dir, "test")
        elif mode == "test_single":
            default_dir = os.path.join(default_dir, "test_single")
        elif mode == "sin":
            default_dir = os.path.join(default_dir, "sin")
        elif mode == "sin_single":
            default_dir = os.path.join(default_dir, "sin_single")
        else:
            default_dir = os.path.join(default_dir, "main")


        if not vessel_data_path:
            vessel_data_path = os.path.join(default_dir, "vessel_data.json")

        if not bif_data_path:
            bif_data_path = os.path.join(default_dir, "bif_data.json")

        if not inflows_dir_path:
            inflows_dir_path = os.path.join(default_dir, "inflows")

        self.vessel_data_path = vessel_data_path
        self.bif_data_path = bif_data_path
        self.inflows_dir_path = inflows_dir_path

    def load(self) -> tuple[dict, dict, dict]:
        """
        Load the vessel data, bifurcation data, and inflow data from the specified paths.
        Returns:
            tuple: A tuple containing:
                - vessels_data (dict): Vessel data loaded from JSON.
                - bif_data (dict): Bifurcation data loaded from JSON.
                - inflows (dict): A dictionary mapping vessel IDs to their inflow functions.
        """

        with open(self.vessel_data_path, 'r') as f:
            vessels_data = json.load(f)

        with open(self.bif_data_path, 'r') as f:
            bif_data = json.load(f)

        inflows = {}
        for filename in os.listdir(self.inflows_dir_path):
            if not filename.endswith('.csv'):
                continue

            vessel_id = os.path.splitext(filename)[0]
            
            file_path = os.path.join(self.inflows_dir_path, filename)
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            time = data[:, 0] - min(data[:, 0])
            vel = data[:, 1]
            velocity = interp1d(time, vel, bounds_error=False, fill_value="extrapolate")
            period = np.max(time)
            inflows[vessel_id] = lambda t, v=velocity, p=period: v(np.mod(t, p))

        return vessels_data, bif_data, inflows
