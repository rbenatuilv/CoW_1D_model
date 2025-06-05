import numpy as np
import os
import json
from scipy.interpolate import interp1d
from typing import Literal


class VascularDataLoader:
    def __init__(
        self,
        mode: Literal["main", "test"] = "main",
        vessel_data_path: str = "",
        bif_data_path: str = "",
        inflows_dir_path: str = "",
    ):
        
        default_dir = "input_data"
        if mode not in ["main", "test"]:
            raise ValueError("Mode must be either 'main' or 'test'.")
        if mode == "test":
            default_dir = os.path.join(default_dir, "test")
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

    def load(self):
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



    