from elasticCG import ElasticCGVessel


class VascularNetwork:
    def __init__(self, vessels_data: dict, bifurcations_data: dict, inflows: dict):
        self.vessels_data = vessels_data
        self.bifurcations = bifurcations_data
        self.inflows = inflows

        self.vessels = {}

    def setup_network(self, h: float, dt: float, model: str = "Elastic", method: str = "CG"):
        for id, params in self.vessels_data.items():
            if model == "Elastic":
                if method == "CG":
                    vessel = ElasticCGVessel(id=id, **params)
                else:
                    raise ValueError(f"Method {method} not recognized for model {model}.")
            else:
                raise ValueError(f"Model type {model} not recognized.")

            vessel.setup(h, dt)
            self.vessels[id] = vessel
        
        self.set_inflows()
                
    def set_inflows(self):
        for v_id, inflow in self.inflows.items():
            if v_id in self.vessels:
                self.vessels[v_id].inflow = inflow
            else:
                raise ValueError(f"Vessel ID {v_id} not found in the network.")
