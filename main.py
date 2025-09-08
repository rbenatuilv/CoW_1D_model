from vascular_solver import VascularSolver
from dataload import VascularDataLoader
from vascular_net import VascularNetwork


from typing import Literal

def main(T: float = 1.0, mode: Literal["main", "test"] = "main", method: Literal["CG", "DG"] = "CG"):

    h = 3 * 0.03125
    dt = 1 * 1e-5

    data_loader = VascularDataLoader(mode=mode)
    vessels_data, bif_data, inflows = data_loader.load()

    network = VascularNetwork(vessels_data, bif_data, inflows)

    solver = VascularSolver(network, method=method)
    solver.setup(h=h, dt=dt)

    solver.solve(t_end=T)
    solver.plot_solutions(T, mode=mode)
    solver.save_solutions(mode=mode)


if __name__ == "__main__":
    import sys

    T = 1.0
    mode = "main"
    method = "CG"

    for i, arg in enumerate(sys.argv):
        if arg == "-T" and i + 1 < len(sys.argv):
            T = float(sys.argv[i + 1])
        elif arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--method" and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]

    if mode not in ["main", "test"]:
        raise ValueError("Mode must be either 'main' or 'test'.")
    if T <= 0:
        raise ValueError("T must be a positive number.")
    if method not in ["CG", "DG"]:
        raise ValueError("Method must be either 'CG' or 'DG'.")
    
    print(f"Running with T={T}, mode={mode}, method={method}")
    main(T=T, mode=mode, method=method) # type: ignore

