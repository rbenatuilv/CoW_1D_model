from vesselsystem import VesselSystem
from vascular import VascularSolver
from dataload import VascularDataLoader
import profiler.profiler as prf


def main(T: float = 1.0, mode: str = "main", profile: bool = False):
    prf.ENABLE_PROFILING = profile

    T = T
    h = 2 * 0.03125
    dt = 1 * 1e-5 

    mode = mode  # Change to "main" for main mode

    data_loader = VascularDataLoader(mode=mode)
    vessels_data, bif_data, inflows = data_loader.load()

    system = VesselSystem(
        vessels_data=vessels_data,
        bifurcations_data=bif_data
    )

    system.set_inflows(inflows)

    solver = VascularSolver(h=h, dt=dt)
    solver.set_system(system, "viscoelastic")

    solver.solve(T)
    solver.plot_solutions(T=T, mode=mode)
    solver.save_solutions(mode=mode)

    if mode == "test" and prf.ENABLE_PROFILING:
        prf.print_profiled_summary(limit=10)


if __name__ == "__main__":
    import sys

    # Search the -T argument in the command line arguments
    T = 1.0
    mode = "main"
    profile = False
    for i, arg in enumerate(sys.argv):
        if arg == "-T" and i + 1 < len(sys.argv):
            T = float(sys.argv[i + 1])
        elif arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--time-profile":
            profile = True

    if mode not in ["main", "test"]:
        raise ValueError("Mode must be either 'main' or 'test'.")
    if T <= 0:
        raise ValueError("T must be a positive number.")
    print(f"Running with T={T}, mode={mode}, time profile={profile}")

    main(T=T, mode=mode, profile=profile)


