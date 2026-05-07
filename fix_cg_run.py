from main import main
import numpy as np

if __name__ == "__main__":
    import sys

    T = 1.0
    mode = "main"
    method = "CG"
    num_flux = "LxF"
    name = None

    for i, arg in enumerate(sys.argv):
        if arg == "-T" and i + 1 < len(sys.argv):
            T = float(sys.argv[i + 1])
        elif arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--name" and i + 1 < len(sys.argv):
            name = sys.argv[i + 1]

    if mode not in ["main", "test", "test_single", "sin", "sin_single"]:
        raise ValueError("Mode must be either 'main' or 'test'.")
    if T <= 0:
        raise ValueError("T must be a positive number.")
    else:
        print(f"Running with T={T}, mode={mode}\n")
    methods = ["CG"] # 
    Ns = np.array([16, 32, 64, 128, 256]) # 
    dt = np.array([3.2e-5, 1.6e-5, 8e-6, 4e-6, 2e-6]) # 
    hs = 1/Ns
    for i,h in enumerate(hs):
        for method in methods:
            if method=="CG":
                main(T=T, mode=mode, method=method, num_flux=num_flux, name=name, h=h, force_dt=True, dt_forced=dt[i]) # type: ignore
                print(f"Simulation completed: Method: {method}, h: {h}\n")

            else:
                for num_flux in num_fluxes:
                    main(T=T, mode=mode, method=method, num_flux=num_flux, name=name, h=h) # type: ignore
                    print(f"Simulation completed: Method: {method}, num_flux: {num_flux}, h: {h}\n")
