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
    num_fluxes = ["LxF", "HLL"] # 
    methods = ["CG", "DG"] # 
    Ns = np.array([16, 32, 64, 128, 256]) # 
    hs = 1/Ns
    for h in hs:
        for method in methods:
            if method=="CG":
                main(T=T, mode=mode, method=method, num_flux=num_flux, name=name, h=h) # type: ignore
                print(f"Simulation completed: Method: {method}, h: {h}\n")

            else:
                for num_flux in num_fluxes:
                    if 1/h <= 128 and num_flux == "HLL":
                        continue
                    main(T=T, mode=mode, method=method, num_flux=num_flux, name=name, h=h) # type: ignore
                    print(f"Simulation completed: Method: {method}, num_flux: {num_flux}, h: {h}\n")
