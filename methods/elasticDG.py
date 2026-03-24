from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type # type: ignore
from dolfinx.fem import petsc # type: ignore
import ufl
from basix.ufl import element
from petsc4py import PETSc # type: ignore

import numpy as np
from typing import Optional

from vessel_models.elastic_vessel import ElasticVessel


class ElasticDGVessel(ElasticVessel):

    method_type = "DG"

    def __init__(self, num_flux: str = "HLL", **kwargs):
        super().__init__(**kwargs)

        self.LB_fem = None
        self.RB_fem = None

        
        self.u = None  # Current time step solution
        self.u_n = None  # Previous time step solution

        self.diff_const = 12.53125
        
        if num_flux not in ["LxF", "HLL"]:
            raise ValueError(f"Numerical flux {num_flux} not recognized. Choose 'LxF' or 'HLL'.")
        self.num_flux = num_flux

    def create_mesh(self, h: float):

        n = int(self.L / h)
        self.mesh = mesh.create_interval(MPI.COMM_WORLD, n, (0, self.L))
        self.h = h

    def create_mesh_tags(self):
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim - 1)

        facet_imap = self.mesh.topology.index_map(tdim - 1)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts

        indices = np.arange(num_facets)
        values = np.zeros(num_facets)

        left_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
        right_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, lambda x: np.isclose(x[0], self.L))

        boundary_id = {"Gamma_L": 1, "Gamma_R": 2}
        values[left_facets] = boundary_id["Gamma_L"]
        values[right_facets] = boundary_id["Gamma_R"]

        self.msh_tags = mesh.meshtags(self.mesh, tdim - 1, indices, values)

    def create_fem_space(self):
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        elem = element("Lagrange", self.mesh.topology.cell_name(), 1, shape=(2, ), discontinuous=True)
        self.V = fem.functionspace(self.mesh, elem)

        self.u = fem.Function(self.V)  # Current time step solution
        self.v = fem.Function(self.V)  # auxiliary solution

        self.u_n = fem.Function(self.V)  # Previous
        self.v_n = fem.Function(self.V)  # auxiliary previous solution

    def set_initial_condition(self):
        if self.V is None or self.u_n is None:
            raise ValueError("Function space not created. Call create_fem_space() first.")
        
        self.u_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)
        self.v_n.interpolate(lambda x: np.tile(self.LB, (x.shape[1], 1)).T)

        comm = MPI.COMM_WORLD

        u = self.u_n

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 2)

        self.last_sol = global_sol



    def LxF(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u('+')), self.max_eigval(u('-')))

        # Lax-Friedrichs numerical flux
        # In fenicsx, u('+') and u('-') represent values from neighboring elements.
        # For some reason, the '+' represents the value from the "LEFT" side of the interface (instead of the right)
        # and the '-' represents the value from the "RIGHT" side of the interface (instead of the left).
        # That is why we use -ufl.jump(u) here.

        flux_avg = ufl.avg(self.F(u)) # type: ignore
        jump = ufl.jump(u)  # type: ignore
        return flux_avg + 0.5 * lambda_max * jump # type: ignore

    def LxF_bound_L(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.LB_fem))

        # Lax-Friedrichs numerical flux at left boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.LB_fem)) # type: ignore
        jump = self.LB_fem - u
        return flux_avg + 0.5 * lambda_max * jump # type: ignore

    def LxF_bound_R(self, u: fem.Function):
        lambda_max = ufl.max_value(self.max_eigval(u), self.max_eigval(self.RB_fem))

        # Lax-Friedrichs numerical flux at right boundary
        flux_avg = 0.5 * (self.F(u) + self.F(self.RB_fem)) # type: ignore
        jump = u - self.RB_fem
        return flux_avg + 0.5 * lambda_max * jump # type: ignore
    

    # HLL flux
    
    # def HLL_flux(self, u: fem.Function):
    #     SL = ufl.min_value(self.lambda1(u('-')), self.lambda1(u('+')))
    #     SR = ufl.max_value(self.lambda2(u('+')), self.lambda2(u('-')))

    #     FL = self.F(u('+'))  # type: ignore
    #     FR = self.F(u('-'))  # type: ignore

    #     jump = -ufl.jump(u)  # type: ignore 

    #     flux = ufl.conditional(
    #         ufl.ge(SL, 0),
    #         FL,
    #         ufl.conditional(
    #             ufl.le(SR, 0),
    #             FR,
    #             (SR * FL - SL * FR + SL * SR * jump) / (SR - SL)  # type: ignore
    #         )
    #     )

    #     return flux

    def HLL_flux(self, u: fem.Function):
        SL = ufl.min_value(self.lambda1(u('-')), self.lambda1((u('+')+u('-'))/2))
        SR = ufl.max_value(self.lambda2(u('+')), self.lambda2((u('+')+u('-'))/2))

        FL = self.F(u('-'))  # type: ignore
        FR = self.F(u('+'))  # type: ignore

        jump = ufl.jump(u)  # type: ignore 

        flux = ufl.conditional(
            ufl.ge(SL, 0),
            FL,
            ufl.conditional(
                ufl.le(SR, 0),
                FR,
                (SR * FL - SL * FR + SL * SR * -1*(jump)) / (SR - SL)  # type: ignore
            )
        )

        return flux


    # def HLL_bound_L(self, u: fem.Function):
    #     SL = ufl.min_value(self.lambda1(u), self.lambda1(self.LB_fem))
    #     SR = ufl.max_value(self.lambda2(self.LB_fem), self.lambda2(u))

    #     FL = self.F(self.LB_fem)  # type: ignore
    #     FR = self.F(u)  # type: ignore

    #     jump = u - self.LB_fem  # type: ignore

    #     flux = ufl.conditional(
    #         ufl.ge(SL, 0),
    #         FL,
    #         ufl.conditional(
    #             ufl.le(SR, 0),
    #             FR,
    #             (SR * FL - SL * FR + SL * SR * jump) / (SR - SL)  # type: ignore
    #         )
    #     )

    #     return flux
    
    def HLL_bound_L(self, u: fem.Function):
        SL = ufl.min_value(self.lambda1(self.LB_fem), self.lambda1((u+self.LB_fem)/2))
        SR = ufl.max_value(self.lambda2(u), self.lambda2((u+self.LB_fem)/2))

        FL = self.F(self.LB_fem)  # type: ignore
        FR = self.F(u)  # type: ignore

        jump = self.LB_fem - u  # type: ignore 

        flux = ufl.conditional(
            ufl.ge(SL, 0),
            FL,
            ufl.conditional(
                ufl.le(SR, 0),
                FR,
                (SR * FL - SL * FR + SL * SR * -1*(jump)) / (SR - SL)  # type: ignore
            )
        )

        return flux
    
    # def HLL_bound_R(self, u: fem.Function):
    #     SL = ufl.min_value(self.lambda1(self.RB_fem), self.lambda1(u))
    #     SR = ufl.max_value(self.lambda2(u), self.lambda2(self.RB_fem))

    #     FL = self.F(u)  # type: ignore
    #     FR = self.F(self.RB_fem)  # type: ignore

    #     jump = self.RB_fem - u  # type: ignore

    #     flux = ufl.conditional(
    #         ufl.ge(SL, 0),
    #         FL,
    #         ufl.conditional(
    #             ufl.le(SR, 0),
    #             FR,
    #             (SR * FL - SL * FR + SL * SR * jump) / (SR - SL)  # type: ignore
    #         )
    #     )

    #     return flux

    def HLL_bound_R(self, u: fem.Function):
        SL = ufl.min_value(self.lambda1(u), self.lambda1((u+self.RB_fem)/2))
        SR = ufl.max_value(self.lambda2(self.RB_fem), self.lambda2((u+self.RB_fem)/2))

        FL = self.F(u)  # type: ignore
        FR = self.F(self.RB_fem)  # type: ignore

        jump = u-self.RB_fem  # type: ignore 

        flux = ufl.conditional(
            ufl.ge(SL, 0),
            FL,
            ufl.conditional(
                ufl.le(SR, 0),
                FR,
                (SR * FL - SL * FR + SL * SR * -1*(jump)) / (SR - SL)  # type: ignore
            )
        )

        return flux
    
    def get_flux_function(self):
        """Return the appropriate flux function based on the selected numerical flux."""
        if self.num_flux == "LxF":
            return self.HxF_LxF
        elif self.num_flux == "HLL":
            return self.HxF_HLL
        else:
            raise ValueError(f"Numerical flux {self.num_flux} not recognized.")
    
    def dU_dz(self, u: np.ndarray):

        n_elems = u.shape[0] // 2
        h = self.L / n_elems

        u_reshaped = u.reshape(n_elems, 2, 2)

        grads = (u_reshaped[:, 1, :] - u_reshaped[:, 0, :]) / h

        return grads

    def HxF_LxF(self, u: fem.Function, v: fem.Function):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.msh_tags)

        jump = ufl.jump(v)  # type: ignore

        L = -ufl.inner(self.B(u), v) * ufl.dx # type: ignore
        L += ufl.inner(self.F(u), v.dx(0)) * ufl.dx # type: ignore
        L -= ufl.inner(self.LxF(u), jump) * ufl.dS # type: ignore
        L -= ufl.inner(self.LxF_bound_L(u), -v) * ds(1) # type: ignore
        L -= ufl.inner(self.LxF_bound_R(u), v) * ds(2) # type: ignore

        return L
    
    def HxF_HLL(self, u: fem.Function, v: fem.Function):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.msh_tags)

        jump = -ufl.jump(v)  # type: ignore

        L = -ufl.inner(self.B(u), v) * ufl.dx # type: ignore
        L += ufl.inner(self.F(u), v.dx(0)) * ufl.dx # type: ignore
        L += ufl.inner(self.HLL_flux(u), jump) * ufl.dS # type: ignore
        L += ufl.inner(self.HLL_bound_L(u), v) * ds(1) # type: ignore
        L += ufl.inner(self.HLL_bound_R(u), -v) * ds(2) # type: ignore

        return L

    def set_variational_problem(self, dt: float):
        self.dt = fem.Constant(self.mesh, default_scalar_type(dt))

        u1 = ufl.TrialFunction(self.V)
        v1 = ufl.TestFunction(self.V)
        
        # Get the selected numerical flux function
        flux_func = self.get_flux_function()

        a1 = ufl.inner(u1, v1) * ufl.dx
        L1 = ufl.inner(self.u_n, v1) * ufl.dx
        L1 += self.dt * flux_func(self.u_n, v1) # type: ignore

        self.bilinear_form_1 = fem.form(a1)
        self.linear_form_1 = fem.form(L1)

        u2 = ufl.TrialFunction(self.V)
        v2 = ufl.TestFunction(self.V)

        a2 = ufl.inner(u2, v2) * ufl.dx
        L2 = 0.5 * ufl.inner(self.u_n, v2) * ufl.dx # type: ignore
        L2 += 0.5 * ufl.inner(self.v_n, v2) * ufl.dx # type: ignore
        L2 += 0.5 * self.dt * flux_func(self.v_n, v2) # type: ignore

        self.bilinear_form_2 = fem.form(a2)
        self.linear_form_2 = fem.form(L2)

    def update_BCs(self, LB: Optional[np.ndarray] = None, RB: Optional[np.ndarray] = None):
        # Ensure we have valid boundary values
        left_bc = LB if LB is not None else self.LB
        right_bc = RB if RB is not None else self.RB
        
        if left_bc is None or right_bc is None:
            raise ValueError("Boundary conditions not properly initialized")
        
        if self.LB_fem is None:
            self.LB_fem = fem.Constant(self.mesh, left_bc)
        else:
            self.LB_fem.value = left_bc

        if self.RB_fem is None:
            self.RB_fem = fem.Constant(self.mesh, right_bc)
        else:
            self.RB_fem.value = right_bc

    def assemble_solver(self):
        if self.mesh is None or self.V is None:
            raise ValueError("Mesh or function space not created. Call create_mesh() and create_fem_space() first.")
        
        self.A1 = petsc.assemble_matrix(self.bilinear_form_1)
        self.A1.assemble()

        self.rhs_1 = petsc.create_vector(self.linear_form_1)

        self.solver_1 = PETSc.KSP().create(self.mesh.comm)
        self.solver_1.setOperators(self.A1)
        self.solver_1.setType(PETSc.KSP.Type.CG)
        self.solver_1.getPC().setType(PETSc.PC.Type.BJACOBI)

        self.A2 = petsc.assemble_matrix(self.bilinear_form_2)
        self.A2.assemble()

        self.rhs_2 = petsc.create_vector(self.linear_form_2)

        self.solver_2 = PETSc.KSP().create(self.mesh.comm)
        self.solver_2.setOperators(self.A2)
        self.solver_2.setType(PETSc.KSP.Type.CG)
        self.solver_2.getPC().setType(PETSc.PC.Type.BJACOBI)

    def setup(self, h: float, dt: float):
        self.create_mesh(h)
        self.create_mesh_tags()
        self.create_fem_space()
        self.set_initial_condition()  # Move this before update_BCs
        self.update_BCs()
        self.set_variational_problem(dt)
        self.assemble_solver()

        self.h = h

    def minmod(self, a: float, b: float, c: float) -> float:
        """
        Standard minmod function
        Returns the value with smallest absolute value if all have same sign, 0 otherwise
        """
        if a > 0 and b > 0 and c > 0:
            return min(abs(a), abs(b), abs(c))
        elif a < 0 and b < 0 and c < 0:
            return -min(abs(a), abs(b), abs(c))
        else:
            return 0.0

    def tvb_minmod(self, a: float, b: float, c: float, M: float, h: float) -> float:
        """
        TVB (Total Variation Bounded) modified minmod function
        M: TVB parameter (M=0 → standard minmod, M>0 → less limiting in smooth regions)
        """
        if abs(a) <= M * h**2:
            return a
        else:
            return self.minmod(a, b, c)

    def apply_slope_limiter(self, u_func: fem.Function, M: float = 0.0) -> np.ndarray:
        """
        Apply TVB slope limiter to a FEM function for DG1 elements
        
        Parameters:
        -----------
        u_func : fem.Function
            The function to limit (should have 2 components: A and Q)
        M : float
            TVB parameter (M=0 → minmod, M>0 → less limiting in smooth regions)
            Typical values: 0 (most diffusive), 10, 50, 100 (less diffusive)
        
        Returns:
        --------
        limited_array : np.ndarray
            The limited solution array
        """
        
        # Create a copy of the solution array to modify
        uA_loc = u_func.sub(0).collapse().x.array    # area component
        uQ_loc = u_func.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        u = local_sol
        n_elems = u.shape[0] // 2
        u_reshaped = u.reshape(n_elems, 2, 2) 

        h = self.L / n_elems

        for i in range(n_elems):
            for comp in range(2):  # Loop over components A and Q
                uL = u_reshaped[i, 0, comp]
                uR = u_reshaped[i, 1, comp]
                u_avg = 0.5 * (uL + uR)
                slope = (uR - uL) / h

                if i == 0:
                    uL_neighbor = u_reshaped[i, 0, comp]  # Use own left value for left boundary
                else:
                    uL_neighbor = 0.5 * (u_reshaped[i-1, 0, comp] + u_reshaped[i-1, 1, comp])

                if i == n_elems - 1:
                    uR_neighbor = u_reshaped[i, 1, comp]  # Use own right value for right boundary
                else:
                    uR_neighbor = 0.5 * (u_reshaped[i+1, 0, comp] + u_reshaped[i+1, 1, comp])

                delta_left = u_avg - uL_neighbor
                delta_right = uR_neighbor - u_avg

                limited_slope = self.tvb_minmod(slope, delta_left / h, delta_right / h, M, h)

                # Update limited values
                u_reshaped[i, 0, comp] = u_avg - 0.5 * limited_slope * h
                u_reshaped[i, 1, comp] = u_avg + 0.5 * limited_slope * h

        return u_reshaped.reshape(-1)

    def solve(self):
        with self.rhs_1.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs_1, self.linear_form_1)
        self.rhs_1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.solver_1.solve(self.rhs_1, self.v.x.petsc_vec)
        self.v.x.scatter_forward()

        self.v.x.array[:] = self.apply_slope_limiter(self.v)
        self.v_n.x.array[:] = self.v.x.array[:]

        with self.rhs_2.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(self.rhs_2, self.linear_form_2)
        self.rhs_2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) 
        
        self.solver_2.solve(self.rhs_2, self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        self.u.x.array[:] = self.apply_slope_limiter(self.u)
        self.u_n.x.array[:] = self.u.x.array[:]

        if np.any(np.isnan(self.u.x.array)):
            raise ValueError("NaN values encountered in the solution.")

    def get_max_cfl_dt(self):
        """Calculate maximum stable time step based on CFL condition"""
        # Get maximum wave speed (eigenvalue)
        c_max = np.sqrt(self.beta / (2 * self.rho * self.A0))  # Approximate wave speed
        return 0.5 * self.h / c_max  # CFL < 0.5 for stability


    def add_solution(self, t: float):
        if self.mesh is None or self.V is None:
            raise ValueError("Mesh or function space not created. Call create_mesh() and create_fem_space() first.")

        if self.u is None:
            raise ValueError("No solution available. Call solve() first.")
        
        comm = self.mesh.comm
        rank = comm.Get_rank()

        u = self.u

        # 1) Extract local solution from the function without ghosts
        uA_loc = u.sub(0).collapse().x.array    # area component
        uQ_loc = u.sub(1).collapse().x.array    # flux component
        local_sol = np.stack([uA_loc, uQ_loc], axis=-1)  # shape (n_local, 2)

        # 2) Gather all local solutions across processes
        all_sols = comm.allgather(local_sol)   # returns array list [(n1,2), (n2,2), ...]

        # 3) Concatenate all local solutions into a global solution
        global_sol = np.vstack(all_sols)        # shape (n_total, 2)

        self.last_sol = global_sol

        if (rank == 0) and (t - self.last_saved_time) >= 1e-5:
            self.solutions["t"].append(t)
            self.solutions["A"].append(global_sol[:, 0])
            self.solutions["Q"].append(global_sol[:, 1])

            mid_index = len(global_sol) // 2
            self.middlepoints["A"].append(global_sol[mid_index, 0])
            self.middlepoints["Q"].append(global_sol[mid_index, 1])

            self.last_saved_time = t

    def visualize_mesh_and_tags(self, save_path: str = "mesh_visualization.png"):
        """
        Visualize the mesh, mesh tags, and DG elements to verify correct setup.
        
        Parameters:
        -----------
        save_path : str
            Path to save the visualization image
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if self.mesh is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")
        
        comm = self.mesh.comm
        rank = comm.Get_rank()
        
        if rank == 0:
            # Get mesh coordinates
            tdim = self.mesh.topology.dim
            coords = self.mesh.geometry.x
            
            # Get cells (elements)
            num_cells = self.mesh.topology.index_map(tdim).size_local
            cell_indices = np.arange(num_cells)
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
            
            # Plot 1: Mesh elements with node numbers
            ax1.set_title("Mesh Elements and Nodes", fontsize=14, fontweight='bold')
            for i in cell_indices:
                # Get cell vertices
                cell = self.mesh.topology.connectivity(tdim, 0).links(i)
                x_coords = coords[cell, 0]
                
                # Draw element
                x_min, x_max = x_coords.min(), x_coords.max()
                rect = Rectangle((x_min, -0.1), x_max - x_min, 0.2, 
                            fill=True, facecolor='lightblue', 
                            edgecolor='black', linewidth=2)
                ax1.add_patch(rect)
                
                # Label element center
                x_center = (x_min + x_max) / 2
                ax1.text(x_center, 0, f'Cell {i}', ha='center', va='center', 
                        fontsize=8, fontweight='bold')
                
                # Mark nodes
                for node_idx, x in zip(cell, x_coords):
                    ax1.plot(x, 0, 'ro', markersize=8)
                    ax1.text(x, 0.15, f'Node {node_idx}', ha='center', 
                            fontsize=7, rotation=45)
            
            ax1.set_xlim(-0.05 * self.L, 1.05 * self.L)
            ax1.set_ylim(-0.3, 0.3)
            ax1.set_xlabel("x [cm]", fontsize=12)
            ax1.set_yticks([])
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 2: Mesh tags (boundary markers)
            ax2.set_title("Mesh Tags (Boundary Markers)", fontsize=14, fontweight='bold')
            if hasattr(self, 'msh_tags') and self.msh_tags is not None:
                # Get facet coordinates
                facet_dim = tdim - 1
                facet_imap = self.mesh.topology.index_map(facet_dim)
                num_facets = facet_imap.size_local
                
                for i in range(num_facets):
                    tag_value = self.msh_tags.values[i]
                    facet_vertices = self.mesh.topology.connectivity(facet_dim, 0).links(i)
                    x_facet = coords[facet_vertices, 0][0]
                    
                    if tag_value == 1:  # Gamma_L
                        ax2.plot(x_facet, 0, 'gs', markersize=15, label='Gamma_L (Left)' if i == 0 else '')
                        ax2.text(x_facet, 0.15, 'Γ_L', ha='center', fontsize=12, fontweight='bold', color='green')
                    elif tag_value == 2:  # Gamma_R
                        ax2.plot(x_facet, 0, 'rs', markersize=15, label='Gamma_R (Right)' if i == num_facets-1 else '')
                        ax2.text(x_facet, 0.15, 'Γ_R', ha='center', fontsize=12, fontweight='bold', color='red')
                    else:
                        ax2.plot(x_facet, 0, 'ko', markersize=5)
                
                ax2.legend(loc='upper right', fontsize=10)
            
            # Draw mesh outline
            for i in cell_indices:
                cell = self.mesh.topology.connectivity(tdim, 0).links(i)
                x_coords = coords[cell, 0]
                x_min, x_max = x_coords.min(), x_coords.max()
                rect = Rectangle((x_min, -0.05), x_max - x_min, 0.1, 
                            fill=False, edgecolor='black', linewidth=1, linestyle='--')
                ax2.add_patch(rect)
            
            ax2.set_xlim(-0.05 * self.L, 1.05 * self.L)
            ax2.set_ylim(-0.3, 0.3)
            ax2.set_xlabel("x [cm]", fontsize=12)
            ax2.set_yticks([])
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 3: DG1 DOF structure
            ax3.set_title("DG1 Element Structure (DOFs per element)", 
                        fontsize=14, fontweight='bold')
            
            if self.V is not None:
                dofmap = self.V.dofmap
                for i in cell_indices:
                    cell = self.mesh.topology.connectivity(tdim, 0).links(i)
                    x_coords = coords[cell, 0]
                    x_min, x_max = x_coords.min(), x_coords.max()
                    
                    # Draw element
                    rect = Rectangle((x_min, -0.1), x_max - x_min, 0.2, 
                                fill=True, facecolor='lightyellow', 
                                edgecolor='black', linewidth=2)
                    ax3.add_patch(rect)
                    
                    # Get DOFs for this cell
                    dofs = dofmap.cell_dofs(i)
                    print(f"Cell {i}: DOFs = {dofs}")  # Debug print
                    
                    # DG elements have DOFs at element centers, not nodes
                    # For vector DG1 with 2 nodes and 2 components, we expect different structure
                    x_center = (x_min + x_max) / 2
                    
                    # Display all DOFs for this cell
                    dof_text = f"Cell {i} DOFs:\n" + "\n".join([f"DOF {d}" for d in dofs])
                    ax3.text(x_center, 0, dof_text, ha='center', va='center',
                            fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax3.set_xlim(-0.05 * self.L, 1.05 * self.L)
            ax3.set_ylim(-0.35, 0.35)
            ax3.set_xlabel("x [cm]", fontsize=12)
            ax3.set_yticks([])
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✓ Mesh visualization saved to {save_path}")
            
            # Print summary statistics
            print("\n" + "="*60)
            print("MESH SUMMARY")
            print("="*60)
            print(f"Total length: {self.L} cm")
            print(f"Number of elements: {num_cells}")
            print(f"Element size (h): {self.h:.6f} cm")
            if self.V:
                print(f"Total DOFs: {self.V.dofmap.index_map.size_global}")
                print(f"DOFs per element: {len(dofmap.cell_dofs(0))}")  # type: ignore
            print("="*60)


if __name__ == "__main__":
    # Test mesh visualization
    data = {
        "id": 0,
        "length": 1,
        "initial_area": 0.126,
        "beta_coeff": 0.060606e7,
        "left_bound": "inflow",
        "right_bound": "outflow"
    }

    vessel = ElasticDGVessel(**data)
    h = 0.125  # Use coarser mesh for visualization
    vessel.create_mesh(h)
    vessel.create_mesh_tags()
    vessel.create_fem_space()
    
    # Visualize the mesh structure
    vessel.visualize_mesh_and_tags("mesh_test.png")
    