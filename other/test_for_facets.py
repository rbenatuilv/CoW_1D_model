import dolfinx
from mpi4py import MPI
import ufl


mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
n = ufl.FacetNormal(mesh)

# Integrate the x-component of the (+) normal over all internal facets
# In 1D, n[0] is just the scalar value of the normal direction
form = dolfinx.fem.form(n[0]('+') * ufl.dS)
orientation = dolfinx.fem.assemble_scalar(form)

# Since we have 9 internal facets in a 10-element interval:
# If orientation is 9.0 -> (+) is always the Left cell.
# If orientation is -9.0 -> (+) is always the Right cell.
print(f"Total orientation sum: {orientation}")