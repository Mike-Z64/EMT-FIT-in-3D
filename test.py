from Rudy_Open_Boundary import FITGridEven
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pprint import pprint
import matplotlib.animation as animation
from matplotlib.widgets import Slider



def topological_matrices(grid):
    """Compute topological operators and store them in the nested structure.""" 
    
    # we can just extract the grid dimensions into nx,ny,nz for simplicity
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]

    # Differentiation matrices (P) for x, y, z directions using central differences ; .tocsr() converts to Compressed Sparse Row format for more efficient math
    Px = sp.diags([-1, 1], [0, 1], shape=(nx, nx)).tocsr()
    Py = sp.diags([-1, 1], [0, 1], shape=(ny, ny)).tocsr()
    Pz = sp.diags([-1, 1], [0, 1], shape=(nz, nz)).tocsr()

    # Identity matrices for Kronecker products
    Ix, Iy, Iz = sp.eye(nx), sp.eye(ny), sp.eye(nz)

    # 3D Discrete Gradient operators
    Dx = sp.kron(Iz, sp.kron(Iy, Px)).tocsr()
    Dy = sp.kron(Iz, sp.kron(Py, Ix)).tocsr()
    Dz = sp.kron(Pz, sp.kron(Iy, Ix)).tocsr()

    # Curl Matrix (C)
    C = sp.bmat([
        [None, -Dz, Dy],
        [Dz, None, -Dx],
        [-Dy, Dx, None],
    ], format="csr")

    C_hat = C.transpose()

    # Divergence Matrix (S)
    S = sp.bmat([[Dx, Dy, Dz]], format="csr")
    
    S_hat = S.transpose()

    # Store matrices in the top level or a specific sub-dict
    grid["operators"] = {"C": C, "S": S, "C_hat": C_hat, "S_hat": S_hat}

    return C, S, C_hat, S_hat

def main():

    # grid parameters
    x_len, y_len, z_len = 1, 1, 1
    nx, ny, nz = 3, 3, 3    



    # Define grid and topological matrices
    grid = FITGridEven(x_len, y_len, z_len, nx, ny, nz)
    C, S, C_hat, S_hat = topological_matrices(grid)
    print(f"--- Topological Verification ---")

    # Extract the gradient matrices
    Dx = grid["operators"]["S"].toarray()[:, :nx*ny*nz]  # Wait, no: S is [[Dx, Dy, Dz]], so S is (N, 3N), Dx is the first N columns? Wait.

    # Actually, S = sp.bmat([[Dx, Dy, Dz]]), so S is (N, 3N), with Dx in columns 0 to N-1, Dy N to 2N-1, Dz 2N to 3N-1.

    N = nx * ny * nz
    Dx_mat = S[:, :N]
    Dy_mat = S[:, N:2*N]
    Dz_mat = S[:, 2*N:3*N]

    # Plot the 2D matrices as sparsity patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].spy(Dx_mat, markersize=1)
    axes[0].set_title('Dx Matrix Sparsity')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')

    axes[1].spy(Dy_mat, markersize=1)
    axes[1].set_title('Dy Matrix Sparsity')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')

    axes[2].spy(Dz_mat, markersize=1)
    axes[2].set_title('Dz Matrix Sparsity')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change 'dipole' to 'point' to use a point source instead
    main()