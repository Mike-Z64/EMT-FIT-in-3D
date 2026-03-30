import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pprint import pprint
import matplotlib.animation as animation

###############################
## GRID definition function  ##
###############################

def FITGridEven(x_len, y_len, z_len, nx, ny, nz):
    """Create a 3D FIT grid using a nested dictionary structure."""

    ### Main dictionary with metadata and nested grids: G and G_hat have the same variable names for consistency
    grid = {
        "nx": nx,     # Number of points in x-direction (same for primary and dual)
        "ny": ny,     # Number of points in y-direction (same for primary and dual)
        "nz": nz,     # Number of points in z-direction (same for primary and dual)
        "G": {},      # Primary Grid
        "G_hat": {}   # Dual/Staggered Grid
    }

    ### ------ PRIMARY GRID G -------
    # Coordinates
    grid["G"]["x"] = np.linspace(0, x_len, nx)
    grid["G"]["y"] = np.linspace(0, y_len, ny)
    grid["G"]["z"] = np.linspace(0, z_len, nz)

    # Compute physical spacings of each cell (assuming uniform grid)
    dx = x_len / (nx - 1)
    dy = y_len / (ny - 1)
    dz = z_len / (nz - 1)
    
    grid["G"]["dx"] = dx
    grid["G"]["dy"] = dy
    grid["G"]["dz"] = dz

    # Meshgrids for areas and volumes
    # Note: We use nx-1 for cell-based dimensions
    DX, DY, DZ = np.meshgrid(
        np.full(nx-1, dx), np.full(ny-1, dy), np.full(nz-1, dz), indexing="ij"
    )
    
    grid["G"]["area_x"] = DY * DZ
    grid["G"]["area_y"] = DX * DZ
    grid["G"]["area_z"] = DX * DY
    grid["G"]["volume"] = DX * DY * DZ

    ### ------ DUAL GRID G_HAT -------
    # Shifted points (staggered by half a cell)
    grid["G_hat"]["x"] = grid["G"]["x"][:-1] + (dx / 2.0) # previously used grid["G"]["x"][:-1] but we want to keep the same number of points
    grid["G_hat"]["y"] = grid["G"]["y"][:-1] + (dy / 2.0)
    grid["G_hat"]["z"] = grid["G"]["z"][:-1] + (dz / 2.0)

    # In an even grid, dual spacings are identical
    grid["G_hat"]["dx"] = dx
    grid["G_hat"]["dy"] = dy
    grid["G_hat"]["dz"] = dz

    # Dual geometry matches primary cell geometry in this setup, so we can copy the areas and volumes
    grid["G_hat"]["area_x"] = grid["G"]["area_x"].copy()
    grid["G_hat"]["area_y"] = grid["G"]["area_y"].copy()
    grid["G_hat"]["area_z"] = grid["G"]["area_z"].copy()
    grid["G_hat"]["volume"] = grid["G"]["volume"].copy()

    # plotting for verification
    #plot_grids(grid)
    print(f"Grid successfully created with {nx * ny * nz} total nodes!")
    return grid

############################################
## C and S topological matrices function  ##
############################################

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

##############################
## Node indexing functions  ##
##############################

def get_node_index(i, j, k, nx, ny):
    """
    Calculates the 1D global node index 'n' from 3D coordinates (i, j, k) based on equation in my notes
    """
    return i + (j * nx) + (k * nx * ny)

def get_ijk_of_node(n, nx, ny):
    """
    Inverse of get_node_index: 
    Converts a 1D global node index 'n' back into 3D coordinates (i, j, k).
    """
    i = n % nx
    j = (n // nx) % ny
    k = n // (nx * ny)
    
    return i, j, k

#############################
## GRID PLOTTING function  ##
#############################

def plot_grids(grid):
    """Generates a 3D plot and two 2D cross-sections of the primary and dual grids."""
    
    # Extract coordinates and dimensions
    x_G, y_G, z_G = grid["G"]["x"], grid["G"]["y"], grid["G"]["z"]
    x_H, y_H, z_H = grid["G_hat"]["x"], grid["G_hat"]["y"], grid["G_hat"]["z"]
    
    # Get lengths for both grids
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    nx_H, ny_H, nz_H = len(x_H), len(y_H), len(z_H)

    fig = plt.figure(figsize=(14, 10))

    # --- 1. 3D Plot (Left Side) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Create 3D meshes
    XG, YG, ZG = np.meshgrid(x_G, y_G, z_G, indexing='ij')
    XH, YH, ZH = np.meshgrid(x_H, y_H, z_H, indexing='ij')
    
    ax1.scatter(XG, YG, ZG, color='blue', alpha=0.3, s=20, label='Primary (G)')
    ax1.scatter(XH, YH, ZH, color='red', alpha=0.8, s=40, label='Dual (G_hat)')

    # Primary Numbering (Black)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = get_node_index(i, j, k, nx, ny)
                ax1.text(x_G[i], y_G[j], z_G[k], f' {n}', color='black', fontsize=9, fontweight='bold')

    # Dual Numbering (Dark Red)
    for i in range(nx_H):
        for j in range(ny_H):
            for k in range(nz_H):
                # We use the primary nx, ny here to keep the numbering system strictly identical
                n = get_node_index(i, j, k, nx, ny)
                ax1.text(x_H[i], y_H[j], z_H[k], f' {n}', color='darkred', fontsize=9, fontweight='bold')

    ax1.set_title("3D Grid View with Global Numbering")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    # --- 2. 2D XY Cross-section (Top Right) ---
    ax2 = fig.add_subplot(2, 2, 2)
    XG2, YG2 = np.meshgrid(x_G, y_G, indexing='ij')
    XH2, YH2 = np.meshgrid(x_H, y_H, indexing='ij')
    
    ax2.scatter(XG2, YG2, color='blue', alpha=0.3, s=20)
    ax2.scatter(XH2, YH2, color='red', s=40)
    
    # Primary Numbering (Z plane k=0)
    k = 0
    for i in range(nx):
        for j in range(ny):
            n = get_node_index(i, j, k, nx, ny)
            ax2.text(x_G[i]+0.02, y_G[j]+0.02, str(n), color='black', fontsize=10, fontweight='bold')

    # Dual Numbering (Dual Z plane k=0)
    for i in range(nx_H):
        for j in range(ny_H):
            n = get_node_index(i, j, 0, nx, ny)
            ax2.text(x_H[i]+0.02, y_H[j]+0.02, str(n), color='darkred', fontsize=10, fontweight='bold')

    ax2.set_title("XY Plane Cross-section (Primary Z=0)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- 3. 2D XZ Cross-section (Bottom Right) ---
    ax3 = fig.add_subplot(2, 2, 4)
    XG3, ZG3 = np.meshgrid(x_G, z_G, indexing='ij')
    XH3, ZH3 = np.meshgrid(x_H, z_H, indexing='ij')
    
    ax3.scatter(XG3, ZG3, color='blue', alpha=0.3, s=20)
    ax3.scatter(XH3, ZH3, color='red', s=40)
    
    # Primary Numbering (Y plane j=0)
    j = 0
    for i in range(nx):
        for k in range(nz):
            n = get_node_index(i, j, k, nx, ny)
            ax3.text(x_G[i]+0.02, z_G[k]+0.02, str(n), color='black', fontsize=10, fontweight='bold')

    # Dual Numbering (Dual Y plane j=0)
    for i in range(nx_H):
        for k in range(nz_H):
            n = get_node_index(i, 0, k, nx, ny)
            ax3.text(x_H[i]+0.02, z_H[k]+0.02, str(n), color='darkred', fontsize=10, fontweight='bold')

    ax3.set_title("XZ Plane Cross-section (Primary Y=0)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

######################################################################
## Medium initialization function based on Flisgen - slides 40-41  ##
######################################################################

def initialize_vacuum_medium(grid):
    """
    Initializes the material relation matrices D_epsilon and D_mu for vacuum.
    Based on Slides 40-41 of Flisgen 
    """
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    Np = nx * ny * nz  # Total number of primary nodes
    
    # Extract physical spacings from the primary grid
    dx = grid["G"]["dx"]
    dy = grid["G"]["dy"]
    dz = grid["G"]["dz"]
    
    # Fundamental Constants
    eps0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
    mu0 = 1.25663706212e-6   # Vacuum permeability [H/m]
    
    # --- 1. Construct D_epsilon (Slide 40) ---
    # Entries: eps0 * (Dual Area) / (Primary Edge Length)
    # For X-edges: eps0 * (dy * dz) / dx
    # For Y-edges: eps0 * (dx * dz) / dy
    # For Z-edges: eps0 * (dx * dy) / dz
    diag_eps_x = np.full(Np, eps0 * (dy * dz) / dx)
    diag_eps_y = np.full(Np, eps0 * (dx * dz) / dy)
    diag_eps_z = np.full(Np, eps0 * (dx * dy) / dz)
    
    # Stack into 3Np diagonal vector
    diag_eps = np.concatenate([diag_eps_x, diag_eps_y, diag_eps_z])
    D_eps = sp.diags(diag_eps, format="csr")
    
    # --- 2. Construct D_mu (Slide 41) ---
    # Entries: mu0 * (Primary Area) / (Dual Edge Length)
    # In an even grid, dual and primary dimensions match
    diag_mu_x = np.full(Np, mu0 * (dy * dz) / dx)
    diag_mu_y = np.full(Np, mu0 * (dx * dz) / dy)
    diag_mu_z = np.full(Np, mu0 * (dx * dy) / dz)
    
    diag_mu = np.concatenate([diag_mu_x, diag_mu_y, diag_mu_z])
    D_mu = sp.diags(diag_mu, format="csr")
    
    # Store in the grid metadata
    grid["D_eps"] = D_eps
    grid["D_mu"] = D_mu
    
    return D_eps, D_mu

#######################################################
## Define point source with pre-specified direction  ##
#######################################################

def create_point_source(nx, ny, nz, i, j, k, direction):
    """
    Creates a point source (infinitesimal dipole) at a specific 3D coordinate.
    Returns the exact 1D index for the solver's 3N vector.
    """
    # Get the base node number
    n = get_node_index(i, j, k, nx, ny)
    N_total = nx * ny * nz
    
    # find the corresponding edge index based on the direction and slide 22 of the Flisgen slides
    if direction == 'x':
        return n
    elif direction == 'y':
        return n + N_total
    elif direction == 'z':
        return n + (2 * N_total)

def leapfrog_scheme(e_initial, h_initial, b, d, C, C_hat, M_eps, M_mu, j_source, delta_t, n_steps, frequency, source_index, pec_indices=None, pmc_indices=None):
    """
    Leapfrog scheme function
    ------------------------
    Parameters:
    e_initial - initial electric field
    h_initial - initial magnetic field
    h - magnetic field
    b - magnetic flux density
    d - electric displacement
    C - curl matrix
    C_hat - dual curl matrix
    M_eps - permittivity matrix
    M_mu_inv - inverse of permeability matrix
    j_source - current source
    delta_t - time step
    n_steps - number of time steps
    frequency - frequency of the source (default 1 GHz)
    pec_indices - list of indices where PEC boundary conditions are applied (optional)
    pmc_indices - list of indices where PMC boundary conditions are applied (optional)

    Returns:
    e - updated electric field
    h - updated magnetic field

    General Notes:
    - The leapfrog scheme updates the electric and magnetic fields in a staggered manner 
    - The shape of the output arrays e and h are (n_steps + 1, N_e) and (n_steps + 1, N_h) respectively, 
    where N_e and N_h are spatial components of the fields and the rows are the time steps.
    - The initial conditions are stored in the first row (index 0) of the output arrays.
    """
    print(f"Updating fields for {n_steps} time steps...")

    # Preprocessing for efficiency: compute inverses of M_mu and M_eps once
    M_mu_inv = sp.diags(1.0 / M_mu.diagonal())
    M_eps_inv = sp.diags(1.0 / M_eps.diagonal())

    N_e = e_initial.shape[0]
    N_h = h_initial.shape[0]

    e = np.zeros((n_steps + 1, N_e))
    h = np.zeros((n_steps + 1, N_h))

    # Store initial conditions
    e[0, :] = e_initial
    h[0, :] = h_initial

    b = M_mu @ h_initial  # Initial magnetic flux density
    d = M_eps @ e_initial  # Initial electric displacement

    omega = 2 * np.pi * frequency  # Angular frequency for the source
    amplitude = 1.0  # Amplitude of the source 

    for n in range(n_steps):
        # e and d are updated at time steps n * delta_t
        # h, b and j are updated at time steps (n + 0.5) * delta_t
        # delta_t_half = (n + 0.5) * delta_t

        # Step 1: Update Magnetic Flux (b)
        # Equation: b^(n+1/2) = b^(n-1/2) - delta_t * (C * e^n)
        b = b - delta_t * (C @ e[n,:])

        # Step 2: Update Magnetic Field (h)
        # Equation: h^(n+1/2) = M_mu^-1 * b^(n+1/2)
        h_temp = M_mu_inv @ b

        # Apply PMC Boundary Condition (if provided)
        if pmc_indices is not None:
            h_temp[pmc_indices] = 0.0
            b[pmc_indices] = 0.0 # Zero out flux to prevent non-physical accumulation

        h[n + 1, :] = h_temp

        # Step 3: Update Electric Flux (d)
        # Equation: d^(n+1) = d^n + delta_t * (C_hat * h^(n+1/2))
        d = d + delta_t * (C_hat @ h_temp)

        # Step 4: Convert Electric Flux (d) to Electric Field (e)
        # Equation: e^(n+1) = M_eps^-1 * d^(n+1)
        e_temp = M_eps_inv @ d

        # Apply PEC Boundary Condition (if provided)
        if pec_indices is not None:
            e_temp[pec_indices] = 0.0
            d[pec_indices] = 0.0 # Zero out flux to prevent non-physical accumulation

        e[n + 1, :] = e_temp

        curr_time = (n + 1) * delta_t
        src_voltage = amplitude * np.sin(omega * curr_time)  # Sinusoidal source for demonstration
        e[n+1,source_index] = src_voltage
        d[source_index] = M_eps.diagonal()[source_index] * src_voltage
    return e, h


def animate_slice(history_array, grid, plane, slice_index, component='z', is_db=True):
    """
    Creates an animation of a specific field component in a chosen plane.
    
    Parameters:
    - plane: 'xy', 'xz', or 'yz'
    - slice_index: The grid index (0 to nx/ny/nz) to slice through.
    - component: 'x', 'y', or 'z'
    - is_db: If True, uses sequential colormap for dB. If False, uses diverging for raw fields.
    """
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    N_total = nx * ny * nz
    total_steps = history_array.shape[0]
    
    # 1. Determine the offset for the chosen component
    if component == 'x':
        offset = 0
    elif component == 'y':
        offset = N_total
    elif component == 'z':
        offset = 2 * N_total
    else:
        raise ValueError("Component must be 'x', 'y', or 'z'")
        
    print(f"\nExtracting {component}-directed component in the {plane.upper()} plane...")

    # 2. Reshape the 1D data into a 4D array: (time, Z, Y, X)
    # This completely removes the need for slow nested for-loops!
    field_3d = history_array[:, offset : offset + N_total].reshape((total_steps, nz, ny, nx))
    
    # 3. Extract the requested 2D slice and setup the plot limits
    if plane.lower() == 'xy':
        frames = field_3d[:, slice_index, :, :]
        extent = [0, grid["G"]["x"][-1], 0, grid["G"]["y"][-1]]
        xlabel, ylabel = 'X coordinate (m)', 'Y coordinate (m)'
    elif plane.lower() == 'xz':
        frames = field_3d[:, :, slice_index, :]
        extent = [0, grid["G"]["x"][-1], 0, grid["G"]["z"][-1]]
        xlabel, ylabel = 'X coordinate (m)', 'Z coordinate (m)'
    elif plane.lower() == 'yz':
        frames = field_3d[:, :, :, slice_index]
        extent = [0, grid["G"]["y"][-1], 0, grid["G"]["z"][-1]]
        xlabel, ylabel = 'Y coordinate (m)', 'Z coordinate (m)'
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

    print("Frames extracted! Starting playback...")

    # 4. Setup the Animation Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Choose colormap based on whether we are plotting dB or raw fields
    if is_db:
        cmap = 'magma'
        vmin, vmax = np.min(frames), np.max(frames)
        label = f'Field Intensity (dB)'
    else:
        cmap = 'seismic'
        max_val = np.max(np.abs(frames))
        vmin, vmax = -max_val, max_val
        label = f'Field Amplitude'
        
    im = ax.imshow(frames[0], origin='lower', cmap=cmap, 
                   vmin=vmin, vmax=vmax, extent=extent)
    
    fig.colorbar(im, ax=ax, label=label)
    title = ax.set_title(f'{plane.upper()} Cross-Section of {component}-component\nTime Step = 0')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 5. Update Function for Animation
    def update(frame):
        im.set_array(frames[frame])
        title.set_text(f'{plane.upper()} Cross-Section of {component}-component\nTime Step = {frame}')
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=total_steps, interval=50, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani


def calculate_boundary_indices(nx, ny, nz):
    """
    Generates the 1D indices for PEC or PMC boundaries.
    as well as zeroes out E_tan on the outer walls of the 3D grid
    """
    print("Calculating PEC/PMC boundary indices...")
    N_total = nx * ny * nz
    boundary_list = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # We have this implemented by Carmelo
                n = get_node_index(i, j, k, nx, ny)

                # 1) X-normal walls
                # y_tan=0, z_tan=0
                if i == 0 or i == nx - 1:
                    boundary_list.append(n + N_total)      # y_tan offset
                    boundary_list.append(n + 2 * N_total)  # z_tan offset

                # 2) Y-normal walls
                # x_tan=0, z_tan=0
                if j == 0 or j == ny - 1:
                    boundary_list.append(n)                # x_tan offset (0)
                    boundary_list.append(n + 2 * N_total)  # z_tan offset

                # 3) Z-normal walls
                # x_tan=0, y_tan=0
                if k == 0 or k == nz - 1:
                    boundary_list.append(n)                # x_tan offset (0)
                    boundary_list.append(n + N_total)      # y_tan offset

    # Remove duplicates at corners and edges
    boundary_indices = np.unique(boundary_list)
    print(f"Boundary nodes mapped: {len(boundary_indices)}")

    return boundary_indices

####################
## MAIN function  ##
####################

def main():
    # grid parameters
    x_len, y_len, z_len = 1, 2, 1
    nx, ny, nz = 41, 2*41, 41

    # Define grid and topological matrices
    grid = FITGridEven(x_len, y_len, z_len, nx, ny, nz)
    C, S, C_hat, S_hat = topological_matrices(grid)

    print("\n--- Verification ---")
    print(f"Primary X points: {grid['G']['x']}")
    print(f"Dual X points:    {grid['G_hat']['x']}")
    print(f"Curl Matrix Shape: {grid['operators']['C'].shape}")
    print(f"Div Matrix Shape:  {grid['operators']['S'].shape}")

    ## Verification step from Weiland, 2001, (19): S*C=0
    verification_matrix = S @ C
    verification_matrix_dual =  C_hat @ S_hat 
    # Check how many non-zero elements exist
    non_zero_count = verification_matrix.nnz
    non_zero_count_dual = verification_matrix_dual.nnz
    # Print a clean report
    print(f"--- Topological Verification ---")
    print(f"Is Div * Curl = 0? {'YES' if non_zero_count == 0 else 'NO'}")
    print(f"Number of non-zero elements in product: {non_zero_count}")
    print(f"Is Div * Curl = 0? {'YES' if non_zero_count_dual == 0 else 'NO'}")
    print(f"Number of non-zero elements in product (dual): {non_zero_count_dual}")

    #=========================
    #   D_epsilon, D_mu 
    #=========================

    # Initialize the medium (vacuum)
    D_eps, D_mu = initialize_vacuum_medium(grid)
    
    print("\n--- Medium Initialization ---")
    print(f"D_epsilon initialized as {D_eps.shape[0]}x{D_eps.shape[1]} diagonal matrix.")
    print(f"D_mu initialized as {D_mu.shape[0]}x{D_mu.shape[1]} diagonal matrix.")
    
    # Example check: look at the first entry of the diagonal
    print(f"Vacuum scaling factor (epsilon side): {D_eps.diagonal()[0]:.4e}")

    # ============================================
    #   Testing the Global Numbering System   (get_node_index and get_ijk_of_node)
    # ============================================
    print("\n--- Global Numbering Test ---")
    
    # Use diemnsions of grid defined previously
    N_total = nx * ny * nz

    # we can insert the coordinates and find out the node number here
    i_test, j_test, k_test = 1,1,1
    
    # Get the local node number
    node_n = get_node_index(i_test, j_test, k_test, nx, ny)
    print(f"The 3D point ({i_test}, {j_test}, {k_test}) is node number: {node_n}")

    # Now we can reverse it back to check
    i_retrieved, j_retrieved, k_retrieved = get_ijk_of_node(node_n, nx, ny)
    print(f"Node number {node_n} corresponds to 3D point: ({i_retrieved}, {j_retrieved}, {k_retrieved})")

    # =======================
    #  Defining the source (at center for now, pointing in Z)
    # =======================

    # Coordinates of the center node
    center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
    # here we choose the orientation
    orientation = 'z'
    source_index = create_point_source(nx, ny, nz, center_x, center_y, center_z, orientation)

    print(f"\n--- Source Definition ---")
    print(f"Point source placed at ({center_x}, {center_y}, {center_z}) pointing in {orientation}.")  # indexing starts at 0, so this is the middle node in a 3x3x3 grid
    print(f"So the edge index is: {source_index}")

    # =========================================
    #  Preparation for the solver (leapfrog scheme) 
    # =========================================
    print("\n--- what is needed for leapfrog ---")
    
    # Size of the full field vectors (3 edges per node)
    N_vector = 3 * N_total
    
    # Initialize State Vectors (All zeros to start)
    e_initial = np.zeros(N_vector)
    h_initial = np.zeros(N_vector)
    b_initial = np.zeros(N_vector)
    d_initial = np.zeros(N_vector)
    
    # Universal constants
    eps0 = 8.8541878128e-12
    mu0 = 1.25663706212e-6
    c0 = 1.0 / np.sqrt(eps0 * mu0)  # Speed of light in vacuum
    
    # Extract grid spacings for CFL condition
    dx = grid["G"]["dx"]
    dy = grid["G"]["dy"]
    dz = grid["G"]["dz"]
    
    # Maximum allowed time step before simulation becomes unstable (COURANT)
    dt_limit = 1.0 / (c0 * np.sqrt((1/dx**2) + (1/dy**2) + (1/dz**2)))
    delta_t = 0.99 * dt_limit  # actual run-time, less than limit for stability
    
    n_steps = 300  # Total time iterations
    
    print(f"Speed of light: {c0:.2e} m/s")
    print(f"Calculated Time Step (based on Courant): {delta_t:.4e} s")
    print(f"Total physical simulation time: {n_steps * delta_t:.4e} s")

    # =============================
    #   Running the Simulation 
    # =============================
    print("\n--- Starting Solver ---")
    
    # run the solver
    e_history, h_history = leapfrog_scheme(
        e_initial=e_initial, 
        h_initial=h_initial, 
        b=b_initial, 
        d=d_initial, 
        C=C,          
        C_hat=C_hat,  
        M_eps=D_eps,  
        M_mu=D_mu,    
        j_source=1,  # we can replace this with j_source_vector if we want a time-varying source
        delta_t=delta_t, 
        n_steps=n_steps,
        frequency=1e9,
        source_index=source_index,
        #pec_indices=None,
        pec_indices=calculate_boundary_indices(nx, ny, nz),
        pmc_indices=None
    )
    
    print(f"\nSimulation Complete!")
    print(f"E-field history matrix shape: {e_history.shape}")

    # =============================
    #   Visualization
    # =============================
    print("\n--- Plotting Results ---")
    
    # Define a tiny floor value to prevent log10(0)
    epsilon = 1e-12
    
    # Convert fields to a 20*log10 dB scale using the absolute magnitude
    e_history_db = 20 * np.log10(np.abs(e_history) + epsilon)
    h_history_db = 20 * np.log10(np.abs(h_history) + epsilon)

    # Plot E_z in the XZ plane (slicing through the center Y-coordinate)
    ani_e = animate_slice(e_history_db, grid, plane='yz', slice_index=center_x, component='x', is_db=True)

if __name__ == "__main__":
    main()