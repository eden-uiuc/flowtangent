import subprocess
import concurrent.futures
import uuid
import os
import time
import numpy as np
import queue
import shutil
import zarr

from pathlib import Path

from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.stats.qmc import Sobol

display_queue = queue.Queue()
for i in range (100, 148):
    display_queue.put(i)

BOUNDS = np.array([
    # Flap Angle, Flap Hinge, Reynolds, Mach
    [-10, 0.6, np.log10(50000), 0.0],      # Lower
    [15, 0.95, np.log10(3_000_000), 0.6]    # Upper
])

ALPHAS = np.concatenate([
    np.arange(-5, 8, 1.0), # Sparse linear region
    np.arange(8, 15.25, 0.25) # Dense stall region
])

N_ALPHAS = len(ALPHAS)
N_PANELS = 256

BEND_KEYS = [
    "Area", "Slen",
    "solid_X_Xc", "solid_X_max X-Xc", "solid_X_min X-Xc", "solid_X_Iyy", "solid_X_Iyy/(X-Xc)",
    "solid_Y_Yc", "solid_Y_max Y-Yc", "solid_Y_min Y-Yc", "solid_Y_Ixx", "solid_Y_Ixx/(Y-Yc)",
    "J",
    "skin_X_Xc", "skin_X_max X-Xc", "skin_X_min X-Xc", "skin_X_Iyy/t", "skin_X_Iyy/t(X-Xc)",
    "skin_Y_Yc", "skin_Y_max Y-Yc", "skin_Y_min Y-Yc", "skin_Y_Ixx/t", "skin_Y_Ixx/t(Y-Yc)",
    "J/t"
]

N_BEND_FEATURES = len(BEND_KEYS)

def generate_sobol_samples(n_samples):
    sampler = Sobol(4) # Flap Angle, Flap Hinge, Reynolds, Mach
    raw_samples = sampler.random_base2(m=int(np.log2(n_samples)))  # e.g. m=7 -> 128 samples

    scaled = BOUNDS[0] + raw_samples * (BOUNDS[1] - BOUNDS[0])

    scaled[:, 2] = 10 ** scaled[:, 2] # Logarithmic Reynolds space
    return scaled

def parse_bend_stdout(stdout_text):
    bend_data = {}
    
    # Check if BEND actually executed successfully in this run
    if "Area =" not in stdout_text:
        return bend_data
        
    mode = "general"
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Track which subsection of the BEND output we are currently reading
        if 'X-bending parameters(solid)' in line:
            mode = 'solid_X'
        elif 'Y-bending parameters(solid)' in line:
            mode = 'solid_Y'
        elif 'X-bending parameters(skin)' in line:
            mode = 'skin_X'
        elif 'Y-bending parameters(skin)' in line:
            mode = 'skin_Y'
        elif '=' in line:
            # Split "max X-Xc =  0.577965" into key and value
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = parts[1].strip()
                try:
                    # Python's float() natively handles Fortran 'E-01' scientific notation
                    val = float(val_str)
                    
                    if mode == "general":
                        bend_data[key] = val
                    else:
                        # Append the mode prefix to prevent key collisions (e.g., solid_X_Xc vs skin_X_Xc)
                        bend_data[f"{mode}_{key}"] = val
                        
                except ValueError:
                    pass
                    
        # The BEND output always ends with J/t. Break to avoid parsing OPER garbage.
        if 'J/t' in line and '=' in line:
            break
            
    return bend_data

def run_xfoil_point(run_idx, airfoil, flap_angle, flap_hinge, reynolds, mach, zarr_root):
    
    uid = uuid.uuid4().hex[:8]

    # Use RAM disk for I/O
    ram_dir = Path(f"/dev/shm/xfoil_{uid}")
    ram_dir.mkdir(exist_ok=True)

    polar_file = ram_dir / "macro.pol"

    from_file = "." in airfoil
    
    if from_file:
        temp_dat = f"in_{uid}.dat"

        shutil.copy(airfoil, temp_dat)
        af_str = f"LOAD {temp_dat}"
    else:
        # Assume NACA
        af_str = f"NACA {airfoil}"
    
    # Building the command string as a list guarantees exact newline placement
    cmds = [
        af_str,                     # NACA or DAT File
        "PANE",                     # Inital panelization
        "GDES",                     # Geometry Design Routine
        "FLAP",                     # Flap Deflection
        f"{flap_hinge:.2f},"        # Flap hinge x
        "999",                      # Relative y position
        "0.5",                      # Flap hinge y: 50% to prevent breaks
        f"{flap_angle:.2f}",        # Deflection angle
        "CADD",                     # Add corner points
        "",                         # Accept default corner angle
        "",                         # Accept default spline parameter
        "",                         # Accept refinement limits
        "EXEC",                     # Apply changes to buffer
        "",                         # Exit GDES back to top level
        "PPAR",                     # Enter panel parameters
        "N",                        # Number of nodes
        f"{N_PANELS}",              # Number of panels
        "",                         # Exit N prompt
        "",                         # Exit PPAR menu
        "PANE",                     # Repanel
        "BEND",                     # Calculate structural properties
        "OPER",                     # Enter OPER menu
        "ITER 100",                 # 100 maximum iterations
        f"MACH {mach:.3f}",         # Set Mach number
        f"VISC {reynolds}",         # Set Reynolds Number
        "PACC",                     # Activate polar accumulation
        f"{polar_file}",            # Store polar file
        "",                         # Skip dump file
    ]

    for a in ALPHAS:
        cmds.extend([
            f"ALFA {a:.2f}",
            f"CPWR {ram_dir}/cp_{a:.2f}.txt",
            f"DUMP {ram_dir}/bl_{a:.2f}.txt"
        ])
    cmds.extend(["", "QUIT"])
    
    xfoil_cmds = "\n".join(cmds) + "\n"

    display_port = display_queue.get()

    lock_file = f"/tmp/.X{display_port}-lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    xvfb_proc = None
    
    try:
        # 3. Launch a private Xvfb server for this specific thread
        xvfb_proc = subprocess.Popen(
            ['Xvfb', f':{display_port}', '-screen', '0', '1024x768x16'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(0.1) # Give the display a fraction of a second to spin up
        
        # 4. Inject the specific display port into the environment
        env = os.environ.copy()
        env['DISPLAY'] = f':{display_port}'
        
        # 5. Call xfoil DIRECTLY. If it times out, the exact binary is killed.
        process = subprocess.run(
            ['xfoil'], 
            input=xfoil_cmds, 
            text=True, 
            capture_output=True,
            timeout=15,
            env=env
        )
        stdout = process.stdout

        panel_arr = np.zeros((N_ALPHAS, N_PANELS, 13)) #BL info plus Cp

        for i, a in enumerate(ALPHAS):
            cp_file = ram_dir / f"cp_{a:.2f}.txt"
            bl_file = ram_dir / f"bl_{a:.2f}.txt"

            if cp_file.exists() and bl_file.exists():
                try:
                    cp_data = np.loadtxt(cp_file, skiprows=1)
                    bl_data = np.loadtxt(cp_file, skiprows=1)

                    panel_arr[i, :, :-1] = bl_data[:N_PANELS, :]
                    panel_arr[i, :, -1] = cp_data[:N_PANELS, :]
                except Exception:
                    pass

        polar_arr = np.full((N_ALPHAS, 9), np.nan)

        if polar_file.exists():
            with open(polar_file, 'r') as f:
                lines = f.readlines()
                
            # Find the header separator
            start_idx = next((i for i, line in enumerate(lines) if '------' in line), None)
            
            if start_idx is not None:
                # Extract lines and filter out any empty trailing whitespace lines
                data_lines = [line for line in lines[start_idx + 1:] if line.strip()]
                
                # Only proceed if there is actually data
                if data_lines:
                    raw_polar = np.loadtxt(data_lines)
                    
                    # Ensure the array actually has elements (safeguard against weird XFOIL outputs)
                    if raw_polar.size > 0:
                        # Force 2D if only one run converged
                        if raw_polar.ndim == 1:
                            raw_polar = raw_polar.reshape(1, -1)
                        
                        for row in raw_polar:
                            converged_alpha = np.round(row[0], 2)
                            
                            # Find matching index in our master ALPHAS array
                            idx = np.where(np.isclose(np.round(ALPHAS, 2), converged_alpha))[0]
                            
                            if len(idx) > 0:
                                polar_arr[idx[0]] = row

        bend_dict = parse_bend_stdout(process.stdout)
        bend_vector = np.array([bend_dict.get(k, np.nan) for k in BEND_KEYS], dtype=np.float32)

        zarr_root['panel_data'][run_idx] = panel_arr
        zarr_root['conditions'][run_idx] = np.array([flap_angle, flap_hinge, reynolds, mach])
        zarr_root['polar_data'][run_idx] = polar_arr
        zarr_root['bend_data'][run_idx] = bend_vector
        
    finally:
        # 6. Always kill the virtual monitor and return the port to the queue
        if xvfb_proc:
            xvfb_proc.terminate()
            xvfb_proc.wait()
        display_queue.put(display_port)

        if ram_dir.exists():
            shutil.rmtree(ram_dir)

# ==========================================
# 1. Coordinate Alignment (Cosine Spacing)
# ==========================================
def align_airfoil(x_coords, y_coords, num_points=100):
    """
    Interpolates arbitrary airfoil coordinates onto a standardized cosine-spaced grid.
    Expects coordinates starting at trailing edge, over the top, to leading edge, 
    and back along the bottom to the trailing edge.
    """
    # Create the standard cosine-spaced x grid (clustered at LE and TE)
    beta = np.linspace(0, np.pi, num_points)
    x_standard = 0.5 * (1.0 - np.cos(beta))
    
    # Split airfoil into upper and lower surfaces based on the leading edge (min X)
    le_idx = np.argmin(x_coords)
    
    x_upper = x_coords[:le_idx+1][::-1] # Reverse to go LE -> TE
    y_upper = y_coords[:le_idx+1][::-1]
    
    x_lower = x_coords[le_idx:]
    y_lower = y_coords[le_idx:]
    
    # Interpolate using cubic splines
    f_upper = interp1d(x_upper, y_upper, kind='cubic', fill_value="extrapolate", assume_sorted=False)
    f_lower = interp1d(x_lower, y_lower, kind='cubic', fill_value="extrapolate", assume_sorted=False)
    
    y_upper_std = f_upper(x_standard)
    y_lower_std = f_lower(x_standard)
    
    # Flatten into a single 1D vector: [y_upper_0 ... y_upper_N, y_lower_0 ... y_lower_N]
    # (We drop x_standard because it is identical for every airfoil)
    return np.concatenate([y_upper_std, y_lower_std])

# ==========================================
# 2. Basis Concatenation & Gram-Schmidt
# ==========================================
def build_hybrid_basis(uiuc_data, naca_data, uiuc_dims=32, naca_dims=3):
    """
    uiuc_data: shape (N_uiuc, 200) - Standardized UIUC vectors
    naca_data: shape (N_naca, 200) - Standardized NACA vectors
    """
    print("Fitting independent PCAs...")
    pca_uiuc = PCA(n_components=uiuc_dims).fit(uiuc_data)
    pca_naca = PCA(n_components=naca_dims).fit(naca_data)
    
    # 1. Extract means and basis vectors
    mu_U = pca_uiuc.mean_
    mu_N = pca_naca.mean_
    
    V_U = pca_uiuc.components_  # Shape: (32, 200)
    V_N = pca_naca.components_  # Shape: (3, 200)
    
    # 2. Calculate the Mean Shift Vector
    delta_mu = mu_N - mu_U
    delta_mu /= np.linalg.norm(delta_mu) # Normalize for numerical stability
    
    # 3. Assemble the raw concatenated matrix
    # Order matters! We force the QR decomposition to prioritize the mean shift
    # and NACA variance before filling the rest of the space with UIUC variance.
    M_raw = np.vstack([
        delta_mu,      # 1 vector
        V_N,           # 3 vectors
        V_U            # 32 vectors
    ])
    
    # Transpose so vectors are columns (expected by np.linalg.qr)
    M_raw = M_raw.T 
    
    print("Running QR Decomposition...")
    # 4. Gram-Schmidt Orthogonalization
    Q, R = np.linalg.qr(M_raw)
    
    # Q is now our orthonormal basis. Shape: (200, 36)
    # Transpose back to scikit-learn standard format: (36, 200)
    hybrid_basis = Q.T
    
    # Optional: Truncate back to exactly 32 dimensions if you want to keep the 
    # latent space size strictly matched to your original UIUC estimate.
    hybrid_basis = hybrid_basis[:uiuc_dims, :]
    
    return mu_U, hybrid_basis

# ==========================================
# 3. Encoding / Decoding Helper
# ==========================================
def encode_airfoil(airfoil_vector, basis, origin):
    """Projects a 200D airfoil vector into the low-dimensional latent space."""
    return np.dot(airfoil_vector - origin, basis.T)

def decode_airfoil(latent_vector, basis, origin):
    """Reconstructs the 200D airfoil from the latent weights."""
    return np.dot(latent_vector, basis) + origin
    

if __name__ == '__main__':
    airfoils = ['0012', '2412'
                # '4412', '0009', '2415', '4415', '6409', '0015', '23012', '23015'
                ]

    N_SAMPLES = 8 # Power of 2 for Sobol
    TOTAL_RUNS = len(airfoils) * N_SAMPLES

    print(f"Initializing Zarr store for {TOTAL_RUNS} total condition sweeps...")

    file_dir = Path(__file__).resolve().parent
    root = zarr.group(file_dir / "data.zarr", overwrite=True)

    root.create_array("panel_data", shape=(TOTAL_RUNS, N_ALPHAS, N_PANELS, 13), chunks=(1, N_ALPHAS, N_PANELS, 13), dtype='f4')
    root.create_array("conditions", shape=(TOTAL_RUNS, 4), chunks=(1, 4), dtype='f4')
    root.create_array("polar_data", shape=(TOTAL_RUNS, N_ALPHAS, 9), chunks=(1, N_ALPHAS, 9), dtype='f4')
    root.create_array("bend_data", shape=(TOTAL_RUNS, N_BEND_FEATURES), chunks=(1, N_BEND_FEATURES), dtype='f4')
    
    tasks = []
    run_counter = 0
    for airfoil in airfoils:
        samples = generate_sobol_samples(N_SAMPLES)
        for sample in samples:
            flap_angle = float(sample[0])
            flap_hinge = float(sample[1])
            reynolds = float(sample[2])
            mach = float(sample[3])
            tasks.append((run_counter, airfoil, flap_angle, flap_hinge, reynolds, mach, root))
            run_counter += 1
    
    print(f"Starting {TOTAL_RUNS} runs on {os.cpu_count() - 4} threads...") #type: ignore
    start_time = time.time()
    # converged_points = 0
    # errors = []

    print("Running a single test point for debugging...")
    test_task = tasks[0] # (run_idx, airfoil, flap, reynolds, mach, root)
    run_xfoil_point(*test_task)
    print("Test point completed successfully!")

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 4) as executor: #type: ignore
        futures = [executor.submit(run_xfoil_point, *task) for task in tasks]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result() 
            except Exception as e:
                print(f"THREAD CRASHED - TASK {i}: {repr(e)}")
                raise e
                
            if i % 10 == 0:
                print(f"Progress: {i} / {TOTAL_RUNS} completed.")

    end_time = time.time()

    # if errors:
    #     with open(Path(__file__).resolve().parent/"xfoil_errors.log", "w") as f:
    #         f.writelines(errors)
    #     print(f"\n[!] Logged {len(errors)} exceptions to xfoil_errors.log")
    
    # Calculate statistics
    benchmark_duration = end_time - start_time
    time_per_run = benchmark_duration / TOTAL_RUNS
    
    # Extrapolate to 2,000 baseline airfoils
    target_airfoils = 2000
    target_samples = 128
    multiplier = (target_airfoils / len(airfoils)) * (target_samples / N_SAMPLES)
    estimated_total_time_seconds = benchmark_duration * multiplier
    estimated_total_time_hours = estimated_total_time_seconds / 3600
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Benchmark duration:   {benchmark_duration:.2f} seconds")
    print(f"Avg time per run:     {time_per_run:.3f} seconds")
    print(f"\nESTIMATE FOR {target_airfoils} AIRFOILS:")
    print(f"Total runs needed:    {multiplier * TOTAL_RUNS:,}")
    print(f"Estimated time:       {estimated_total_time_hours:.2f} hours (approx {estimated_total_time_hours/24:.1f} days)")
    print("="*40)