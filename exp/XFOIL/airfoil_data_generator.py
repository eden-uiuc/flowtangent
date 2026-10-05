import subprocess
import concurrent.futures
import uuid
import os
import time
import numpy as np
import queue
import shutil
import zarr
import csv

from pathlib import Path
from scipy.stats.qmc import Sobol
from flowtangent.utils.io import _ft_root

display_queue = queue.Queue()
for i in range (100, 148):
    display_queue.put(i)

BOUNDS = np.array([
    # Flap Angle, Flap Hinge, Reynolds, Mach
    [-10, 0.6, np.log10(50000), 0.0],      # Lower
    [15, 0.95, np.log10(3_000_000), 0.6]    # Upper
])

# Start at 0, go up sparsely, then densely into the stall regime
ALPHAS_POS = np.concatenate([
    np.arange(0, 8, 1.0),       # [0, 1, 2 ... 7]
    np.arange(8, 15.25, 0.25)   # [8, 8.25 ... 15.0]
])

# Start just below 0, go down sparsely
ALPHAS_NEG = np.arange(-1, -6, -1.0) # [-1, -2, -3, -4, -5]

# The combined array for Zarr sizing and index matching
ALPHAS = np.concatenate([ALPHAS_POS, ALPHAS_NEG])
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
LOG_FILE = Path(__file__).resolve().parent / "run_log.csv"
ERROR_LOG_FILE = Path(__file__).resolve().parent / "error_log.csv"

def generate_sobol_samples(total_samples):
    sampler = Sobol(4) # Flap Angle, Flap Hinge, Reynolds, Mach
    m = int(np.ceil(np.log2(total_samples)))
    raw_samples = sampler.random_base2(m=m)[:total_samples]  # e.g. m=7 -> 128 samples, truncated to target

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
    
    airfoil_name = Path(airfoil).stem
    uid = uuid.uuid4().hex[:8]

    # Use RAM disk for I/O
    ram_dir = Path(f"/dev/shm/xfoil_{uid}")
    ram_dir.mkdir(exist_ok=True)
    polar_file = ram_dir / "macro.pol"

    # Store 
    temp_dat = f"in_{uid}.dat"
    shutil.copy(airfoil, temp_dat)

    
    # Building the command string as a list guarantees exact newline placement
    cmds = [
        f"LOAD {temp_dat}",         # Load DAT File
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
        "ITER 200",                 # 200 maximum iterations
        f"MACH {mach:.3f}",         # Set Mach number
        f"VISC {reynolds}",         # Set Reynolds Number
        "PACC",                     # Activate polar accumulation
        f"{polar_file}",            # Store polar file
        "",                         # Skip dump file
    ]

    # 1. Sweep the Positive Alphas (Cold start at 0 is safe)
    for a in ALPHAS_POS:
        cmds.extend([
            f"ALFA {a:.2f}",
            f"CPWR {ram_dir}/cp_{a:.2f}.txt",
            f"DUMP {ram_dir}/bl_{a:.2f}.txt"
        ])
        
    # 2. WIPE THE BOUNDARY LAYER MEMORY
    # Without this, XFOIL uses the separated 15-degree wake to guess the -1 degree flow.
    cmds.append("INIT")
    
    # 3. Sweep the Negative Alphas
    for a in ALPHAS_NEG:
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
    converged_alphas = 0
    
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
                    bl_data = np.loadtxt(bl_file, skiprows=1)

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
                                converged_alphas += 1

        bend_dict = parse_bend_stdout(process.stdout)
        bend_vector = np.array([bend_dict.get(k, np.nan) for k in BEND_KEYS], dtype=np.float32)

        zarr_root['panel_data'][run_idx] = panel_arr
        zarr_root['conditions'][run_idx] = np.array([flap_angle, flap_hinge, reynolds, mach])
        zarr_root['polar_data'][run_idx] = polar_arr
        zarr_root['bend_data'][run_idx] = bend_vector
        zarr_root['foil_name'][run_idx] = airfoil_name
        
    finally:
        # 6. Always kill the virtual monitor and return the port to the queue
        if xvfb_proc:
            xvfb_proc.terminate()
            xvfb_proc.wait()
        display_queue.put(display_port)

        if ram_dir.exists():
            shutil.rmtree(ram_dir)

    return run_idx, airfoil_name, flap_angle, flap_hinge, reynolds, mach, converged_alphas
    

if __name__ == '__main__':
    # airfoils = list(Path(_ft_root() / "data" / "airfoils").glob('*.dat'))

    from flowtangent.components.airfoils._data import validate_library, analyze_split_safety, _AF_REGISTRY
    valid_airfoils = validate_library(k=16)
    train_set, test_set = analyze_split_safety(valid_airfoils, k=16)

    airfoils = [_AF_REGISTRY[name] for name in train_set][:10]

    N_SAMPLES = 128 # Power of 2 for Sobol
    TOTAL_RUNS = len(airfoils) * N_SAMPLES

    file_dir = Path(__file__).resolve().parent
    zarr_path = file_dir / "data.zarr"

    completed_runs = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row: 
                    completed_runs.add(int(row[0]))
                    
    is_resuming = len(completed_runs) > 0
    
    print(f"Initializing Zarr store for {TOTAL_RUNS} total condition sweeps...")
    root = zarr.open_group(zarr_path, mode='a' if is_resuming else 'w')

    if not is_resuming:
        root.create_array("panel_data", shape=(TOTAL_RUNS, N_ALPHAS, N_PANELS, 13), chunks=(1, N_ALPHAS, N_PANELS, 13), dtype='f4')
        root.create_array("conditions", shape=(TOTAL_RUNS, 4), chunks=(1, 4), dtype='f4')
        root.create_array("polar_data", shape=(TOTAL_RUNS, N_ALPHAS, 9), chunks=(1, N_ALPHAS, 9), dtype='f4')
        root.create_array("bend_data", shape=(TOTAL_RUNS, N_BEND_FEATURES), chunks=(1, N_BEND_FEATURES), dtype='f4')
        root.create_array("foil_name", shape=(TOTAL_RUNS,), chunks=(1000,), dtype='U50') # Fixed-length Unicode

        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_idx', 'airfoil', 'flap_angle', 'flap_hinge', 'reynolds', 'mach', 'converged_alphas', 'total_alphas'])
        with open(ERROR_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_idx', 'airfoil', 'flap_angle', 'flap_hinge', 'reynolds', 'mach', 'error_message'])
    else:
        print(f"RESUMING: Found {len(completed_runs)} completed runs in log. Skipping...")

    sobol_samples = generate_sobol_samples(TOTAL_RUNS)
    
    tasks = []
    run_counter = 0
    for airfoil in airfoils:
        for _ in range(N_SAMPLES):
            if run_counter not in completed_runs:
                sample = sobol_samples[run_counter]
                tasks.append((
                    run_counter, airfoil, 
                    float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3]), 
                    root
                ))
            run_counter += 1
    
    print(f"Starting {TOTAL_RUNS} runs on {os.cpu_count() - 4} threads...") #type: ignore

    total_converged = 0
    total_attempted = 0

    with open(LOG_FILE, 'a', newline='') as success_f, open(ERROR_LOG_FILE, 'a', newline='') as error_f:
        success_writer = csv.writer(success_f)
        error_writer = csv.writer(error_f)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
            future_to_task = {executor.submit(run_xfoil_point, *task): task for task in tasks}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):

                task = future_to_task[future]
                t_idx, t_airfoil, t_f_ang, t_f_hinge, t_re, t_mach, _ = task
                t_af_name = Path(t_airfoil).stem

                try:
                    # If it succeeds, unpack the exact results
                    run_idx, af_name, f_ang, f_hinge, re, mach, conv_alphas = future.result() 
                    
                    total_converged += conv_alphas
                    total_attempted += N_ALPHAS
                    
                    success_writer.writerow([run_idx, af_name, f"{f_ang:.2f}", f"{f_hinge:.2f}", f"{re:.0f}", f"{mach:.3f}", conv_alphas, N_ALPHAS])
                    success_f.flush()
                    
                except Exception as e:
                    # If it crashes, log the exact inputs from the task dictionary
                    error_msg = repr(e)
                    print(f"THREAD CRASHED [{t_af_name} | idx: {t_idx}]: {error_msg}")
                    
                    error_writer.writerow([t_idx, t_af_name, f"{t_f_ang:.2f}", f"{t_f_hinge:.2f}", f"{t_re:.0f}", f"{t_mach:.3f}", error_msg])
                    error_f.flush()
                    
                if i % 20 == 0:
                    yield_pct = (total_converged / total_attempted) * 100 if total_attempted > 0 else 0
                    print(f"Progress: {i} / {len(tasks)} | Yield: {yield_pct:.1f}% | Recent: {t_af_name}")


    
    # # Calculate statistics
    # benchmark_duration = end_time - start_time
    # time_per_run = benchmark_duration / TOTAL_RUNS
    
    # # Extrapolate to 2,000 baseline airfoils
    # target_airfoils = 2000
    # target_samples = 128
    # multiplier = (target_airfoils / len(airfoils)) * (target_samples / N_SAMPLES)
    # estimated_total_time_seconds = benchmark_duration * multiplier
    # estimated_total_time_hours = estimated_total_time_seconds / 3600
    
    # print("\n" + "="*40)
    # print("BENCHMARK RESULTS")
    # print("="*40)
    # print(f"Benchmark duration:   {benchmark_duration:.2f} seconds")
    # print(f"Avg time per run:     {time_per_run:.3f} seconds")
    # print(f"\nESTIMATE FOR {target_airfoils} AIRFOILS:")
    # print(f"Total runs needed:    {multiplier * TOTAL_RUNS:,}")
    # print(f"Estimated time:       {estimated_total_time_hours:.2f} hours (approx {estimated_total_time_hours/24:.1f} days)")
    # print("="*40)