import subprocess
import concurrent.futures
import uuid
import os
import time
import numpy as np
import pandas as pd
import queue
import shutil
import zarr
import csv

from datetime import datetime
from pathlib import Path
from scipy.stats.qmc import Sobol

import plotly.graph_objects as go
from plotly.subplots import make_subplots

display_queue = queue.Queue()
for i in range (100, 148):
    display_queue.put(i)

xvfb_processes = []

def setup_display_pool(num_displays):
    print(f"Pre-spawning {num_displays} Xvfb servers to avoid contention...")
    for i in range(num_displays):
        port = 100 + i
        for f in [f"/tmp/.X{port}-lock", f"/tmp/.X11-unix/X{port}"]:
            if os.path.exists(f):
                try: os.remove(f)
                except OSError: pass
        
        proc = subprocess.Popen(
            ['Xvfb', f':{port}', '-screen', '0', '1024x768x16', '-ac'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        xvfb_processes.append(proc)
        display_queue.put(port)
        
    time.sleep(2) # Kernel stabilization 
    print("Display pool ready.")

def cleanup_display_pool():
    print("Cleaning up display pool...")
    for proc in xvfb_processes:
        proc.terminate()
        proc.wait()
    subprocess.run(['killall', 'Xvfb'], stderr=subprocess.DEVNULL)

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

FILE_DIR = Path(__file__).resolve().parent

N_BEND_FEATURES = len(BEND_KEYS)
LOG_FILE = FILE_DIR / "run_log.csv"
ERROR_LOG_FILE = FILE_DIR / "error_log.csv"

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
        f"{flap_hinge:.2f}",        # Flap hinge x
        "999",                      # Relative y position
        "0.5",                      # Flap hinge y: 50% to prevent breaks
        f"{flap_angle:.2f}",        # Deflection angle
        "CADD",                     # Add corner points
        "",                         # Accept default corner angle
        "",                         # Accept default spline parameter
        "",                         # Accept refinement limits
        "EXEC",                     # Apply changes to buffer
        "",                         # Exit GDES back to top level
        # "PPAR",                     # Enter panel parameters
        # "N",                        # Number of nodes
        # f"{N_PANELS}",              # Number of panels
        # "",                         # Exit N prompt
        # "",                         # Exit PPAR menu
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
    env = os.environ.copy()
    env['DISPLAY'] = f':{display_port}'



    lock_file = f"/tmp/.X{display_port}-lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    converged_alphas = 0
    
    try:
        xfoil_log_path = ram_dir / "xfoil_stdout.txt"
        debug_dump_path = FILE_DIR / f"errors/timeout_{airfoil_name}_{uid}.log"
        
        # 5. Call xfoil DIRECTLY. If it times out, the exact binary is killed.
        with open(xfoil_log_path, "w") as out_file:
            try:
                process = subprocess.Popen(
                    ['xfoil'],
                    stdin=subprocess.PIPE,
                    text=True, 
                    stdout=out_file,
                    stderr=subprocess.STDOUT,
                    env=env
                )

                if process.stdin:
                    process.stdin.write(xfoil_cmds)
                    process.stdin.flush()
                    process.stdin.close()

                last_size = 0
                start_time = time.time()
                output_time = time.time()

                while process.poll() is None:
                    time.sleep(0.5)
                    try:
                        current_size = os.path.getsize(xfoil_log_path)
                    except OSError:
                        current_size = 0

                    if current_size > last_size:
                        last_size = current_size
                        output_time = time.time()
                    elif (time.time() - output_time) > 15:
                        process.terminate()
                        shutil.copy(xfoil_log_path, debug_dump_path)
                        raise RuntimeError("No new terminal output for 15 seconds.")
                    
                    if (time.time() - start_time) > 1200:
                        process.terminate()
                        shutil.copy(xfoil_log_path, debug_dump_path)
                        raise RuntimeError("Exceeded 20-minute absolute maximum limit.")

            except Exception as e:
                raise RuntimeError(f"Subprocess failed: {str(e)}")

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
        
        with open(xfoil_log_path, "r") as f:
            stdout_text = f.read()

        bend_dict = parse_bend_stdout(stdout_text)
        bend_vector = np.array([bend_dict.get(k, np.nan) for k in BEND_KEYS], dtype=np.float32)

        zarr_root['panel_data'][run_idx] = panel_arr
        zarr_root['conditions'][run_idx] = np.array([flap_angle, flap_hinge, reynolds, mach])
        zarr_root['polar_data'][run_idx] = polar_arr
        zarr_root['bend_data'][run_idx] = bend_vector
        zarr_root['foil_name'][run_idx] = airfoil_name
    
    finally:
        display_queue.put(display_port) # Instantly return for the next thread
        os.remove(temp_dat)
        if ram_dir.exists():
            shutil.rmtree(ram_dir)

    return run_idx, airfoil_name, flap_angle, flap_hinge, reynolds, mach, converged_alphas

import json

def save_valid_airfoils(kept_dict, filepath="train_set.json"):
    """Saves the pruned airfoil names to a JSON file to bypass the PCA pipeline."""
    names = list(kept_dict.keys())
    with open(filepath, 'w') as f:
        json.dump(names, f, indent=4)
    print(f"Saved {len(names)} airfoil names to {filepath}")

def load_valid_airfoils(filepath="train_set.json"):
    """Loads the valid airfoil names directly into a list."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Could not find {filepath}. Run the pruner once to generate it.")
    with open(filepath, 'r') as f:
        names = json.load(f)
    print(f"Loaded {len(names)} airfoil names from {filepath}")
    return names

def generate_data():
    from flowtangent.components.airfoils._data import validate_library, regularize_design_space, _AF_REGISTRY

    try:
        train_set = set(load_valid_airfoils(str(FILE_DIR / "train_set.json")))
    except:
        valid_airfoils = validate_library(k=16)
        train_set, _ = regularize_design_space(valid_airfoils, k=16)

        save_valid_airfoils(train_set, filepath = str(FILE_DIR / "train_set.json"))

    # train_set = ["goe184", "naca001064", "rhodesg32"]

    airfoils = sorted([_AF_REGISTRY[name] for name in train_set])

    N_SAMPLES = 128 # Power of 2 for Sobol
    TOTAL_RUNS = len(airfoils) * N_SAMPLES
    
    zarr_path = FILE_DIR / "data.zarr"

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

    n_threads = os.cpu_count() - 4
    setup_display_pool(n_threads)
    
    print(f"Starting {TOTAL_RUNS} runs on {n_threads} threads...") #type: ignore

    total_converged = 0
    total_attempted = 0
    total_crashed = 0

    # print("Running a single test point for debugging...")
    # test_task = tasks[0] # (run_idx, airfoil, flap, reynolds, mach, root)
    # run_xfoil_point(*test_task)
    # print("Test point completed successfully!")

    try:
        with open(LOG_FILE, 'a', newline='') as success_f, open(ERROR_LOG_FILE, 'a', newline='') as error_f:
            success_writer = csv.writer(success_f)
            error_writer = csv.writer(error_f)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
                future_to_task = {executor.submit(run_xfoil_point, *task): task for task in tasks}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):

                    total_attempted += N_ALPHAS

                    task = future_to_task[future]
                    t_idx, t_airfoil, t_f_ang, t_f_hinge, t_re, t_mach, _ = task
                    t_af_name = Path(t_airfoil).stem

                    try:
                        # If it succeeds, unpack the exact results
                        run_idx, af_name, f_ang, f_hinge, re, mach, conv_alphas = future.result() 
                        
                        total_converged += conv_alphas
                        
                        success_writer.writerow([run_idx, af_name, f"{f_ang:.2f}", f"{f_hinge:.2f}", f"{re:.0f}", f"{mach:.3f}", conv_alphas, N_ALPHAS])
                        success_f.flush()
                        
                    except Exception as e:
                        # If it crashes, log the exact inputs from the task dictionary
                        error_msg = repr(e)
                        
                        error_writer.writerow([t_idx, t_af_name, f"{t_f_ang:.2f}", f"{t_f_hinge:.2f}", f"{t_re:.0f}", f"{t_mach:.3f}", error_msg])
                        error_f.flush()

                        total_crashed += N_ALPHAS
                        
                    if i % 100 == 0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        yield_pct = (total_converged / total_attempted) * 100 if total_attempted > 0 else 0
                        print(f"Progress: {i} / {len(tasks)} ({1/len(tasks) * 100:.1f}%)| Yield: {yield_pct:.1f}% | Recent: {t_af_name} | Time: {timestamp}")
    finally:
        cleanup_display_pool()
        subprocess.run(['killall', 'xfoil'], stderr=subprocess.DEVNULL)

    return root

def yield_dashboard(zarr_path: Path | str="data.zarr"):
    print("Loading Zarr store...")
    root = zarr.open_group(zarr_path, mode='r')
    
    conditions = root['conditions'][:]
    polar_data = root['polar_data'][:]
    
    # Calculate yield per run
    converged_alphas = np.sum(~np.isnan(polar_data[:, :, 0]), axis=1)
    total_alphas = polar_data.shape[1]
    yield_pct = (converged_alphas / total_alphas) * 100
    
    failed_mask = yield_pct == 0
    success_mask = yield_pct > 0
    overall_yield = np.sum(converged_alphas) / (polar_data.shape[0] * total_alphas) * 100
    print(f"Overall Dataset Yield: {overall_yield:.2f}%")
    
    # Helper to generate trace data and hover tooltips
    def create_trace_data(mask):
        m_mach = conditions[mask, 3]
        m_re = conditions[mask, 2]
        m_flap = conditions[mask, 0]
        m_yield = yield_pct[mask]
        
        hover_text = [
            f"<b>Yield: {y:.1f}%</b><br>"
            f"Mach: {m:.3f}<br>"
            f"Reynolds: {r:.1e}<br>"
            f"Flap: {f:.1f}°"
            for y, m, r, f in zip(m_yield, m_mach, m_re, m_flap)
        ]
        return m_mach, m_re, m_flap, m_yield, hover_text

    # Extract masked data
    f_mach, f_re, f_flap, _, f_text = create_trace_data(failed_mask)
    s_mach, s_re, s_flap, s_yield, s_text = create_trace_data(success_mask)

    # Base marker styling
    marker_failed = dict(symbol='x', color='red', size=6, opacity=0.5)
    marker_success = dict(
        color=s_yield, colorscale='Viridis', cmin=0, cmax=100,
        size=7, opacity=0.8, showscale=False # Enabled only on the last plot
    )
    
    marker_success_with_cbar = marker_success.copy()
    marker_success_with_cbar['showscale'] = True
    marker_success_with_cbar['colorbar'] = dict(title="Yield (%)", thickness=15, x=1.02)

    # Build the 1x3 dashboard
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Mach vs Flap", "Reynolds vs Flap", "Mach vs Reynolds"),
        horizontal_spacing=0.08
    )

    # Panel 1: Mach vs Flap Angle
    fig.add_trace(go.Scatter(x=f_mach, y=f_flap, mode='markers', marker=marker_failed, text=f_text, hoverinfo='text', name="Failed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_mach, y=s_flap, mode='markers', marker=marker_success, text=s_text, hoverinfo='text', name="Success"), row=1, col=1)

    # Panel 2: Reynolds vs Flap Angle
    fig.add_trace(go.Scatter(x=f_re, y=f_flap, mode='markers', marker=marker_failed, text=f_text, hoverinfo='text', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=s_re, y=s_flap, mode='markers', marker=marker_success, text=s_text, hoverinfo='text', showlegend=False), row=1, col=2)

    # Panel 3: Mach vs Reynolds
    fig.add_trace(go.Scatter(x=f_mach, y=f_re, mode='markers', marker=marker_failed, text=f_text, hoverinfo='text', showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=s_mach, y=s_re, mode='markers', marker=marker_success_with_cbar, text=s_text, hoverinfo='text', showlegend=False), row=1, col=3)

    # Configure axes formatting
    fig.update_xaxes(title_text="Mach Number", row=1, col=1)
    fig.update_yaxes(title_text="Flap Angle (deg)", row=1, col=1)
    
    fig.update_xaxes(type="log", title_text="Reynolds Number", row=1, col=2)
    fig.update_yaxes(title_text="Flap Angle (deg)", row=1, col=2)
    
    fig.update_xaxes(title_text="Mach Number", row=1, col=3)
    fig.update_yaxes(type="log", title_text="Reynolds Number", row=1, col=3)

    fig.update_layout(
        title_text=f"<b>XFOIL Convergence Boundary Diagnostics</b> (Overall Yield: {overall_yield:.1f}%)",
        height=500,
        width=1400,
        template="plotly_white",
        hovermode="closest",
        showlegend=False
    )
    
    fig.show()

def purge_airfoil_from_zarr(airfoil_name="goe513", zarr_path=FILE_DIR / "data.zarr", log_path=FILE_DIR / "run_log.csv"):
    print(f"Purging {airfoil_name}...")
    
    root = zarr.open_group(zarr_path, mode='a')
    names = root['foil_name'][:]
    
    bad_idx = np.where(names == airfoil_name)[0]
    if len(bad_idx) == 0:
        print("Airfoil not found in Zarr array.")
        return

    print(f"Found {len(bad_idx)} rows. Overwriting with NaNs...")

    # A direct loop safely handles the N-dimensional chunk writes
    for idx in bad_idx:
        root['polar_data'][idx] = np.nan
        root['panel_data'][idx] = np.nan
        root['bend_data'][idx] = np.nan
        root['foil_name'][idx] = ""
    
    print("Zarr arrays cleared.")

    # 2. Remove from the checkpoint log
    df = pd.read_csv(log_path)
    initial_len = len(df)
    df_clean = df[~df['run_idx'].isin(bad_idx)]
    df_clean.to_csv(log_path, index=False)
    
    print(f"Removed {initial_len - len(df_clean)} entries from {log_path}.")

def test_nominal_inviscid(airfoil_path):
    af_name = Path(airfoil_path).stem
    uid = uuid.uuid4().hex[:8]

    temp_dat = f"{af_name}_{uid}.dat"
    shutil.copy(airfoil_path, temp_dat)

    xfoil_log = FILE_DIR / f"logs/{af_name}_{uid}.txt"
    cp_log = f"cp_{af_name}_{uid}.txt"
    
    cmds = [
        f"LOAD {temp_dat}",
        "PANE",
        "OPER",
        "ALFA 0",
        f"CPWR {cp_log}",
        "",
        "QUIT"
    ]
    
    xfoil_cmds = "\n".join(cmds) + "\n"
    solved = False
    
    # 1. Grab a pre-warmed display from the queue
    display_port = display_queue.get()
    env = os.environ.copy()
    env['DISPLAY'] = f':{display_port}'

    with open(xfoil_log, "w") as out_file:
        try:
            subprocess.run(
                ['xfoil'], 
                input=xfoil_cmds, 
                text=True, 
                stdout=out_file,
                stderr=subprocess.STDOUT,
                timeout=15, 
                env=env
            )
        except Exception:
            pass
            
    # 2. Return the display instantly for the next thread
    display_queue.put(display_port) 
    
    # Validation and cleanup
    os.remove(temp_dat)
    if Path(cp_log).exists():
        solved = True
        os.remove(cp_log)
    if solved and Path(xfoil_log).exists():
        os.remove(xfoil_log)
    
    return af_name, solved

if __name__ == "__main__":
    generate_data()