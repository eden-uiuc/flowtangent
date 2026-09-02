import subprocess
import concurrent.futures
import uuid
import os
import itertools
import re
import time
import numpy as np
import queue
import shutil

from pathlib import Path

display_queue = queue.Queue()
for i in range (100, 148):
    display_queue.put(i)

def run_xfoil_benchmark(airfoil, reynolds, flap_angle):
    uid = uuid.uuid4().hex[:8]
    temp_polar = f"temp_{uid}.pol"
    
    # Building the command string as a list guarantees exact newline placement
    cmds = [
        f"NACA {airfoil}",
        "PANE",             # Inital panelization
        "GDES", 
        "FLAP", 
        "0.8 999",          # Prompt 1: Flap hinge x: 80%
        "0.5"               # Prompt 2: Flap hinge y: 50%
        f"{flap_angle}",    # Prompt 3: Deflection angle
        "CADD",             # Add corner points
        "",                 # Accept default corner angle
        "",                 # Accept default spline parameter
        "",                 # Accept refinement limits
        "EXEC",             # Apply changes to buffer
        "",                 # ENTER: Exit GDES back to top level
        "PANE",             # Repanel
        "BEND",             # Calculate structural properties
        "OPER",             # Enter OPER menu
        "ITER 100", 
        f"VISC {reynolds}"  ,
        "PACC", 
        f"{temp_polar}",    # Polar save file
        "",                 # Skip dump file prompt
        "ASEQ -5 15 0.25",
        "PACC",             # Close polar file
        "",
        "QUIT"
    ]
    
    xfoil_cmds = "\n".join(cmds) + "\n"
    display_port = display_queue.get()

    lock_file = f"/tmp/.X{display_port}-lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    xvfb_proc = None
    stdout = ""
    
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
        
        area_match = re.search(r'Area\s*=\s*([0-9.]+)', stdout)
        ixx_match = re.search(r'Ixx\s*=\s*([0-9.]+)', stdout)
        area = float(area_match.group(1)) if area_match else None
        ixx = float(ixx_match.group(1)) if ixx_match else None
        
        data_points = 0

        if os.path.exists(temp_polar):
            # Optional: Parse data into a DataFrame here if you still want to
            
            # Define the complex absolute path
            polar_file = polar_base / airfoil / f"{uid}.pol"
            
            # Move and rename the file
            shutil.move(temp_polar, polar_file)

        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                lines = f.readlines()
                
            start_idx = next((i for i, line in enumerate(lines) if '------' in line), None)
            if start_idx is not None:
                data_points = len([line for line in lines[start_idx + 1:] if line.strip()])
                        
            os.remove(polar_file)
            
        if data_points == 0:
             return 0, f"No data points. Stdout:\n{stdout[-300:]}"
             
        return data_points, None

    except Exception as e:
        if os.path.exists(polar_file):
            os.remove(polar_file)
        return 0, f"Exception: {str(e)}\nStdout snippet:\n{stdout[-300:]}"
        
    finally:
        # 6. Always kill the virtual monitor and return the port to the queue
        if xvfb_proc:
            xvfb_proc.terminate()
            xvfb_proc.wait()
        display_queue.put(display_port)

    

if __name__ == '__main__':
    airfoils = ['0012', 
                # '2412', '4412', '0009', '2415', '4415', '6409', '0015', '23012', '23015'
                ]
    flap_angles = np.linspace(-10, 15, 10).round(1)
    reynolds_nums = np.geomspace(50000, 3000000, 20).astype(int)

    polar_base = Path(__file__).resolve().parent / "polars"
    for airfoil in airfoils:
        (polar_base / airfoil).mkdir(parents=True, exist_ok=True)
    
    tasks = list(itertools.product(airfoils, reynolds_nums, flap_angles))
    total_tasks = len(tasks)
    
    print(f"Starting {total_tasks} runs on {os.cpu_count()} threads...")
    start_time = time.time()
    converged_points = 0
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_xfoil_benchmark, *task) for task in tasks]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            pts, err = future.result()
            converged_points += pts
            if err:
                errors.append(err)

    end_time = time.time()

    if errors:
        with open(Path(__file__).resolve().parent/"xfoil_errors.log", "w") as f:
            f.writelines(errors)
        print(f"\n[!] Logged {len(errors)} exceptions to xfoil_errors.log")
    
    # Calculate statistics
    benchmark_duration = end_time - start_time
    time_per_run = benchmark_duration / total_tasks
    
    # Extrapolate to 2,000 baseline airfoils
    target_airfoils = 2000
    multiplier = target_airfoils / len(airfoils)
    estimated_total_time_seconds = benchmark_duration * multiplier
    estimated_total_time_hours = estimated_total_time_seconds / 3600
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Benchmark duration:   {benchmark_duration:.2f} seconds")
    print(f"Avg time per run:     {time_per_run:.3f} seconds (for 1 Re/Flap/Airfoil sweep)")
    print(f"Total converged pts:  {converged_points} (out of {total_tasks * 80} attempted)")
    print(f"\nESTIMATE FOR {target_airfoils} AIRFOILS:")
    print(f"Total runs needed:    {target_airfoils * 10 * 20:,}")
    print(f"Estimated time:       {estimated_total_time_hours:.2f} hours (approx {estimated_total_time_hours/24:.1f} days)")
    print("="*40)