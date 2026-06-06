import os
import zarr
import glob

import numpy as np
import pandas as pd
import numexpr as ne
import streamlit as st

import dask.array as da
import plotly.express as px
import plotly.graph_objects as go


#-----------------------------------------------------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------------------------------------------------

def filter_widget(
    label: str, 
    min_val: float, 
    max_val: float, 
    default_val: float, 
    key_prefix: str, 
    step: float = None, 
    allow_exact_toggle: bool = True
):
    # Dynamically allocate columns based on whether we allow the Exact toggle
    if allow_exact_toggle:
        col_lbl, col_t1, col_t2 = st.columns([0.4, 0.3, 0.3])
        with col_t1:
            is_exact = st.toggle("Exact", key=f"ex_{key_prefix}")
        with col_t2:
            is_manual = st.toggle("Manual", key=f"man_{key_prefix}")
    else:
        col_lbl, col_t2 = st.columns([0.7, 0.3])
        is_exact = False # Force Range mode
        with col_t2:
            is_manual = st.toggle("Manual", key=f"man_{key_prefix}")

    with col_lbl:
        st.markdown(f"**{label}**")

    min_val, max_val = float(min_val), float(max_val)

    if is_exact:
        # --- EXACT MATCH MODE ---
        if is_manual:
            val = st.number_input(
                "Target", min_value=min_val, max_value=max_val, value=float(default_val),
                step=step, key=f"num_{key_prefix}", label_visibility="collapsed"
            )
        else:
            val = st.slider(
                "Target", min_value=min_val, max_value=max_val, value=float(default_val),
                step=step, key=f"sli_{key_prefix}", label_visibility="collapsed"
            )
        tol = 1e-4
        return (val - tol, val + tol)
        
    else:
        # --- RANGE BOUNDS MODE ---
        if is_manual:
            c3, c4 = st.columns(2)
            with c3:
                m_min = st.number_input("Min", min_value=min_val, max_value=max_val, value=min_val, step=step, key=f"min_{key_prefix}")
            with c4:
                m_max = st.number_input("Max", min_value=min_val, max_value=max_val, value=max_val, step=step, key=f"max_{key_prefix}")
            return (m_min, m_max)
        else:
            bounds = st.slider(
                "Range", min_value=min_val, max_value=max_val, value=(min_val, max_val),
                step=step, key=f"sli_{key_prefix}", label_visibility="collapsed"
            )
            return bounds

def apply_filters(data, ar, sweep, taper, mach, alpha):
    mask = (
        (data["AR"] >= ar[0]) & (data["AR"] <= ar[1]) &
        (data["QC_Sweep"] >= sweep[0]) & (data["QC_Sweep"] <= sweep[1]) &
        (data["taper"] >= taper[0]) & (data["taper"] <= taper[1]) &
        (data["mach"] >= mach[0]) & (data["mach"] <= mach[1]) &
        (data["alpha"] >= alpha[0]) & (data["alpha"] <= alpha[1])
    )
    return {key: val[mask] for key, val in data.items()}

#-----------------------------------------------------------------------------------------------------------------------
# Streamlit App
#-----------------------------------------------------------------------------------------------------------------------

# Set layout to wide mode to comfortably fit the three-frame grid
st.set_page_config(layout="wide", page_title="Aero Data Explorer")

# ==========================================
# 0. MOCK DATA GENERATION (A few dozen points)
# ==========================================
@st.cache_data
def load_mock_data():
    np.random.seed(42)
    n_points = 100 
    data = {
        "AR": np.random.uniform(5, 30, n_points),
        "taper": np.random.uniform(0.1, 1.0, n_points),
        "QC_Sweep": np.random.uniform(0, 60, n_points),
    }
    
    # Force mock data to align with our discrete grid!
    raw_alpha = np.random.uniform(-5, 15, n_points)
    data["alpha"] = np.round(raw_alpha / 0.25) * 0.25
    
    raw_mach = np.random.uniform(0.1, 2.0, n_points)
    data["mach"] = np.round(raw_mach / 0.05) * 0.05
    
    data["CL"] = 0.1 * data["alpha"] * (1 + 0.1 * data["AR"])
    data["CD"] = 0.02 + (data["CL"]**2) / (np.pi * data["AR"]) + 0.05 * (data["mach"] > 1.0)
    return data

@st.cache_resource
def get_zarr_root():
    """
    Finds all shards and stitches them into a single lazy Dask dictionary.
    Zero RAM used for the actual data arrays here.
    """
    data_dir = "/media/jordan/Ashley_Backup/Wing Data Generation/W1/"
    
    # Grab all shards and sort them so row indices remain perfectly consistent
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "*_shard_*.zarr")))
    
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_dir}")
        
    # The variables we want to load
    available_cols = [
        "alpha",
        "beta",
        "mach",
        "AR",
        "Dihedral",
        "taper",
        "QC_Sweep",
        "CL",
        "CD",
        "CX",
        "CY",
        "CZ",
        "C_l",
        "C_m",
        "C_n",
        "dCL_da",
        "dCL_db",
        "dCL_dM",
        "dCD_da",
        "dCD_db",
        "dCD_dM",
        ]
    
    stitched_data = {}
    for col in available_cols:
        # Create a lazy Dask array for this column across all shards
        lazy_arrays = [da.from_zarr(p, component=col) for p in shard_paths]
        
        # Concatenate them along the row axis
        stitched_data[col] = da.concatenate(lazy_arrays, axis=0)
        
    return stitched_data

@st.cache_data
def load_exploration_sample(sample_size=50000):
    """Pulls a randomized downsample into a Pandas DataFrame for the UI."""
    root = get_zarr_root()
    total_rows = root["alpha"].shape[0]
    
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(total_rows, size=min(sample_size, total_rows), replace=False))
    
    data = {}
    for key, dask_arr in root.items():
        # .compute() pulls it into RAM
        # .ravel() squashes it from (N, 1) to (N,) so Pandas doesn't freak out!
        data[key] = dask_arr[sample_idx].compute().ravel()
        
    df = pd.DataFrame(data)
    
    # --- ROUGH OUTLIER REJECTION ---
    # Filter out physically impossible aerodynamic coefficients
    valid_aero = (
        (df["CL"] > -5.0) & (df["CL"] < 5.0) &
        (df["CD"] > -0.1) & (df["CD"] < 2.0) # CD can occasionally be slightly negative in bad VLM meshes
    )
    
    # Apply the mask and drop the invalid rows
    df_clean = df[valid_aero].copy()
    
    return df_clean

# Initialize the baseline data for the app
raw_data = load_exploration_sample(sample_size=50000)

# # Load baseline data
# raw_data = load_mock_data()

# Initialize session state for active data if it doesn't exist
if "active_data" not in st.session_state:
    st.session_state.active_data = raw_data

# ==========================================
# TOP ROW: MAIN VISUALIZATION & POLARS
# ==========================================
# Split the top row into a large left frame (70% width) and a right frame (30% width)
top_left_col, top_right_col = st.columns([0.7, 0.3])

with top_left_col:
    st.subheader("📊 Primary Visualization Canvas")
    viz_tab1, viz_tab2 = st.tabs(["🚀 Design Space Exploration", "📈 Data Distributions"])
    
    # Notice we are passing st.session_state.active_data to Plotly instead of raw_data!
    available_cols = list(st.session_state.active_data.keys())

    VIZ_HEIGHT = 600

    with viz_tab1:
        color_options = ["None"] + available_cols
        
        # Split the main canvas into two side-by-side 3D plots
        plot_col1, plot_divider, plot_col2 = st.columns([0.48, 0.04, 0.48])
        
        # --- LEFT 3D PLOT ---
        with plot_col1:
            # 8 micro-columns for inline labels and boxes
            c_lx1, cx1, c_ly1, cy1, c_lz1, cz1, c_lc1, cc1 = st.columns([1, 4, 1, 4, 1, 4, 1, 4])
            
            with c_lx1: st.markdown("<div style='margin-top:8px;'><b>X:</b></div>", unsafe_allow_html=True)
            with cx1: px1_x = st.selectbox("X Axis 1", available_cols, index=available_cols.index("alpha"), key="p1_x", label_visibility="collapsed")
            
            with c_ly1: st.markdown("<div style='margin-top:8px;'><b>Y:</b></div>", unsafe_allow_html=True)
            with cy1: px1_y = st.selectbox("Y Axis 1", available_cols, index=available_cols.index("mach"), key="p1_y", label_visibility="collapsed")
            
            with c_lz1: st.markdown("<div style='margin-top:8px;'><b>Z:</b></div>", unsafe_allow_html=True)
            with cz1: px1_z = st.selectbox("Z Axis 1", available_cols, index=available_cols.index("CL"), key="p1_z", label_visibility="collapsed")
            
            with c_lc1: st.markdown("<div style='margin-top:8px;'><b>🎨</b></div>", unsafe_allow_html=True)
            with cc1: px1_c = st.selectbox("Color 1", color_options, index=color_options.index("None"), key="p1_c", label_visibility="collapsed")
            
            actual_c1 = None if px1_c == "None" else px1_c
            hover_dict1 = {px1_x: ':.3f', px1_y: ':.3f', px1_z: ':.3f'}
            if actual_c1: hover_dict1[actual_c1] = ':.3f'
            
            if len(st.session_state.active_data[px1_x]) > 0:
                fig1 = px.scatter_3d(
                    st.session_state.active_data, 
                    x=px1_x, y=px1_y, z=px1_z, color=actual_c1,
                    hover_data=hover_dict1
                )
                # Strip redundant axis titles from the 3D projection
                fig1.update_layout(
                    margin=dict(l=0, r=0, b=0, t=10), 
                    height=VIZ_HEIGHT
                )
                st.plotly_chart(fig1, width='stretch', key="3d_plot_1")

        with plot_divider:
            st.markdown(f"<div style='border-left: 2px solid rgba(150, 150, 150, 0.3); height: {VIZ_HEIGHT*1.2}px; margin-left: 50%;'></div>", unsafe_allow_html=True)

        # --- RIGHT 3D PLOT ---
        with plot_col2:
            c_lx2, cx2, c_ly2, cy2, c_lz2, cz2, c_lc2, cc2 = st.columns([1, 4, 1, 4, 1, 4, 1, 4])
            
            with c_lx2: st.markdown("<div style='margin-top:8px;'><b>X:</b></div>", unsafe_allow_html=True)
            with cx2: px2_x = st.selectbox("X Axis 2", available_cols, index=available_cols.index("AR"), key="p2_x", label_visibility="collapsed")
            
            with c_ly2: st.markdown("<div style='margin-top:8px;'><b>Y:</b></div>", unsafe_allow_html=True)
            with cy2: px2_y = st.selectbox("Y Axis 2", available_cols, index=available_cols.index("QC_Sweep"), key="p2_y", label_visibility="collapsed")
            
            with c_lz2: st.markdown("<div style='margin-top:8px;'><b>Z:</b></div>", unsafe_allow_html=True)
            with cz2: px2_z = st.selectbox("Z Axis 2", available_cols, index=available_cols.index("CD"), key="p2_z", label_visibility="collapsed")
            
            with c_lc2: st.markdown("<div style='margin-top:8px;'><b>🎨</b></div>", unsafe_allow_html=True)
            with cc2: px2_c = st.selectbox("Color 2", color_options, index=color_options.index("None"), key="p2_c", label_visibility="collapsed")
            
            actual_c2 = None if px2_c == "None" else px2_c
            hover_dict2 = {px2_x: ':.3f', px2_y: ':.3f', px2_z: ':.3f'}
            if actual_c2: hover_dict2[actual_c2] = ':.3f'
            
            if len(st.session_state.active_data[px2_x]) > 0:
                fig2 = px.scatter_3d(
                    st.session_state.active_data, 
                    x=px2_x, y=px2_y, z=px2_z, color=actual_c2,
                    hover_data=hover_dict2
                )
                fig2.update_layout(
                    margin=dict(l=0, r=0, b=0, t=10), 
                    height=VIZ_HEIGHT
                )
                st.plotly_chart(fig2, width='stretch', key="3d_plot_2")
        
    with viz_tab2:
        if len(st.session_state.active_data["AR"]) > 0:
            
            # Pre-select 4 interesting defaults for the grid
            hist_defaults = ["CL", "CD", "AR", "mach"]
            
            # Generate a 2x2 grid
            for row in range(2):
                grid_cols = st.columns(2)
                
                for col in range(2):
                    idx = row * 2 + col
                    with grid_cols[col]:
                        
                        # Micro-columns for the variable selector
                        c_lbl, c_box = st.columns([1, 4])
                        with c_lbl: 
                            st.markdown("<div style='margin-top:8px;'><b>Var:</b></div>", unsafe_allow_html=True)
                        with c_box: 
                            hist_col = st.selectbox(
                                f"Histogram {idx}", 
                                available_cols, 
                                index=available_cols.index(hist_defaults[idx]), 
                                key=f"hist_{idx}", 
                                label_visibility="collapsed"
                            )
                        
                        # Generate the histogram
                        fig_hist = px.histogram(
                            st.session_state.active_data, 
                            x=hist_col, 
                            nbins=20, 
                            color_discrete_sequence=['#1f77b4']
                        )
                        
                        # Strip axis titles to save massive amounts of vertical space
                        fig_hist.update_xaxes(title_text='')
                        fig_hist.update_yaxes(title_text='')
                        
                        # Height set to 220px so two rows total ~440px (matching the 450px 3D plots)
                        fig_hist.update_layout(margin=dict(l=20, r=20, b=20, t=10), height=VIZ_HEIGHT//2)
                        st.plotly_chart(fig_hist, width='stretch', key=f"hist_chart_{idx}")
        else:
            st.warning("No data points match the current filter criteria!")

with top_right_col:
    st.subheader("🎯 Aero Performance Polars")
    
    if len(st.session_state.active_data["AR"]) > 0:
        
        defaults = [
            {"x": "CD", "y": "CL", "color": "None"},     
            {"x": "alpha", "y": "CL", "color": "None"},   
            {"x": "alpha", "y": "CD", "color": "None"}    
        ]
        
        color_options = ["None"] + available_cols
        
        for i in range(3):
            # 6 micro-columns: [Label, Box, Label, Box, Label, Box]
            # Streamlit automatically normalizes these ratios!
            c_lx, cx, c_ly, cy, c_lc, cc = st.columns([1, 4, 1, 4, 1, 4])
            
            # Use a tiny bit of HTML to vertically center the text with the dropdown
            with c_lx: st.markdown("<div style='margin-top:8px;'><b>X:</b></div>", unsafe_allow_html=True)
            with cx: px_x = st.selectbox(f"X{i}", available_cols, index=available_cols.index(defaults[i]["x"]), key=f"px_{i}", label_visibility="collapsed")
            
            with c_ly: st.markdown("<div style='margin-top:8px;'><b>Y:</b></div>", unsafe_allow_html=True)
            with cy: px_y = st.selectbox(f"Y{i}", available_cols, index=available_cols.index(defaults[i]["y"]), key=f"py_{i}", label_visibility="collapsed")
            
            with c_lc: st.markdown("<div style='margin-top:8px;'><b>🎨</b></div>", unsafe_allow_html=True)
            with cc: px_c = st.selectbox(f"C{i}", color_options, index=color_options.index(defaults[i]["color"]), key=f"pc_{i}", label_visibility="collapsed")
            
            actual_color = None if px_c == "None" else px_c
            
            hover_dict = {px_x: ':.3f', px_y: ':.3f'}
            if actual_color:
                hover_dict[actual_color] = ':.3f'
            
            fig_2d = px.scatter(
                st.session_state.active_data, 
                x=px_x, y=px_y, color=actual_color,
                hover_data=hover_dict
            )
            
            # Strip the redundant axis titles to save massive amounts of space
            fig_2d.update_xaxes(title_text='')
            fig_2d.update_yaxes(title_text='')
            
            # With labels gone, we can compress the margins even tighter and drop the height to 250!
            fig_2d.update_layout(margin=dict(l=5, r=5, b=5, t=5), height=200)
            st.plotly_chart(fig_2d, width='stretch', key=f"polar_chart_{i}")
            
    else:
        st.warning("No data points match the current filter criteria!")

st.markdown("---")

# ==========================================
# BOTTOM ROW: INPUT & CONTROL DASHBOARD
# ==========================================
# st.subheader("🎛️ Control Dashboard")

# Removed the st.form! Just standard columns now.
ctrl_col1, c_div1, ctrl_col2, c_div2, ctrl_col3 = st.columns([0.24, 0.02, 0.24, 0.02, 0.48])

with ctrl_col1:
    st.markdown("##### 📐 Geometric Bounds")
    ar_bounds    = filter_widget("Aspect Ratio", 5.0, 30.0, 15.0, "ar", allow_exact_toggle=False)
    sweep_bounds = filter_widget("QC Sweep (deg)", 0.0, 60.0, 20.0, "sweep", allow_exact_toggle=False)
    taper_bounds = filter_widget("Taper Ratio", 0.1, 1.0, 0.5, "taper", allow_exact_toggle=False)

with ctrl_col2:
    st.markdown("##### 💨 Flow Constraints")
    alpha_bounds = filter_widget("Alpha (deg)", -5.0, 15.0, 3.0, "alpha", step=0.25, allow_exact_toggle=True)
    mach_bounds  = filter_widget("mach Number", 0.1, 2.0, 0.5, "mach", step=0.05, allow_exact_toggle=True)

with ctrl_col3:
    st.markdown("### 🚀 Execute Calculations")
    user_expr = st.text_input("Objective Expression", value="CL / CD")
    top_n = st.number_input("Top N Results", min_value=1, max_value=50, value=5, step=1)
    
    if st.button("Apply Filters & Run Sweep", width='stretch'):
        
        # ==========================================
        # TIER 1: THE VISUALIZATION PROXY (NumPy)
        # ==========================================
        # This uses your existing apply_filters function on the 50k in-memory sample
        filtered_sample = apply_filters(raw_data, ar_bounds, sweep_bounds, taper_bounds, mach_bounds, alpha_bounds)
        st.session_state.active_data = filtered_sample
        
        # ==========================================
        # TIER 2: THE TRUE OPTIMIZER (Dask)
        # ==========================================
        try:
            root = get_zarr_root() 
            
            # 1. Build the Dask lazy mask (Include the UI filters)
            global_mask = (root["AR"] >= ar_bounds[0]) & (root["AR"] <= ar_bounds[1])
            global_mask &= (root["QC_Sweep"] >= sweep_bounds[0]) & (root["QC_Sweep"] <= sweep_bounds[1])
            global_mask &= (root["taper"] >= taper_bounds[0]) & (root["taper"] <= taper_bounds[1])
            global_mask &= (root["mach"] >= mach_bounds[0]) & (root["mach"] <= mach_bounds[1])
            global_mask &= (root["alpha"] >= alpha_bounds[0]) & (root["alpha"] <= alpha_bounds[1])
            
            # --- ROUGH OUTLIER REJECTION (Full Database) ---
            global_mask &= (root["CL"] >= -5.0) & (root["CL"] <= 5.0)
            global_mask &= (root["CD"] >= -0.1) & (root["CD"] <= 2.0)
            
            # 2. Execute the mask search across the HDD
            valid_indices = da.where(global_mask)[0].compute().ravel()
            
            if len(valid_indices) > 0:
                # 3. Stream ONLY the required columns into RAM for the valid indices
                available_cols = list(root.keys())
                required_cols = [c for c in available_cols if c in user_expr]
                
                eval_dict = {}
                for col in required_cols:
                    # Load only the data we strictly need for the math
                    eval_dict[col] = root[col][valid_indices].compute().ravel()
                
                # 4. Run Numexpr on the full dataset slice
                raw_scores = ne.evaluate(user_expr, local_dict=eval_dict).ravel()
                
                finite_mask = np.isfinite(raw_scores)
                scores = raw_scores[finite_mask]
                safe_valid_indices = valid_indices[finite_mask]
                
                if len(scores) > 0:
                    # 4. Rank and pull the final Top N rows
                    n_actual = min(top_n, len(scores))
                    best_relative_idx = np.argsort(scores)[-n_actual:][::-1]
                    
                    # FLATTEN 4: Map relative subset indices back to global index
                    best_global_idx = safe_valid_indices[best_relative_idx].ravel()
                    
                    top_dict = {"Score": scores[best_relative_idx], "Row_ID": best_global_idx}
                    for col in available_cols:
                        # FLATTEN 5: Pull the final data safely
                        top_dict[col] = root[col][best_global_idx].compute().ravel()
                        
                    st.session_state.top_results = pd.DataFrame(top_dict)
                    st.session_state.expr_error = None
                else:
                    st.session_state.top_results = None
                    st.session_state.expr_error = "All evaluated results were Infinity or NaN (e.g., division by zero)."
            else:
                st.session_state.top_results = None
                st.session_state.expr_error = "No data points in the full database matched the filters!"
                
        except Exception as e:
            st.session_state.top_results = None
            st.session_state.expr_error = f"Computation Error: {e}"
            
        st.rerun()

# ==========================================
# BOTTOM TIER: OPTIMIZATION RESULTS
# ==========================================
if "top_results" in st.session_state:
    st.markdown("---")
    
    if st.session_state.top_results is not None:
        st.subheader(f"🏆 Top {len(st.session_state.top_results)} Results for: `{user_expr}`")
        
        # Display the dataframe. It will automatically span the wide layout.
        st.dataframe(
            st.session_state.top_results, 
            width='stretch',
            hide_index=True # Hides the arbitrary row numbers for a cleaner look
        )
        
    elif st.session_state.expr_error:
        st.error(f"Sweep Error: {st.session_state.expr_error}")