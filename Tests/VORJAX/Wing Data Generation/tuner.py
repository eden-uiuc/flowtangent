import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

st.set_page_config(layout="wide", page_title="Physics-Informed Beta Tuner")

st.title("🎯 Physics-Informed Beta Tuner")
st.markdown("""
Instead of guessing statistical parameters, design your distributions using physical intuition.
* **Mode (Peak):** The exact real-world value where you want the highest density of samples.
* **Tightness:** How aggressively the samples cluster around the peak (higher = tighter grouping).
""")
st.markdown("---")

# Pre-load reasonable physical defaults
params = {
    "Aspect Ratio": {"min": 0.1, "max": 30.0, "mode": 8.0, "tightness": 12.0},
    "Taper Ratio": {"min": 0.0, "max": 1.0, "mode": 0.4, "tightness": 8.0},
    "QC Sweep (°)": {"min": -25.0, "max": 65.0, "mode": 15.0, "tightness": 10.0},
    "Twist (°)": {"min": -10.0, "max": 5.0, "mode": -2.0, "tightness": 15.0},
    "Dihedral (°)": {"min": -10.0, "max": 15.0, "mode": 0.0, "tightness": 10.0}
}

cols = st.columns(3)

for i, (name, defaults) in enumerate(params.items()):
    col = cols[i % 3]
    
    with col:
        st.markdown(f"### {name}")
        
        # Row 1: Bounds
        c1, c2 = st.columns(2)
        p_min = c1.number_input("Min Boundary", value=defaults["min"], key=f"{name}_min")
        p_max = c2.number_input("Max Boundary", value=defaults["max"], key=f"{name}_max")
        
        # Row 2: Shaping
        c3, c4 = st.columns(2)
        # We constrain the mode so it can't sit exactly on the absolute min/max boundary
        mode_val = c3.number_input("Mode (Peak)", min_value=p_min + 0.01, max_value=p_max - 0.01, value=defaults["mode"], key=f"{name}_mode")
        tightness = c4.slider("Tightness", 1.0, 50.0, defaults["tightness"], 1.0, key=f"{name}_tight")
        
        if p_min >= p_max:
            st.error("Min must be strictly less than Max.")
            continue
            
        # --- THE MATH ---
        # 1. Normalize the physical mode to a [0, 1] scale
        m = (mode_val - p_min) / (p_max - p_min)
        
        # 2. Back-calculate Alpha and Beta based on the Mode and Tightness
        a = (m * tightness) + 1
        b = ((1 - m) * tightness) + 1
        
        # --- THE PLOT ---
        x = np.linspace(p_min, p_max, 500)
        z = (x - p_min) / (p_max - p_min)
        y = beta.pdf(z, a, b) / (p_max - p_min)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', fill='tozeroy', 
            line=dict(color='#FF5F05') # UIUC Orange
        ))
        
        # Add a vertical dashed line to explicitly show the peak
        fig.add_vline(x=mode_val, line_dash="dash", line_color="#13294B", annotation_text="Peak")
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=20), 
            height=200, 
            xaxis_title="Physical Value", 
            yaxis_showticklabels=False,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- CODE EXPORT ---
        # Generate the exact Python code for the user
        var_name = name.split()[0].lower()
        code_str = f"{var_name}_samples = beta.ppf(u_{var_name}, a={a:.3f}, b={b:.3f}, loc={p_min}, scale={p_max - p_min})"
        st.code(code_str, language="python")