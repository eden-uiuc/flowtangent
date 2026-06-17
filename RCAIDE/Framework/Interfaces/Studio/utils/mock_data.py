import plotly.graph_objects as go

flight_profile = {
    'Time/Progress': [i/200.0 for i in range(201)],
    'Alt (ft)': [], 'Mach': [], 'Alpha (deg)': [], 'CL': [], 'CD': [], 'L/D': [], 'Throttle (%)': [], 'Fuel Burn (lb/hr)': []
}

for step in range(201):
    prog = step / 200.0
    if prog <= 0:
        for k in ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'Throttle (%)', 'Fuel Burn (lb/hr)']: flight_profile[k].append(0.0)
    elif prog < 0.25:
        cp = prog / 0.25
        flight_profile['Alt (ft)'].append(35000 * cp)
        flight_profile['Mach'].append(0.78 * cp)
        flight_profile['Alpha (deg)'].append(6.5 + (step % 3) * 0.2)
        flight_profile['CL'].append(0.650 + (step % 2) * 0.005)
        flight_profile['CD'].append(0.0450 + (step % 2) * 0.001)
        flight_profile['Throttle (%)'].append(88.5 + (step % 4) * 0.3)
        flight_profile['Fuel Burn (lb/hr)'].append(6500 + (step % 5) * 20)
    elif prog < 0.75:
        flight_profile['Alt (ft)'].append(35000 + (step % 10 - 5) * 5)
        flight_profile['Mach'].append(0.78 + (step % 8 - 4) * 0.001)
        flight_profile['Alpha (deg)'].append(2.5 + (step % 4 - 2) * 0.1)
        flight_profile['CL'].append(0.450 + (step % 3 - 1) * 0.002)
        flight_profile['CD'].append(0.0250 + (step % 2) * 0.0005)
        flight_profile['Throttle (%)'].append(62.0 + (step % 3 - 1) * 0.2)
        flight_profile['Fuel Burn (lb/hr)'].append(3200 + (step % 4 - 2) * 10)
    elif prog < 1.0:
        dp = (prog - 0.75) / 0.25
        flight_profile['Alt (ft)'].append(35000 - (35000 * dp))
        flight_profile['Mach'].append(0.78 - (0.78 * dp))
        flight_profile['Alpha (deg)'].append(0.5 - (step % 2) * 0.1)
        flight_profile['CL'].append(0.300)
        flight_profile['CD'].append(0.0200)
        flight_profile['Throttle (%)'].append(15.0)
        flight_profile['Fuel Burn (lb/hr)'].append(1200)
    else:
        for k in ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'Throttle (%)', 'Fuel Burn (lb/hr)']: flight_profile[k].append(0.0)

for i in range(201):
    c_d = flight_profile['CD'][i]
    flight_profile['L/D'].append(flight_profile['CL'][i] / c_d if c_d > 0 else 0)

plot_variables = ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'L/D', 'Throttle (%)', 'Fuel Burn (lb/hr)']

def create_plot(var_name):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flight_profile['Time/Progress'], y=flight_profile[var_name], mode='lines', line=dict(color='#3b82f6', width=2)))
    fig.update_layout(
        margin=dict(l=30, r=20, t=10, b=30),
        xaxis=dict(title='Mission Progress', tickformat='.0%', gridcolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0'),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig