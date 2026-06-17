import math
import airportsdata

AIRPORT_DB = airportsdata.load('IATA')

def calculate_haversine_nm(lat1, lng1, lat2, lng2):
    """Calculates the great-circle distance between two points in Nautical Miles."""
    R_nm = 3440.065 # Earth radius in Nautical Miles
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_phi / 2) ** 2 + 
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R_nm * c

def get_great_circle_point(lat1, lon1, lat2, lon2, fraction):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    if d == 0: return math.degrees(lat1), math.degrees(lon1)
    A = math.sin((1 - fraction) * d) / math.sin(d)
    B = math.sin(fraction * d) / math.sin(d)
    x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
    y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
    z = A * math.sin(lat1) + B * math.sin(lat2)
    return math.degrees(math.atan2(z, math.sqrt(x**2 + y**2))), math.degrees(math.atan2(y, x))

def get_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def generate_procedural_flight_profile(origin_iata, destination_iata):
    # Fallback default values if IATA string isn't found directly
    orig = AIRPORT_DB.get(origin_iata.upper())
    dest = AIRPORT_DB.get(destination_iata.upper())

    if not orig: orig = {'lat': 40.6413, 'lon': -73.7781, 'elevation': 13}
    if not dest: dest = {'lat': 51.4700, 'lon': -0.4543, 'elevation': 83}
    
    # 1. Geometry Calculations
    distance_nm = calculate_haversine_nm(orig['lat'], orig['lon'], dest['lat'], dest['lon'])
    
    # 2. Set Architectural Target Parameters
    cruise_alt_target = 37000 if distance_nm > 1000 else 24000
    target_mach = 0.78 if distance_nm > 500 else 0.55
    
    # Approximate climbing performance profile
    climb_rate_fpm = 2200  
    climb_speed_kt = 320   # average ground speed knots during climb
    descent_rate_fpm = 1800
    descent_speed_kt = 280 # average ground speed knots during descent
    cruise_speed_kt = target_mach * 573 # Ground speed approx proxy
    
    # 3. Process Mission Segments (Time/Distance Math)
    # Climb phase calculations
    time_climb_min = (cruise_alt_target - orig['elevation']) / climb_rate_fpm
    dist_climb_nm = (time_climb_min / 60.0) * climb_speed_kt
    
    # Descent phase calculations
    time_descent_min = (cruise_alt_target - dest['elevation']) / descent_rate_fpm
    dist_descent_nm = (time_descent_min / 60.0) * descent_speed_kt
    
    # Cap segments if the distance is too short to reach full altitude ceiling
    if (dist_climb_nm + dist_descent_nm) > distance_nm:
        # Scale back parameters for short-haul hopper flights
        scale = distance_nm / (dist_climb_nm + dist_descent_nm)
        dist_climb_nm *= scale
        dist_descent_nm *= scale
        cruise_alt_target *= (scale * 0.9)
        time_climb_min = (cruise_alt_target - orig['elevation']) / climb_rate_fpm
        time_descent_min = (cruise_alt_target - dest['elevation']) / descent_rate_fpm
        dist_cruise_nm = 0
        time_cruise_min = 0
    else:
        dist_cruise_nm = distance_nm - dist_climb_nm - dist_descent_nm
        time_cruise_min = (dist_cruise_nm / cruise_speed_kt) * 60.0

    total_time_min = time_climb_min + time_cruise_min + time_descent_min
    
    # 4. Generate the Discrete Data Arrays (201 Profile Points)
    STEPS = 200
    flight_profile = {
        'Time/Progress': [i / float(STEPS - 1) for i in range(STEPS)],
        'Alt (ft)': [0.], 'Mach': [0.], 'Alpha (deg)': [0.], 
        'CL': [0.], 'CD': [0.], 'L/D': [0.], 'Throttle (%)': [0.], 'Fuel Burn (lb/hr)': [0.]
    }
    
    # Split intervals linearly by real-world calculated time fractions
    pct_climb = time_climb_min / total_time_min
    pct_cruise = (time_climb_min + time_cruise_min) / total_time_min

    for step in range(STEPS):
        prog = step / float(STEPS - 1)
        
        # CLIMB PHASE
        if prog < pct_climb:
            phase_fraction = prog / pct_climb if pct_climb > 0 else 0
            alt = orig['elevation'] + (cruise_alt_target - orig['elevation']) * phase_fraction
            mach = 0.25 + (target_mach - 0.25) * phase_fraction
            alpha = 7.0 - (4.0 * phase_fraction) + (step % 3) * 0.1
            cl = 0.70 - (0.25 * phase_fraction)
            cd = 0.050 - (0.023 * phase_fraction)
            throttle = 92.0 - (4.0 * phase_fraction)
            fuel = 7500 - (1500 * phase_fraction) + (step % 4) * 20
            
        # CRUISE PHASE
        elif prog < pct_cruise:
            # Steady-state with flight vibrations noise
            alt = cruise_alt_target + (step % 6 - 3) * 5
            mach = target_mach + (step % 4 - 2) * 0.001
            alpha = 2.8 + (step % 4 - 2) * 0.05
            cl = 0.43 + (step % 3 - 1) * 0.002
            cd = 0.024 + (step % 2) * 0.0003
            throttle = 64.0 + (step % 3 - 1) * 0.15
            fuel = 3400 + (step % 4 - 2) * 12
        
        # LANDED PHASE
        elif prog == 1.0:
            alt = 0.0
            mach = 0.0
            alpha = 0.0
            cl = 0.0
            cd = 0.0
            throttle=0.0
            fuel=0.0

        # DESCENT PHASE
        else:
            phase_fraction = (prog - pct_cruise) / (1.0 - pct_cruise) if (1.0 - pct_cruise) > 0 else 0
            alt = cruise_alt_target - (cruise_alt_target - dest['elevation']) * phase_fraction
            mach = target_mach - (target_mach - 0.22) * phase_fraction
            alpha = 2.0 - (1.5 * phase_fraction) - (step % 2) * 0.05
            cl = 0.35 - (0.10 * phase_fraction)
            cd = 0.026 - (0.005 * phase_fraction)
            throttle = 18.0 - (6.0 * phase_fraction)
            fuel = 1400 - (400 * phase_fraction)

        # Append variables to the simulation engine dictionary
        flight_profile['Alt (ft)'].append(alt)
        flight_profile['Mach'].append(mach)
        flight_profile['Alpha (deg)'].append(alpha)
        flight_profile['CL'].append(cl)
        flight_profile['CD'].append(cd)
        flight_profile['Throttle (%)'].append(throttle)
        flight_profile['Fuel Burn (lb/hr)'].append(fuel)
        flight_profile['L/D'].append(cl / cd if cd > 0 else 0)

    plot_variables = ['Alt (ft)', 'Mach', 'Alpha (deg)', 'CL', 'CD', 'L/D', 'Throttle (%)', 'Fuel Burn (lb/hr)']
    
    # Return the data payload plus metadata the UI needs to update the Map positions
    meta = {
        'orig_lat': orig['lat'], 'orig_lng': orig['lon'],
        'dest_lat': dest['lat'], 'dest_lng': dest['lon'],
        'distance_nm': distance_nm,
        'route_points': [get_great_circle_point(orig['lat'], orig['lon'], dest['lat'], dest['lon'], i/100.0) for i in range(101)]
    }

    return flight_profile, plot_variables, meta