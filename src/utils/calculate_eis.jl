using Unitful
"T and p must be passed with unitful units, returns potential temperature in K"
function calculate_pot_temp(T,p)
    R = 287.05u"J/(kg*K)"  # Specific gas constant for dry air
    c_p = 1004u"J/(kg*K)"  # Specific heat capacity at constant pressure for dry air
    p0 = 100000u"Pa"  # Reference pressure in Pascals
    T = uconvert(u"K", T)
    theta = T * (upreferred(p0 / p))^(upreferred(R / c_p))
    return upreferred(theta)
end

"T must be passed with unitful units, returns saturation vapor pressure in kPa, via buck equation"
function calculate_saturation_vapor_pressure(T)
    T = uconvert(u"°C", T)  # Convert to Celsius
    T = ustrip(T)
    e_s = 0.61121*exp((18.678 - T/234.5) * (T/(257.14 + T))) * u"kPa"  # Saturation vapor pressure in kPa
    return e_s
end

function calculate_dewpoint_temperature(T, RH)
    T = uconvert(u"°C", T)  # Convert to Celsius
    T = ustrip(T)
    dewpoint = 243.04 * (log(RH / 100) + (17.625 * T) / (243.04 + T)) / 
               (17.625 - log(RH / 100) - (17.625 * T) / (243.04 + T))
    return upreferred(dewpoint * u"°C")
end

"T and p must be passed with unitful units, returns saturation mixing ratio in unitless units"
function calculate_saturation_mixing_ratio(T, p)
    T = uconvert(u"K", T)
    e_s = calculate_saturation_vapor_pressure(T)
    p_dry = p - e_s  # Partial pressure of dry air
    R_d = 287.05u"J/(kg*K)"  # Specific gas constant for dry air
    R_v = 461.5u"J/(kg*K)"  # Specific gas constant for water vapor
    density_dry_air = p_dry / (R_d * T)  # Density of dry air
    density_water_vapor = e_s / (R_v * T)  # Density of water vapor
    mixing_ratio = density_water_vapor / density_dry_air  # Saturation mixing ratio (unitless)
    return upreferred(mixing_ratio)
end

"T and p must be unitful, returns moist adiabatic lapse rate in K/km"
function calc_moist_adiabatic_potential_temp_LR(T,p)
    T = uconvert(u"K", T)

    g = 9.81u"m/s^2"  # Gravitational acceleration
    c_p = 1004u"J/(kg*K)"  # Specific heat capacity at constant pressure for dry air
    L_v = 2260u"J/g"

    q_s = calculate_saturation_mixing_ratio(T, p)  # Saturation mixing ratio (unitless)
    R_d = 287.05u"J/(kg*K)"  # Specific gas constant for dry air
    R_v = 461.5u"J/(kg*K)"  # Specific gas constant for water vapor

    moist_potential_alr = (g / c_p) * (1 - (1 + L_v * q_s / (R_d * T)) / (1 + (L_v^2 * q_s) / (c_p * R_v * T^2)))
    return uconvert(u"K/km", moist_potential_alr)
end

"After Espy, calculate the LCL from T_0, RH_0, and p_0. All must be passed with unitful units. Returns LCL in meters."
function calculate_LCL(T_0, RH_0)
    T_0 = uconvert(u"K", T_0)
    T_d = calculate_dewpoint_temperature(T_0, RH_0 * 100)  # Convert RH to percentage for dew point calculation

    h_const = 125u"m/K"

    z_lcl = h_const*(T_0 - T_d)
    return upreferred(z_lcl)
end

"Pass temperatures in Kelvin please, returns EIS in K"
function calculate_EIS(T_1000, T_700; RH_0 = 0.8, z_700 = nothing, p_0 = 1u"atm")
    T_1000 = T_1000 * u"K"
    T_700 = T_700 * u"K"

    theta_1000 = calculate_pot_temp(T_1000, p_0)
    theta_700 = calculate_pot_temp(T_700, 700u"hPa")

    T_850 = (T_1000 + T_700) / 2  # Approximate 850 hPa temperature as average of 1000 and 700 hPa

    R_d = 287.05u"J/(kg*K)"  # Specific gas constant for dry air
    g = 9.81u"m/s^2"  # Gravitational acceleration

    if isnothing(z_700)
        # Estimate z_700 using the hypsometric equation
        z_700 = (R_d * T_850) / g * log(p_0 / 700u"hPa")  # Approximate height of 700 hPa level
    end
    
    T_850 = (T_1000 + T_700) / 2  # Approximate 850 hPa temperature as average of 1000 and 700 hPa

    moist_potential_alr = calc_moist_adiabatic_potential_temp_LR(T_850, 850u"hPa")
        
    z_lcl_over_1000_hpa = calculate_LCL(T_1000, RH_0)
    z_1000 = (R_d * T_1000) / g * log(p_0 / 1000u"hPa")  # Approximate height of 1000 hPa level

    z_lcl = z_1000 + z_lcl_over_1000_hpa  # Height of LCL above sea level

    EIS = theta_700 - theta_1000 - moist_potential_alr * (z_700 - z_lcl)
    return ustrip(uconvert(u"K", EIS))
end