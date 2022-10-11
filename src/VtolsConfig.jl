eth_vtol_param = Dict(
    "air_density" => 1.2, # km/m^3 air density 
    "prop_disk" => 0.0133, # 133 cm^2 (inserted in in m^2) propeller disk area
    "kp1" => 1e-4, # Aerodynamic pitching moment. Parameter kp1 is learned during flight. #For stability, the moment must act against the angle of attack.
    # wing and flap caracteristics
    "b_pitch" => 7.46e-6, # Influence of wing airflow on pitching in Nm/(m/s)^2
    "b_yaw" => 4.12e-6, # Influence of wing airflow on yawing in Nm/(m/s)^2
    "b_roll" => 3.19e-6, # Influence of wing airflow on rolling in Nm/(m/s)^2
    "c_pitch" => 2.18e-4, # Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 3.18e-4, # Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    # Motor Parameters
    "torque_to_thrust" => 8.72e-3, # in Nm/N 
    "prop_distance" => 0.14, # 14 cm Propeller offset along the y_B axis
    "rotation_damping" => [ 0.0; 0.0; 0.0],
    # Estimated aerodynamic parameters
    "kl1" => 1e-1,
    "kl2" => 1e-1,
    "kl3" => 0.0,
    "kd1" => 1e-4,
    "kd2" => 1e-4,
    "kd3" => 0.0,
    "f_min" => 0.1, # N thrust on rotors
    "f_max" => 1.2,
    "δ_min" => -0.87,
    "δ_max" => 0.79,
    # Rigid Body
    "mass" => 0.150,
    "inertia" => [ 4.62e-4 0.0 0.0; 0.0 2.32e-3 0.0 ; 0.0 0.0 1.87e-3],
    "inertia_inv" => inv([ 4.62e-4 0.0 0.0; 0.0 2.32e-3 0.0 ; 0.0 0.0 1.87e-3]),
    "CoM" => [ 0.0; 0.0; 0.0],
    "Cw" => 1.15,
)




brett_vtol3_param = Dict(
    "air_density" => 1.2, # km/m^3 air density 
    "prop_disk" => 0.086, # 133 cm^2 (inserted in in m^2) propeller disk area
    "kp1" => 1e-4, #  ??? Aerodynamic pitching moment. Parameter kp1 is learned during flight. #For stability, the moment must act against the angle of attack.
    # wing and flap caracteristics
    "b_pitch" => 7.46e-6, # ??? Influence of wing airflow on pitching in Nm/(m/s)^2
    "b_yaw" => 4.12e-6, # ??? Influence of wing airflow on yawing in Nm/(m/s)^2
    "b_roll" => 3.19e-6, # ??? Influence of wing airflow on rolling in Nm/(m/s)^2
    "c_pitch" => 2.18e-4, # ??? Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 3.18e-4, # ??? Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    # Motor Parameters
    "torque_to_thrust" => 8.72e-3, # ??? in Nm/N 
    "prop_distance" => 0.50, # 14 cm Propeller offset along the y_B axis
    # Estimated aerodynamic parameters
    "kl1" => 1e-1, # ???
    "kl2" => 1e-1, # ???
    "kl3" => 0.0, # ???
    "kd1" => 1e-4, # ???
    "kd2" => 1e-4, # ???
    "kd3" => 0.0, # ???
    "f_min" => 0.1, # ? N thrust on rotors
    "f_max" => 80.0, # ?
    "δ_min" => -1.0, # ?
    "δ_max" => 1.0, # ?
    # Rigid Body
    "mass" => 2.285,
    "inertia" => [ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320],
    "inertia_inv" => inv([ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320]),
    "CoM->IMU" => [ 0.0; 0.0; 0.0],# Die V_B daten sind mit dem GPS fusioniert die verschiebung ist evtl. überflüssig.
    "Cw" => 1.15,
    "rotation_damping" => [ 0.00957; 0.01616; 0.03822],
    # real Actuator scale 
    "f_r_scale" => 0.01,
    "f_l_scale" => 0.01,
    "δ_r_scale" => (1/500)*(30/180)*pi,
    "δ_l_scale" => -(1/500)*(30/180)*pi,
    "f_r_shift" => 900.0,
    "f_l_shift" => 900.0,
    "δ_r_shift" => 1500.0,
    "δ_l_shift" => 1400.0,
    "motor_delay" => 0.0,#0.02351, # in sec
    "flap_delay" => 0.0,#0.01984, # in sec
)


#=
brett_simple_MC = Dict(
    "c_pitch" => 0.08, # ??? Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 0.08, # ??? Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    "prop_pwm_2_airspeed" => 0.1, # ??? propPWM / x = airspeed in m/s
    "prop_distance" => 0.51, # 14 cm Propeller offset along the y_B axis
    # Rigid Body
    "mass" => 2.285,
    "inertia" => [ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320],
    "inertia_inv" => inv([ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320]),
    "CoM->IMU" => [ -0.04; 0.0; 0.0],# Die V_B daten sind mit dem GPS fusioniert die verschiebung ist evtl. überflüssig.
    "Cw" => 1.15,
    "air_density" => 1.2, # km/m^3 air density 
    "rotation_damping" =>[ 0.00957; 0.01616; 0.03822], # [ 0.01014; 0.000206; 0.00103],   # integral(0 -> wing radius) hight(r)*r dr
    # real Actuator scale 
    "f_r_scale" => 0.03,
    "f_l_scale" => 0.03,
    "δ_r_scale" => (1/500)*(30/180)*pi * 1.5,
    "δ_l_scale" => -(1/500)*(30/180)*pi * 1.5,
    "f_r_shift" => 1438.0 - 80.0 ,
    "f_l_shift" => 1438.0 - 40.0,
    "δ_r_shift" => 1500.0 - 50.0,
    "δ_l_shift" => 1400.0
)
=#

# manuell optimierte Parameter
brett_simple_MC = Dict(
    "c_pitch" => 1.218, # ??? Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 0.024, # ??? Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    "prop_distance" => 0.51, # FIX #Propeller distanze along the y_B axis
    "prop_shift" => 0.012,  # FIX # Displacement of the lever arm of the two rotors. the COM is not exactly between the rotors. 
    # Rigid Body
    "mass" => 2.285, # FIX # in kg
    "inertia" => [ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320],  # 3,3 FIX # in kg*m^2 [ 0.052 0.0 0.0; 0.0 0.037 0.0 ; 0.0 0.0 0.087],
    "inertia_inv" => inv([ 0.09841 0.0 0.0; 0.0 0.04561 0.0 ; 0.0 0.0 0.07320]),  # 3,3 FIX
    "CoM->IMU" => [ -0.04; 0.0; 0.0],# Die V_B daten sind mit dem GPS fusioniert die verschiebung ist evtl. überflüssig.
    "Cw" => 1.15,
    "air_density" => 1.2, # FIX # km/m^3 air density 
    "rotation_damping" =>[ 0.00957; 0.01616; 0.03822],
    # real Actuator scale 
    "f_scale" => 0.018, # FIX #0.01824, #0.016 #0.3,
    "δ_scale" => 0.00036, # (1/500)*(30/180)*pi,
    #"δ_l_scale" => (1/500)*(30/180)*pi, # Direkt im Modell invertiert !!! -(1/500)*(30/180)*pi,
    "f_shift" => 1100.0, # FIX
    "δ_l_shift" => 1527.0,
    "δ_r_shift" => 1320.0,
    "kp1" => 0.022,
)




# Mit diesen Parametern bin ich am 05.09. geflogen
#=
brett_simple_MC = Dict(
    "c_pitch" => 0.08, # ??? Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 0.08, # ??? Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    "prop_pwm_2_airspeed" => 0.1, # ??? propPWM / x = airspeed in m/s
    "prop_distance" => 0.51, # 14 cm Propeller offset along the y_B axis
    # Rigid Body
    "mass" => 2.285,
    "inertia" => [ 0.052 0.0 0.0; 0.0 0.037 0.0 ; 0.0 0.0 0.087],
    "inertia_inv" => inv([ 0.052 0.0 0.0; 0.0 0.037 0.0 ; 0.0 0.0 0.087]),
    "CoM->IMU" => [ -0.04; 0.0; 0.0],# Die V_B daten sind mit dem GPS fusioniert die verschiebung ist evtl. überflüssig.
    "Cw" => 1.15,
    "air_density" => 1.2, # km/m^3 air density 
    "rotation_damping_surface" =>[ 0.2; 0.4; 0.5], # [ 0.01014; 0.000206; 0.00103],   # integral(0 -> wing radius) hight(r)*r dr
    # real Actuator scale 
    "f_r_scale" => 0.03,
    "f_l_scale" => 0.03,
    "δ_r_scale" => (1/500)*(30/180)*pi * 1.5,
    "δ_l_scale" => -(1/500)*(30/180)*pi * 1.5,
    "f_r_shift" => 1438.0 - 80.0 ,
    "f_l_shift" => 1438.0 - 40.0,
    "δ_r_shift" => 1500.0 - 50.0,
    "δ_l_shift" => 1400.0
)
=#


struct VTOLparams
    f_shift::Float64
    f_scale::Float64
    δ_l_shift::Float64
    δ_r_shift::Float64
    δ_scale::Float64

    c_pitch::Float64
    c_roll::Float64
    kp1::Float64
    prop_distance::Float64
    prop_shift::Float64

    J_B::Matrix{Float64}
    J_B_inv::Matrix{Float64}
    mass::Float64
    Cw::Float64
    air_density::Float64
    rotation_damping::Vector{Float64}
end