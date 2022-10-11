# Rigid Body Integration

[RigidBodyDynamics Docu](https://juliarobotics.org/RigidBodyDynamics.jl/stable/)

There are two options for rigid body simulation [`rigid_body_sim_simple`](@ref Main.RigidBodies.rigid_body_sim_simple) and [`rigid_body_simulation_step`](@ref Main.RigidBodies.rigid_body_simulation_step). As well as a simple function [`vtol_add_wind`](@ref Main.RigidBodies.vtol_add_wind) to add the wind speed to the relative drone speed.

!!! warning "Do not use RigidBodyDynamics"
    RigidBodyDynamics.jl is not automatically differentiated and will eventually be deleted from the repository.

## Examples

### [`rigid_body_simple`](@ref Main.RigidBodies.rigid_body_simple)
```julia
include("../Flyonic.jl");
using .Flyonic;
using LinearAlgebra
using Rotations;
using LoopThrottle;

# Inizialise Visual Representation
create_visualization();
create_VTOL("vtol");

# initialise drone position
x_W = [0.0; 0.0; 0.0];
v_B = [0.0; 0.0; 0.0];
R_W = [0.0  0.0  1.0;
       0.0 -1.0  0.0;
       1.0  0.0  0.0];
ω_B = [0.0; 0.0; 0.0];

# Set the global wind
wind_speed = 0.0;
wind_direction_W = [0.0;0.0;0.0];

# start, stop and step time
t = 0.0             
final_time = 2.0
Δt = 1e-3;

# @throttle is used to run the simulation in wall time when the dynamics simulation is faster.
@throttle t while t < final_time
    
    # Get the next actions from somewhere. Here it is a fixes action
    action = [10.0, 10.0, -0.035, -0.035]

    # caluclate wind impact
    v_in_wind_B = vtol_add_wind(v_B, R_W, wind_speed, wind_direction_W)
    
    # caluclate aerodynamic forces
    torque_B, force_B = vtol_model(v_in_wind_B, action, vtol_parameters);
        
    # set the next forces for the rigid body simulation
    x_W, v_B, R_W, ω_B, t = rigid_body_simple(torque_B, force_B, x_W, v_B, R_W, ω_B, t, Δt, vtol_body_inertia_B, vtol_body_mass)
    
    # set Visualisation
    set_transform("vtol", x_W,QuatRotation(R_W));
end
```


### [`rigid_body_simulation_step`](@ref Main.RigidBodies.rigid_body_simulation_step)
This example is reduced. And describes only the extra pieces that are needed for the RigidBodyDynamics.jl package.

```julia
# Setting the rigid body simulation
mechanism_state, vtol_joint = init_RigidBody(x_W, v_B, R_W, ω_B, vtol_body_inertia_B, vtol_body_mass);
integrator, storage = init_integrator(mechanism_state);

# Visualize initial state
x_W, v_B, R_W, ω_B = get_current_state(mechanism_state, vtol_joint);
set_transform("vtol", x_W,QuatRotation(R_W)); # Visualisation must be initialised beforehand.

# Can always be called to reset state
reset_mechanism_state!(mechanism_state, vtol_joint, x_W, v_B, R_W, ω_B);

while t < final_time
        

    # caluclate aerodynamic forces
    vtol_torque_B, vtol_force_B = vtol_model(v_B, action, vtol_parameters);

    # set the next forces for the rigid body simulation
    set_next_torque_force!(vtol_torque_B, vtol_force_B)
        
    # integrate rigid body dynamics for Δt
    x_W, v_B, R_W, ω_B, time = rigid_body_simulation_step(integrator, t, Δt, storage)
        
    # Visualize the new state
    set_transform("vtol", x_W,QuatRotation(R_W));

    t += Δt
end
```




## Functions
```@autodocs
Modules = [RigidBodies]
Order   = [:function, :type]
```