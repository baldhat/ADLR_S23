module Flyonic



# RigidBodies.jl
include("RigidBodies.jl");
using .RigidBodies;
export vtol_add_wind, init_RigidBody, set_next_torque_force!, rigid_body_simulation_step, init_integrator, get_current_state, reset_mechanism_state!, rigid_body_simple, rigid_body_quaternion, discrepancy_integration, parameter_free_integration

# Visualization.jl
include("Visualization.jl");
using .Visualization;
export transform_Vtol, init_visualization, close_visualization, set_Arrow, transform_Vtol2
export create_remote_visualization, create_visualization, create_VTOL, create_sphere, set_transform, close_visualization, create_sphere, set_arrow, transform_arrow, set_actuators

# VtolModel.jl
include("VtolModel.jl");
using .VtolModel;
export vtol_model, init_eth_vtol_model, MC_model, init_MC_vtol_model

# VtolsConfig.jl
include("VtolsConfig.jl");
export eth_vtol_param, brett_vtol3_param, brett_simple_MC #, vtol_body_inertia_B, vtol_body_mass, brett_vtol3_parameters


end # end of module
