module Flyonic


# CrazyflieController.jl
include("CrazyflieController.jl");
using .CrazyflieController
export init_crazyflie_controller


# RigidBodies.jl
include("RigidBodies.jl");
using .RigidBodies;
export vtol_add_wind, rigid_body_simple

# Visualization.jl
include("Visualization.jl");
using .Visualization;
export create_remote_visualization, create_visualization, create_VTOL, create_sphere, set_transform, close_visualization, create_sphere, set_arrow, transform_arrow, set_actuators, set_Crazyflie_actuators, set_color, create_Crazyflie

# VtolModel.jl
include("VtolModel.jl");
using .VtolModel;
export vtol_model, crazyflie_model

# VtolsConfig.jl
include("VtolsConfig.jl");
export eth_vtol_param, crazyflie_param

end # end of module
