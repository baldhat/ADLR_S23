## Frame definition

|   |   | 
|-------------|-------------|
| ![VTOL Frame](./assets/vtol/VTOL_Frame.png)   |     ![VTOL Orientation](./assets/vtol/VTOL_orientation.png)      |   
| ![VTOL Prop](./assets/vtol/VTOL_prop.png)   |     ![VTOL angle of attack](./assets/vtol/VTOL_aoa.png)   |
| ![World Frame](./assets/vtol/World_Frame.png)   |       |


!!! info "Body Frame"
    Body always refers to the CoM. Not the position of the IMU, the GPS or the aerodynamic pressure point.





## Functions
```@docs
VtolModel.vtol_model(v_B, actions, param)
VtolModel.aerodynamic_torque_model(v_B, actions, angle_of_attack, airspeed, param)
VtolModel.propeller_torque_model(actions, param)
VtolModel.aerodynamic_force_model(actions, aoa, airspeed, param)
VtolModel.aerodynamic_force_model_wing_surface(actions, aoa, airspeed, param, lateral_airspeed)
VtolModel.propeller_force_model(actions)
```
