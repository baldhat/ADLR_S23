module CrazyflieController

using Rotations;

export init_crazyflie_controller


"""
    init_PID(KP::Real, KI::Real, iLimit::Real, KD::Real, KFF::Real, DT::Real)

init PID parameter and returns th updat function wich calculates the PID output
"""
function init_PID(KP::Real, KI::Real, iLimit::Real, KD::Real, KFF::Real)

    integ_Error = 0.0;
    prev_Error = 0.0;

    function update_PID(Desired::Real, Actual::Real, DT::Real)

        # Error
        Error = Desired - Actual;

        # P 
        outP = KP * Error;
    
        # I 
        integ_Error += Error * DT;
        integ_Error = clamp(integ_Error, -iLimit, iLimit)
        outI = KI * integ_Error;

        # D
        deriv_error = (Error - prev_Error) / DT;
        # TODO: filter https://github.com/bitcraze/crazyflie-firmware/blob/2748522037cfaa19186d203652da9c723138d0ca/src/modules/src/pid.c#L104
        outD = KD * deriv_error;
        prev_Error = Error;

        # FF
        outFF = KFF * Desired;

        # TODO: filter complete output instead of only D component to compensate for increased noise from increased barometer influence
        # lpf2pApply(&pid->dFilter, output);
        output = outP + outI + outD + outFF;
        # TODO: output = constrain(output, -outputLimit, outputLimit);

        return output;
    end

    return update_PID
end


"""
    crazyflie_PID_controller = init_crazyflie_controller()

    motor_1, motor_2, motor_3, motor_4 crazyflie_PID_controller(
        rollRateActual::Real, pitchRateActual::Real, yawRateActual::Real,
        eulerRollActual::Real, eulerPitchActual::Real, eulerYawActual::Real,
        eulerYawDesired::Real,
        actual_vel_x_B::Real, actual_vel_y_B::Real, actual_vel_z_B::Real,
        actual_pos_x_W::Real, actual_pos_y_W::Real, actual_pos_z_W::Real,
        desired_pos_x_W::Real, desired_pos_y_W::Real, desired_pos_z_W::Real,
        DT::Real)

init Cascaded PID controller. Returns Cascaded PID controller.
https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/
"""
function init_crazyflie_controller()

    vel_x_PID = init_PID(10.0, 0.0, 0.0, 0.0, 0.0);
    vel_y_PID = init_PID(10.0, 0.0, 0.0, 0.0, 0.0);
    vel_z_PID = init_PID(10.0, 0.0, 0.0, 0.0, 0.0);

    function positionController(actual_pos_x_W::Real, actual_pos_y_W::Real, actual_pos_z_W::Real,
                                desired_pos_x_W::Real, desired_pos_y_W::Real, desired_pos_z_W::Real,
                                DT::Real)

        desired_vel_x_W = vel_x_PID(desired_pos_x_W, actual_pos_x_W, DT);
        desired_vel_y_W = vel_y_PID(desired_pos_y_W, actual_pos_y_W, DT);
        desired_vel_z_W = vel_z_PID(desired_pos_z_W, actual_pos_z_W, DT);

        return desired_vel_x_W, desired_vel_y_W, desired_vel_z_W;
    end


    thrustMin = 2000.0;#20000.0; # Minimum thrust value to output
    thrustScale = 1000.0;
    thrustBase = 50000.0; # approximate throttle needed when in perfect hover. More weight/older battery can use a higher value
    pLimit = 20.0; # PID_VEL_PITCH_MAX
    rLimit = 20.0; # PID_VEL_ROLL_MAX

    thrust_PID = init_PID(25.0, 15.0, 1_000.0, 0.0, 0.0);
    eulerPitch_PID = init_PID(25.0, 1.0, 1_000.0, 0.0, 0.0); # Vel_X
    eulerRoll_PID = init_PID(25.0, 1.0, 1_000.0, 0.0, 0.0); # Vel_Y

    function velocityController(actual_vel_x_B::Real, actual_vel_y_B::Real, actual_vel_z_B::Real, 
                                desired_vel_x_B::Real, desired_vel_y_B::Real, desired_vel_z_B::Real, 
                                DT::Real)

        # Roll and Pitch
        eulerPitchDesired = eulerPitch_PID(desired_vel_x_B, actual_vel_x_B, DT);
        eulerRollDesired = -eulerRoll_PID(desired_vel_y_B, actual_vel_y_B, DT);
        eulerPitchDesired = clamp(eulerPitchDesired, -pLimit, pLimit);
        eulerRollDesired  = clamp(eulerRollDesired,  -rLimit, rLimit);

        # Thrust
        thrustRaw = thrust_PID(desired_vel_z_B, actual_vel_z_B, DT);
        # Scale the thrust and add feed forward term
        thrust = thrustRaw * thrustScale + thrustBase;
        # Limit Thrust to UINT16
        thrust = clamp(thrust, thrustMin, 65535.0);

        return thrust, eulerRollDesired, eulerPitchDesired
    end 


    # params from https://github.com/bitcraze/crazyflie-firmware/blob/2748522037cfaa19186d203652da9c723138d0ca/src/platform/interface/platform_defaults_cf2.h#L64
    roll_PID = init_PID(6.0, 3.0, 20.0, 0.0, 0.0);
    pitch_PID = init_PID(6.0, 3.0, 20.0, 0.0, 0.0);
    yaw_PID = init_PID(6.0, 1.0, 360.0, 0.35, 0.0);

    # https://github.com/bitcraze/crazyflie-firmware/blob/2748522037cfaa19186d203652da9c723138d0ca/src/modules/src/attitude_pid_controller.c#L156
    function attitudeControllerCorrectAttitudePID(  eulerRollActual::Real, eulerPitchActual::Real, eulerYawActual::Real,
                                                    eulerRollDesired::Real, eulerPitchDesired::Real, eulerYawDesired::Real,
                                                    DT::Real)

        rollRateDesired  = roll_PID(eulerRollDesired, eulerRollActual, DT);
        pitchRateDesired = pitch_PID(eulerPitchDesired, eulerPitchActual, DT);
    
        # Update PID for yaw axis
        yawError = eulerYawDesired - eulerYawActual;
        if (yawError > 180.0)
            yawError -= 360.0;
        elseif (yawError < -180.0)
            yawError += 360.0;
        end

        eulerYawDesired = yawError + eulerYawActual;
        yawRateDesired = yaw_PID(eulerYawDesired, eulerYawActual, DT);
        yawRateDesired = clamp(yawRateDesired, -20.0, 20.0)

        return rollRateDesired, pitchRateDesired, yawRateDesired
    end


    # params from https://github.com/bitcraze/crazyflie-firmware/blob/2748522037cfaa19186d203652da9c723138d0ca/src/platform/interface/platform_defaults_cf2.h#L45
    rollRate_PID = init_PID(250.0, 500.0, 33.3, 2.5, 0.0);
    pitchRate_PID = init_PID(250.0, 500.0, 33.3, 2.5, 0.0);
    yawRate_PID = init_PID(120.0, 16.7, 166.7, 0.0, 0.0);

    # https://github.com/bitcraze/crazyflie-firmware/blob/2748522037cfaa19186d203652da9c723138d0ca/src/modules/src/attitude_pid_controller.c#L141
    function attitudeControllerCorrectRatePID(
                rollRateActual::Real, pitchRateActual::Real, yawRateActual::Real,
                rollRateDesired::Real, pitchRateDesired::Real, yawRateDesired::Real,
                DT::Real)

                rollOutput  = rollRate_PID(rollRateDesired, rollRateActual, DT);
                pitchOutput = pitchRate_PID(pitchRateDesired, pitchRateActual, DT);
                yawOutput   = yawRate_PID(yawRateDesired, yawRateActual, DT);

                return rollOutput, pitchOutput, yawOutput
    end


    #  https://github.com/bitcraze/crazyflie-firmware/blob/8cbdceeb4ceab868729776b8b6ba6aa48bcc04d5/src/modules/src/power_distribution_quadrotor.c#L79
    function powerDistributionLegacy(roll, pitch, yaw, thrust)
        roll = roll / 2.0;
        pitch = pitch / 2.0;
    
        pitch = -pitch; # Pitch is defined with LEFT hand rule https://www.bitcraze.io/documentation/system/platform/cf2-coordinate-system/
    
        motor_1 = thrust - roll + pitch + yaw;
        motor_2 = thrust - roll - pitch - yaw;
        motor_3 = thrust + roll - pitch + yaw;
        motor_4 = thrust + roll + pitch - yaw;

        return motor_1, motor_2, motor_3, motor_4
    end

    # Position controller
    function crazyflie_PID_controller(  R_W::Array{Real},
                                        ω_B::Vector{Real},
                                        v_B::Vector{Real},
                                        x_W::Vector{Real},
                                        desired_pos_W::Vector{Real},
                                        DT::Real)

        actual_pos_x_W = x_W[1]
        actual_pos_y_W = x_W[2]
        actual_pos_z_W = x_W[3]       
        
        
        desired_pos_x_W = desired_pos_W[1];
        desired_pos_y_W = desired_pos_W[2];
        desired_pos_z_W = desired_pos_W[3];


        # Position PID
        desired_vel_x_W, desired_vel_y_W, desired_vel_z_W = positionController( actual_pos_x_W, actual_pos_y_W, actual_pos_z_W,
                                                                                desired_pos_x_W, desired_pos_y_W, desired_pos_z_W,
                                                                                DT);
        
        desired_v_W = [desired_vel_x_W; desired_vel_y_W; desired_vel_z_W]
        #println(desired_v_W)
        desired_v_B = transpose(R_W)*desired_v_W;
        desired_vel_x_B = desired_v_B[1]
        desired_vel_y_B = desired_v_B[2]
        desired_vel_z_B = desired_v_B[3]

        actual_vel_x_B = v_B[1]
        actual_vel_y_B = v_B[2]
        actual_vel_z_B = v_B[3]
        
        # Velocity PID
        thrustOutput, eulerRollDesired, eulerPitchDesired = velocityController( actual_vel_x_B, actual_vel_y_B, actual_vel_z_B, 
                                                                                desired_vel_x_B, desired_vel_y_B, desired_vel_z_B,
                                                                                DT);
        

        eulerRollActual  = RotZYX(R_W).theta3 * 180.0 / pi
        eulerPitchActual = RotZYX(R_W).theta2 * 180.0 / pi
        eulerYawActual   = RotZYX(R_W).theta1 * 180.0 / pi

        eulerYawDesired = 0.0;


        # Altitude PID
        rollRateDesired, pitchRateDesired, yawRateDesired = attitudeControllerCorrectAttitudePID(   eulerRollActual, eulerPitchActual, eulerYawActual,
                                                                                                    eulerRollDesired, eulerPitchDesired, eulerYawDesired,
                                                                                                    DT);

        rollRateActual  = ω_B[1]
        pitchRateActual = ω_B[2]
        yawRateActual   = ω_B[3]

        # Altitude Rate PID
        rollOutput, pitchOutput, yawOutput = attitudeControllerCorrectRatePID(  rollRateActual, pitchRateActual, yawRateActual,
                                                                                rollRateDesired, pitchRateDesired, yawRateDesired, 
                                                                                DT);
        
        
        # Power distribution 
        motor_1, motor_2, motor_3, motor_4 = powerDistributionLegacy(rollOutput, pitchOutput, yawOutput, thrustOutput);

        return motor_1, motor_2, motor_3, motor_4
    end


    return crazyflie_PID_controller
end

end # end of module 