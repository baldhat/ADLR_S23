include("../../Flyonic.jl");
using .Flyonic;

using Rotations; # used for initial position

using ReinforcementLearning;
using Distributions;
using StableRNGs;
using Flux;
using Flux.Losses;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;

using Plots;
using Statistics;
using CUDA;

using TensorBoardLogger
using Logging
# using BSON
using BSON: @save, @load # save model


# TensorBoard
logger = TBLogger("logs/landing3d/experiment", tb_increment)
Flyonic.Visualization.create_visualization();

mutable struct VtolEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv # Parametric Constructor for a subtype of AbstractEnv
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    previous_action::ACT
    done::Bool
    t::T
    rng::R

    name::String #for multible environoments
    visualization::Bool
    realtime::Bool # For humans recognizable visualization
    
    # Everything you need aditionaly can also go in here.
    x_W::Vector{T}
    v_B::Vector{T}
    R_W::Matrix{T}
    ω_B::Vector{T}
    wind_W::Vector{T}
    Δt::T

    success_reward::T
    stay_alive_reward::T
    translation_reward::T
    angle_penalty::T
    vel_penalty::T
    rot_vel_penalty::T
    output_penalty::T
    Return::T

    x_target::Vector{T}
    R_target::Matrix{T}
end

function VtolEnv(;
    # define a keyword-based constructor for the type declared in the mutable struct typedef. 
    # It could also be done with the macro Base.@kwdef.
     
    #continuous = true,
    rng = Random.GLOBAL_RNG, # Random number generation
    name = "vtol",
    visualization = false,
    realtime = false,
    kwargs... # let the function take an arbitrary number of keyword arguments 
    )
    
    T = Float64; # explicit type which is used e.g. in state. Cannot be altered due to the poor matrix defininon.

    #action_space = Base.OneTo(21) # 21 discrete positions for the flaps
    
    action_space = Space(
        ClosedInterval{T}[
            0.0..2.0, # thrust let
            0.0..2.0, # thrust right
            -1.0..1.0, # flaps left
            -1.0..1.0, # flaps right
            ], 
    )

    
    state_space = Space( # Twelve continuous values in state space.
        ClosedInterval{T}[
            typemin(T)..typemax(T), # rotation velocity arround body x
            typemin(T)..typemax(T), # rotation velocity arround body y
            typemin(T)..typemax(T), # rotation velocity arround body z
            typemin(T)..typemax(T), # velocity along world x
            typemin(T)..typemax(T), # velocity along world y
            typemin(T)..typemax(T), # velocity along world z
            typemin(T)..typemax(T), # delta world position along x
            typemin(T)..typemax(T), # delta world position along y
            typemin(T)..typemax(T), # delta world position along z
            typemin(T)..typemax(T), # delta rotation x axis in world coordinates (x component)
            typemin(T)..typemax(T), # delta rotation x axis in world coordinates (y component)
            typemin(T)..typemax(T), # delta rotation x axis in world coordinates (z component)
            typemin(T)..typemax(T), # delta rotation z axis in world coordinates (x component)
            typemin(T)..typemax(T), # delta rotation z axis in world coordinates (y component)
            typemin(T)..typemax(T), # delta rotation z axis in world coordinates (z component)
            typemin(T)..typemax(T), # last left thrust
            typemin(T)..typemax(T), # last right thrust
            typemin(T)..typemax(T), # last left flap
            typemin(T)..typemax(T), # last right flap
            ], 
    )
    
    if visualization
        Flyonic.Visualization.create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
        Flyonic.Visualization.set_transform(name, [-10.0; 0.0; 5.0] ,QuatRotation([1 0 0; 0 1 0; 0 0 1]));
        Flyonic.Visualization.set_actuators(name, [0.0; 0.0; 0.0; 0.0])
    end


    environment = VtolEnv(
        action_space,
        state_space,
        zeros(T, 15), # current state, needs to be extended.
        rand(action_space),
        zeros(4), # previous action
        false, # episode done ?
        0.0, # time
        rng, # random number generator  
        name,
        visualization, # visualization
        realtime, # realtime visualization
        
        Array{T}([-10.0; 00.0; 7.0]), # x_W
        Array{T}([5.0; 0.0; 0.0]), # v_B
        Array{T}(Matrix(RotY(-20*pi/180))), # R_W
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        T(0.005), # Δt  

        0.0, # reward part for success
        0.0, # reward part for stay_alive
        0.0, # reward part for close_to_target
        0.0, # reward part for not_upright_orientation
        0.0, # reward part for not_still
        0.0, # reward part for fast_rotation
        0.0, # reward for large outputs
        0.0, # overall return
        
        # Array{T}([-5.0+rand()*10.0; -5.0+rand()*10.0; 4.0+2.0*rand()]), # random x_target
        # Array{T}(Matrix(RotY(-pi/2)*RotX(rand()*2*pi))) # random R_target
        Array{T}([0, 0, 0]), # x_target
        Array{T}(Matrix(RotY(-pi/2))), # R_target
    )
    
    RLBase.reset!(environment)
    
    return environment
    
end;


Random.seed!(env::VtolEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::VtolEnv) = env.action_space
RLBase.state_space(env::VtolEnv) = env.observation_space
RLBase.is_terminated(env::VtolEnv) = env.done
RLBase.state(env::VtolEnv) = env.state


function computeReward(env::VtolEnv{A,T}) where {A,T}
    # constants and functions for tuning
    APPROACH_RADIUS = 10 # radius where the drone should transition to hovering
    HOVER_RADIUS = 2.5 # radius where the drone should be hovering
    # gives a increasing gradient towards 0
    shaping_fun = (x, r) -> (1 - (abs(x / r)) ^ 0.4)
    # gives a gradient, but clips loss to zero if far away from target.
    # Used to penalize some metrics only close to target
    masking_fun = (x, r) -> max(0, shaping_fun(x, r))
    
    # extract movement metrics from state_space
    l2_rot_vel = norm(env.state[1:3])
    l2_vel = norm(env.state[4:6])
    l2_dist = norm(env.state[7:9])
    delta_rot = zeros(3,3)
    delta_rot[:, 1] = env.state[10:12]
    delta_rot[:, 2] = env.state[13:15]
    delta_rot[:, 3] = cross(delta_rot[:, 1], delta_rot[:, 2])
    delta_angle = rotation_angle(RotMatrix{3}(delta_rot))
    
    
    # stay alive
    stay_alive_reward = env.t < 2 ? 0.005 : 0.0005
    
    # high reward when very close to target state
    success_reward = 0
    if l2_dist < 0.2
        if l2_vel < 0.05
            success_reward += 0.03
            if delta_angle < 5*pi/180
                success_reward += 2.0
                env.done = true # landed
            end
        end
        success_reward += 0.01
    end
    
    # reward or penalty for being close to target
    translation_reward = shaping_fun(l2_dist, APPROACH_RADIUS) * 0.005
    
    # angle deviation from target
    angle_penalty = delta_angle * masking_fun(l2_dist, HOVER_RADIUS) * 0.01
    
    # deviation from having zero velocity close to target
    vel_penalty = l2_vel * masking_fun(l2_dist, HOVER_RADIUS) * 0.01

    # penalize excessively large rotational speeds (> 1 rad / s)
    rot_vel_penalty = (max(pi, l2_rot_vel) - pi) * 0.001

    # penalize large outputs to suggest energy efficient, stable flight
    output_penalty = 0#mean(env.state[16:19].^2) * 0.0005

    reward = success_reward + stay_alive_reward + translation_reward - angle_penalty - vel_penalty - rot_vel_penalty - output_penalty

    # sum up to part_wise return
    env.success_reward += success_reward
    env.stay_alive_reward += stay_alive_reward
    env.translation_reward += translation_reward
    env.angle_penalty -= angle_penalty
    env.vel_penalty -= vel_penalty
    env.rot_vel_penalty -= rot_vel_penalty
    env.output_penalty -= output_penalty
    env.Return += reward

    return reward
end
RLBase.reward(env::VtolEnv{A,T}) where {A,T} = computeReward(env)


function RLBase.reset!(env::VtolEnv{A,T}) where {A,T}
    # Visualize initial state
    if env.visualization
        Flyonic.Visualization.set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        Flyonic.Visualization.set_actuators(env.name, [0.0; 0.0; 0.0; 0.0])
    end
    
    env.x_W = [-10.0; 0.0; 7.0]
    env.v_B = [5.0; 0.0; 0.0]
    env.R_W = Array{T}(Matrix(RotY(-20*pi/180)))
    env.ω_B = [0.0; 0.0; 0.0]
    env.wind_W = [0.0; 0.0; 0.0]
    v_W = env.R_W * env.v_B

    
    
    # env.x_target = Array{T}([-5.0+rand()*10.0; -5.0+rand()*10.0; 4.0+2.0*rand()]) # x_target
    # env.R_target = Array{T}(Matrix(RotY(-pi/2)*RotX(rand()*2*pi))) # R_target
    env.x_target = Array{T}([0,0,2]) # x_target
    env.R_target = Array{T}(Matrix(RotY(-pi/2))) # R_target
    
    delta_rotation = transpose(env.R_target) * env.R_W
    delta_target = transpose(env.R_W)*(env.x_target - env.x_W)
    # delta_target = env.x_target - env.x_W
    env.state = [
        env.ω_B[1]; # rotation velocity arround body x
        env.ω_B[2]; # rotation velocity arround body y
        env.ω_B[3]; # rotation velocity arround body z
        v_W[1]; # world velocity along x
        v_W[2]; # world velocity along y
        v_W[3]; # world velocity along z
        delta_target[1]; # delta target position along x
        delta_target[2]; # delta target position along y
        delta_target[3]; # delta target position along z
        delta_rotation[1, 1]; # delta rotation x axis in world coordinates (x component)
        delta_rotation[2, 1]; # delta rotation x axis in world coordinates (y component)
        delta_rotation[3, 1]; # delta rotation x axis in world coordinates (z component)
        delta_rotation[1, 2]; # delta rotation z axis in world coordinates (x component)
        delta_rotation[2, 2]; # delta rotation z axis in world coordinates (y component)
        delta_rotation[3, 2]; # delta rotation z axis in world coordinates (z component)
        # env.x_W[1];
        # env.x_W[2];
        # env.x_W[3];
        env.action[1]; # difference in action to last iteration (left thrust)
        env.action[2]; # difference in action to last iteration (right thrust)
        env.action[3]; # difference in action to last iteration (left flap)
        env.action[4] # difference in action to last iteration (right flap)
        ]
    env.t = 0.0
    env.action = rand(env.action_space)
    env.done = false

    env.success_reward = 0.0
    env.stay_alive_reward  = 0.0
    env.translation_reward  = 0.0
    env.angle_penalty = 0.0
    env.vel_penalty = 0.0
    env.rot_vel_penalty = 0.0
    env.output_penalty = 0.0
    env.Return = 0.0

    nothing
end;

function (env::VtolEnv)(a)
    # defines a methods for a callable object.
    # So when a VtolEnv object is created, it has this method that can be called

    # set the propeller trust and the two flaps 2D case

    # enforce a_min < a < a_max
    next_action = [max(range(env.action_space[1])[1], min(a[1], range(env.action_space[1])[end])), 
                   max(range(env.action_space[2])[1], min(a[1], range(env.action_space[2])[end])),
                   max(range(env.action_space[3])[1], min(a[3], range(env.action_space[3])[end])),
                   max(range(env.action_space[4])[1], min(a[3], range(env.action_space[4])[end]))]

    _step!(env, next_action)
end

function _step!(env::VtolEnv, next_action)
    # performs one simulation step, calculates all state transitions and updates the state

    # caluclate wind impact
    v_in_wind_B = Flyonic.RigidBodies.vtol_add_wind(env.v_B, env.R_W, env.wind_W)
    # caluclate aerodynamic forces
    torque_B, force_B = Flyonic.VtolModel.vtol_model(v_in_wind_B, next_action, Flyonic.eth_vtol_param);
    # integrate rigid body dynamics for Δt
    env.x_W, env.v_B, _, env.R_W, env.ω_B, _, env.t = Flyonic.RigidBodies.rigid_body_simple(torque_B, force_B, env.x_W, env.v_B, env.R_W, env.ω_B, env.t, env.Δt, Flyonic.eth_vtol_param)
    

    # Visualize the new state 
    # TODO: Can be removed for real trainings
    if env.visualization
        Flyonic.Visualization.set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        Flyonic.Visualization.set_actuators(env.name, next_action)
    end

    
    if env.realtime
        sleep(env.Δt) # TODO: just a dirty hack. this is of course slower than real time.
    end
 
    env.t += env.Δt
    
    # State space
    env.state[1:3] = env.ω_B # rotation velocity
    v_W = env.R_W * env.v_B
    env.state[4:6] = v_W # world velocity
    env.state[7:9] = transpose(env.R_W) * (env.x_target - env.x_W) # relative target position in body frame
    # env.state[7:9] = env.x_target - env.x_W # relative target position in world frame
    delta_rotation = transpose(env.R_target) * env.R_W
    env.state[10:12] = delta_rotation[:, 1] # delta rotation x axis in world coordinates
    env.state[13:15] = delta_rotation[:, 2] # delta rotation z axis in world coordinates
    env.state[16:19] = next_action

    if eval_mode
        push!(plotting_position_errors, norm(env.x_W - env.x_target))
        push!(plotting_rotation_errors, rotation_angle(RotMatrix{3}(delta_rotation)))
        push!(plotting_actions, next_action)
        push!(plotting_return, env.Return)
    end
    
    # Termination criteria
    env.done =
        #norm(v_B) > 2.0 || # stop if body is to fast
        env.x_W[3] < -0.0 || # stop if body is below 0m
        abs(env.state[7] .> 15) || abs(env.state[8] .> 15) || abs(env.state[9] .> 15) || # Too far from target
        # -pi > rot || # Stop if the drone is pitched 90°.
        # rot > pi/2 || # Stop if the drone is pitched 90°.
        env.t > 15 # stop after 15s
    nothing
end;


seed = 123    
rng = StableRNG(seed)
N_ENV = 12
UPDATE_FREQ = 1024
    
# define multiple environments for parallel training
env = MultiThreadEnv([
    # use different names for the visualization
    VtolEnv(; rng = StableRNG(hash(seed+i)), name = "vtol$i", visualization = false) for i in 1:N_ENV
])



# Define the function approximator
ns, na = length(state(env[1])), length(action_space(env[1]))
approximator = ActorCritic(
    actor = GaussianNetwork(
        pre = Chain(
        Dense(ns, 32, tanh; initW = glorot_uniform(rng)),
        Dense(32, 32, tanh; initW = glorot_uniform(rng)),
        ),
        μ = Dense(32, na; initW = glorot_uniform(rng)),
        logσ = Dense(32, na; initW = glorot_uniform(rng))
    ),
    critic = Chain(
        Dense(ns, 32, tanh; initW = glorot_uniform(rng)),
        Dense(32, 32, tanh; initW = glorot_uniform(rng)),
        Dense(32, 1; initW = glorot_uniform(rng)),
    ),
    optimizer = ADAM(1e-4),
);



agent = Agent( # A wrapper of an AbstractPolicy
    # AbstractPolicy: the policy to use
    policy = PPOPolicy(;
                approximator = approximator |> gpu,
                update_freq=UPDATE_FREQ,
                dist = Normal,
                max_grad_norm = 0.1f0,
                # For parameters visit the docu: https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.PPOPolicy
                ),

    # AbstractTrajectory: used to store transitions between an agent and an environment source
    trajectory = PPOTrajectory(;
        capacity = UPDATE_FREQ,
        state = Matrix{Float64} => (ns, N_ENV),
        action = Matrix{Float64} => (na, N_ENV),
        action_log_prob = Vector{Float64} => (N_ENV,),
        reward = Vector{Float64} => (N_ENV,),
        terminal = Vector{Bool} => (N_ENV,),
    ),
);


function saveModel(t, agent, env)
    model = cpu(agent.policy.approximator)   
    f = joinpath("./src/experiments/exp05_landing2D/RL_models/", "flyonic_landing2D_chkpt_$t.bson")
    @save f model 
    println("parameters at step $t saved to $f")
end

function loadModel()
    f = joinpath("./src/experiments/exp05_landing2D/", "flyonic_landing2D_chkpt_noActionPenalty.bson")
    @load f model
    return model
end

function validate_policy(t, agent, env)
    run(agent.policy, test_env, StopAfterEpisode(1), episode_test_reward_hook)
    # the result of the hook
    println("test reward at step $t: $(episode_test_reward_hook.rewards[end])")
    
end

function show_gradient(t, agent, env)
    v, g = Flux.params(agent.policy.approximator.actor.logσ)

    if(isnan(g[1]))
        fv, fg = Flux.params(agent.policy.approximator.actor.pre)
        println(fg)
        println(fv)
    end
end

episode_test_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
# create a env only for reward test
test_env = VtolEnv(;name = "testVTOL", visualization = true, realtime = true);

# model = loadModel()
# model = Flux.gpu(model)
# agent.policy.approximator = model

eval_mode = true
plotting_position_errors = []
plotting_rotation_errors = []
plotting_actions = []
plotting_return = []

if eval_mode
    model = loadModel()
    model = Flux.gpu(model)
    agent.policy.approximator = model;
    for i = 1:1
        run(
            agent.policy, 
            VtolEnv(;name = "testVTOL", visualization = true, realtime = true), 
            StopAfterEpisode(1), 
            episode_test_reward_hook
        )
    end
else
    model = loadModel()
    model = Flux.gpu(model)
    agent.policy.approximator = model;
    run(
        agent,
        env,
        StopAfterStep(100_000_000),
        ComposedHook(
            DoEveryNStep(saveModel, n=100_000), 
            DoEveryNStep(validate_policy, n=100_000),
            DoEveryNStep(show_gradient, n=1),
            DoEveryNStep(n=5_000) do  t, agent, env
                Base.with_logger(logger) do
                    success_reward = mean([sub_env.success_reward for sub_env in env])
                    stay_alive_reward = mean([sub_env.stay_alive_reward for sub_env in env])
                    translation_reward = mean([sub_env.translation_reward for sub_env in env])
                    angle_penalty = mean([sub_env.angle_penalty for sub_env in env])
                    vel_penalty = mean([sub_env.vel_penalty for sub_env in env])
                    rot_vel_penalty = mean([sub_env.rot_vel_penalty for sub_env in env])
                    output_penalty = mean([sub_env.output_penalty for sub_env in env])                    
                    Return = mean([sub_env.Return for sub_env in env])
                    # l2_dist = mean([norm(sub_env.state[6:9]) for sub_env in env])                  
                    # last_outputs = mean([sub_env.state[16:19] for sub_env in env])
                    @info "reward" success_reward = success_reward
                    @info "reward" stay_alive_reward = stay_alive_reward
                    @info "reward" translation_reward = translation_reward
                    @info "reward" angle_penalty = angle_penalty
                    @info "reward" vel_penalty = vel_penalty
                    @info "reward" rot_vel_penalty = rot_vel_penalty
                    @info "reward" output_penalty = output_penalty
                    @info "reward" Return = Return
                    # @info "metrics" l2_dist = l2_dist
                    # @info "action" last_output1 = last_outputs[1]
                    # @info "action" last_output2 = last_outputs[2]
                    # @info "action" last_output3 = last_outputs[3]
                    # @info "action" last_output4 = last_outputs[4]
                end
            end
        ),
    )
end

if eval_mode
    # transpose action logs
    plotting_actions = [[x[i] for x in plotting_actions] for i in eachindex(plotting_actions[1])]
    x = range(0, length(plotting_position_errors), length(plotting_position_errors))
    p_position_errors = plot(x, plotting_position_errors, ylabel="[m]", title="Position Error")
    p_rotation_errors = plot(x, plotting_rotation_errors.*180/pi, ylabel="[°]", title="Rotation Error")
    p_actions = plot(x, plotting_actions, title="Actions", label=["thrust_L, thrust_R, flap_L, flap_R"], legend=true)
    p_rewards = plot(x, plotting_return, xlabel="time step", title="Return")
    plot(p_position_errors, p_rotation_errors, p_actions, p_rewards, layout=(2,2), legend=false, size=(1200, 500))
end