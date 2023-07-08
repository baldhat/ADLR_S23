include("../../Flyonic.jl");
using .Flyonic;

using Rotations; # used for initial position

using ReinforcementLearning;
using StableRNGs;
using Flux;
using Flux.Losses;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;
using StructArrays;

using Plots;
using Statistics;
using CUDA;

using TensorBoardLogger
using Logging
using BSON: @save, @load # save mode

eval_mode = true
if !eval_mode
    logger = TBLogger("logs/landing3d/new_init_space", tb_increment)
end
print("Starting in eval mode: $eval_mode")

Flyonic.Visualization.create_visualization();

mutable struct VtolEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv # Parametric Constructor for a subtype of AbstractEnv
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    done::Bool
    t::T
    rng::R

    # parameters for the environment
    name::String #for multible environoments
    visualization::Bool
    realtime::Bool # For humans recognizable visualization
    
    # description of evtol and its simulation
    x_W::Vector{T}
    v_B::Vector{T}
    R_W::Matrix{T}
    ω_B::Vector{T}
    wind_W::Vector{T}
    Δt::T

    # setup of landing procedure
    ini_pos_space::Space{Vector{ClosedInterval{T}}}
    ini_rot_space::Space{Vector{ClosedInterval{T}}}
    ini_aoa_space::Space{Vector{ClosedInterval{T}}}
    ini_vel_lb::T
    ini_vel_ub::T
    ini_vel_space::Space{Vector{ClosedInterval{T}}}
    target_pos_space::Space{Vector{ClosedInterval{T}}}
    target_rot_space::Space{Vector{ClosedInterval{T}}}
    target_pos::Vector{T}
    target_rot::Matrix{T}

    # action smoothing
    last_action::Vector{T} # for exponential discount factor
    gamma::T # exponential discount factor

    # logging of (sub-) rewards
    action_rate_penalty::T
    rotation_rate_penalty::T
    stay_alive::T
    distance_reward::T
    inside_cylinder_reward::T
    rotation_reward::T
    slow_descend_reward::T
    landed_reward::T
    Return::T
end


# define a keyword-based constructor for the type declared in the mutable struct typedef. 
# It could also be done with the macro Base.@kwdef.
function VtolEnv(;
     
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
            0.0..2.0, # thrust
            0.0..2.0, # thrust
            -1.0..1.0, # flaps
            -1.0..1.0, # flaps
            ], 
    )

    
    state_space = Space( # Three continuous values in state space.
        ClosedInterval{T}[
            typemin(T)..typemax(T), #  1: delta world position along x to target
            typemin(T)..typemax(T), #  2: delta world position along y to target
            typemin(T)..typemax(T), #  3: delta world position along z to target
            typemin(T)..typemax(T), #  4: W_x_B[1] for delta rotation to target
            typemin(T)..typemax(T), #  5: W_x_B[2] for delta rotation to target
            typemin(T)..typemax(T), #  6: W_x_B[3] for delta rotation to target
            typemin(T)..typemax(T), #  7: W_y_B[1] for delta rotation to target
            typemin(T)..typemax(T), #  8: W_y_B[2] for delta rotation to target
            typemin(T)..typemax(T), #  9: W_y_B[3] for delta rotation to target
            typemin(T)..typemax(T), # 10: rotation velocity arround body x
            typemin(T)..typemax(T), # 11: rotation velocity arround body y
            typemin(T)..typemax(T), # 12: rotation velocity arround body z
            typemin(T)..typemax(T), # 13: velocity along world x
            typemin(T)..typemax(T), # 14: velocity along world y
            typemin(T)..typemax(T), # 15: velocity along world z
            typemin(T)..typemax(T), # 16: previous left thrust output
            typemin(T)..typemax(T), # 17: previous right thrust output
            typemin(T)..typemax(T), # 18: previous left flaps output
            typemin(T)..typemax(T), # 19: previous right flaps output
            ]
    )
    # specifiy sampling ranges
    ini_pos_space = Space(ClosedInterval{T}[ 
        10.5..11.0, # radius
        0.0..deg2rad(360), # angle
        6.5..7.0, # world z
    ])
    ini_rot_space = Space(ClosedInterval{T}[
        -deg2rad(-1)..deg2rad(1) #lobal delta z rotation around heading towards target
    ])
    ini_aoa_space = Space(ClosedInterval{T}[
        deg2rad(10.0)..deg2rad(11.0) #TODO angle of attack in degrees
    ])
    ini_vel_lb = 10.0 # velocity range lower bound in m/s
    ini_vel_ub = 11.0 # velocity range upper bound in m/s
    aoa = rand(ini_aoa_space)[1]
    ini_vel_space = Space(ClosedInterval{T}[
        cos(aoa) * ini_vel_lb..cos(aoa) * ini_vel_ub, # body x
        0.0..0.00001, # body y
        -sin(aoa) * ini_vel_lb.. -sin(aoa) * ini_vel_ub, # body z
    ])
    target_pos_space = Space(ClosedInterval{T}[
        -0.001..0.0, # target x
        -0.001..0.0, # target y
        0.499..0.5, # target z
    ])
    target_rot_space = Space(ClosedInterval{T}[
        -0.001..0.001 # rotation around global z
    ])
    
    # sample spaces to generate random initial conditions
    target_pos = rand(target_pos_space)
    target_rot = Matrix(RotZ(rand(target_rot_space)[1])*RotY(-pi/2))
    ini_pos = rand(ini_pos_space)
    x = ini_pos[1] * cos(ini_pos[2])
    y = ini_pos[1] * sin(ini_pos[2])
    z = ini_pos[3]
    ini_pos = [x; y; z]
    
    if visualization
        Flyonic.Visualization.create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
        Flyonic.Visualization.set_transform(name, ini_pos , QuatRotation(Rotations.UnitQuaternion(RotX(pi))));
        Flyonic.Visualization.set_actuators(name, [0.0; 0.0; 0.0; 0.0])
    end


    environment = VtolEnv(
        action_space,
        state_space,
        zeros(T, 19), # current state, needs to be extended.
        rand(action_space), # current action
        false, # episode done ?
        0.0, # time
        rng, # random number generator  

        name, # name of the environment
        visualization, # visualization
        realtime, # realtime visualization

        zeros(T, 3), # x_W
        zeros(T, 3), # v_B
        Matrix(RotX(pi)), # Float64... so T needs to be Float64
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        # rand(Uniform(0.0, 3.0), 3), # wind_W
        T(0.01), # Δt  

        ini_pos_space, # position space to be sampled
        ini_rot_space, # rotation space to be sampled
        ini_aoa_space, # angle of attack space to be sampled
        ini_vel_ub, # initial velocity upper bound
        ini_vel_lb, # initial velocity lower bound
        ini_vel_space, # velocity space to be sampled
        target_pos_space, # target position space to be sampled
        target_rot_space, # target rotation space to be sampled
        target_pos, # target position
        target_rot, # target rotation

        zeros(T, 4), # last action
        0.9, # gamma for exponential smoothing

        0.0, # penalty for action rates
        0.0, # penalty for rotationi velocity
        0.0, # reward part for stay_alive
        0.0, # penalty part for distance
        0.0, # reward part for being inside cylinder
        0.0, # reward part for being upright
        0.0, # reward part for descending slowly
        0.0, # reward part for having landed
        0.0 # Return
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
    L = 500 # weighting function is smoothed at r/l with a parabola
    weighting_fun_raw = (x, r) -> (1 - (abs(x / r)) ^ 0.4)
    # this weighting function smoothes the raw weighting function inside +-r/L
    # with a parabola to limit the maximum gradient
    weighting_fun = (x, r) ->   if x < -r/L 
                                    weighting_fun_raw(x, r) 
                                elseif x > r/L 
                                    weighting_fun_raw(x, r) 
                                else 
                                    -0.2*x^2*L^1.6/r^2 + 1 - 1/L^0.4 + 0.2*L^1.6/L^2
                                end
    # masking function clips negative values at 0
    masking_fun = (x, r) -> (max(0, weighting_fun(x, r)))

    # reward for staying alive
    stay_alive = 0.05

    # extract movement metrics from state_space
    l2_dist = norm(env.state[1:3])
    # l2_vel = norm(env.state[13:15])
    l2_rot_vel = norm(env.state[10:12])
    delta_rot = zeros(3,3)
    delta_rot[:, 1] = env.state[4:6] / norm(env.state[4:6]) # ensure unit length
    delta_rot[:, 2] = env.state[7:9] / norm(env.state[7:9]) # ensure unit length
    delta_rot[:, 3] = cross(delta_rot[:, 1], delta_rot[:, 2])

    # reward for being close to target, which is reduced, once a certain return
    # threshold is reached. Thisway, in early training, the drone learns to reach
    # the target quickly, but then does not oversaturate the reward.
    distance_reward = weighting_fun(l2_dist, APPROACH_RADIUS) * 0.3
    
    # penalty for high action rates
    action_rate_penalty = norm(env.action - env.last_action) * 1e-2
    env.last_action = env.action # probably there is a better place for this
    
    # penalty for high rotation rates
    rotation_rate_penalty = l2_rot_vel * 3e-3
    
    # model of the landing procedure
    cylinder_radius = 0.5
    cylinder_height = 3.0
    angle_radius = pi/2 # allowed deviation from target angle (= 0.0)
    target_descend_rate = -0.3 # target descend rate
    descend_rate_radius = 0.2 # allowed deviation from target descend rate
    elevation_radius = 0.1 # once being this close to the ground, the drone is considered having landed
    
    inside_cylinder_reward = 0.0
    rotation_reward = 0.0
    slow_descend_reward = 0.0
    landed_reward = 0.0
    
    # reward being inside landing cylifnder
    delta_height = env.state[3]
    delta_radius = norm(env.state[1:2])
    if (0.0 < delta_height < cylinder_height) 
        inside_cylinder_reward = masking_fun(delta_radius, cylinder_radius) * 5.0
    end
    # # REWARDS IN SEQUENCE
    # # reward being correctly oriented towards target
    # if 0.0 < delta_height < cylinder_height && delta_radius < cylinder_radius
    #     delta_angle = rotation_angle(RotMatrix{3}(delta_rot))
    #     rotation_reward = masking_fun(delta_angle, angle_radius) * 2.0
    #     # reward having the right descend rate
    #     if abs(delta_angle) < angle_radius
    #         delta_descend_rate = env.state[15] - target_descend_rate
    #         slow_descend_reward = masking_fun(delta_descend_rate, descend_rate_radius) * 5.0
    #         # reward being close to the ground
    #         if abs(delta_descend_rate) < descend_rate_radius
    #             elevation = env.state[3]
    #             landed_reward = masking_fun(elevation, elevation_radius) * 20.0
    #             if elevation < 0.1 * elevation_radius
    #                 landed_reward += 200.0
    #                 env.done = true
    #             end
    #         end
    #     end
    # end

    # REWARDS IN PARALLEL
    if 0.0 < delta_height < cylinder_height && delta_radius < cylinder_radius
        # reward being correctly oriented towards target
        delta_angle = rotation_angle(RotMatrix{3}(delta_rot))
        # rotation_reward = masking_fun(delta_angle, angle_radius) * 1
        # reward having the right descend rate
        delta_descend_rate = env.state[15] - target_descend_rate
        slow_descend_reward = masking_fun(delta_descend_rate, descend_rate_radius) * 10.0
        # reward being close to the ground
        elevation = env.state[3]
        landed_reward = masking_fun(elevation, elevation_radius) * 5.0
        # detect if landed
        if (0.0 < delta_height < cylinder_height && delta_radius < cylinder_radius) &&
            (abs(delta_angle) < angle_radius) &&
            (abs(delta_descend_rate) < descend_rate_radius) &&
            (elevation < elevation_radius)
            landed_reward += 100.0
            env.done = true
        end
    end

    
    env.stay_alive += stay_alive
    env.distance_reward += distance_reward
    env.inside_cylinder_reward += inside_cylinder_reward
    # env.rotation_reward += rotation_reward 
    env.slow_descend_reward += slow_descend_reward
    env.landed_reward += landed_reward
    env.action_rate_penalty -= action_rate_penalty
    env.rotation_rate_penalty -= rotation_rate_penalty

    reward = stay_alive + distance_reward + inside_cylinder_reward + rotation_reward + slow_descend_reward + landed_reward - action_rate_penalty - rotation_rate_penalty
    env.Return += reward
    
    return reward
end

RLBase.reward(env::VtolEnv{A,T}) where {A,T} = computeReward(env)


function RLBase.reset!(env::VtolEnv{A,T}) where {A,T}
    # sample initial state
    ini_pos = rand(env.ini_pos_space)
    x = ini_pos[1] * cos(ini_pos[2])
    y = ini_pos[1] * sin(ini_pos[2])
    z = ini_pos[3]
    env.x_W = [x; y; z]
    
    aoa = rand(env.ini_aoa_space)[1]
    env.ini_vel_space = Space(ClosedInterval{T}[
        cos(aoa) * env.ini_vel_lb..cos(aoa) * env.ini_vel_ub, # body x
        0.0..0.00001, # body y
        -sin(aoa) * env.ini_vel_lb.. -sin(aoa) * env.ini_vel_ub, # body z
    ])
    env.v_B = rand(env.ini_vel_space);
    # calculate angle between target and drone to initially have the VTOL 
    # point towards the target (with randomisation)
    target_heading = (env.target_pos[1:2] - env.x_W[1:2])
    target_heading /= norm(target_heading)
    angle_to_target = atan(target_heading[2], target_heading[1])
    ini_rot = angle_to_target + rand(env.ini_rot_space)[1]
    env.R_W = Matrix(Rotations.UnitQuaternion(RotZ(ini_rot)*RotY(-aoa)*RotX(pi)))
    # no angular velocity, no wind
    env.ω_B = [0.0; 0.0; 0.0];
    env.wind_W = [0.0; 0.0; 0.0];
    # env.wind_W = rand(Uniform(0.0, 3.0), 3);
    
    # reset tracking variables for rewards
    env.stay_alive = 0.0
    env.inside_cylinder_reward = 0.0
    env.distance_reward = 0.0
    env.rotation_reward = 0.0
    env.slow_descend_reward = 0.0
    env.landed_reward = 0.0
    env.action_rate_penalty = 0.0
    env.rotation_rate_penalty = 0.0
    env.Return = 0.0
    
    env.last_action = zeros(T, 4)
    env.target_pos = rand(env.target_pos_space)
    env.target_rot = Matrix(RotZ(rand(env.target_rot_space)[1])*RotY(-pi/2))
    
    # set initial VTOL state
    delta_rot = transpose(env.R_W) * env.target_rot
    v_W = env.R_W * env.v_B
    env.state = [
        env.x_W - env.target_pos;
        delta_rot[:,1];
        delta_rot[:,2];
        env.ω_B;
        v_W;
        env.last_action;
    ]
    env.t = 0.0
    env.action = rand(env.action_space)
    env.done = false
    
    # Visualize initial state
    if env.visualization
        Flyonic.Visualization.set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        Flyonic.Visualization.set_actuators(env.name, [0.0; 0.0; 0.0; 0.0])
    end
    nothing
end;


# defines a methods for a callable object.
# So when a VtolEnv object is created, it has this method that can be called
function (env::VtolEnv)(a)
    env.action = [a[1], a[2], a[3], a[4]]
    env.action = env.gamma * env.last_action + (1 - env.gamma) * env.action
    # set the propeller trusts and the flaps
    next_action = [env.action[1] + 1, # ranges from 0 to 2 (network predicts in [-1, 1])
                   env.action[2] + 1, # ranges from 0 to 2 (network predicts in [-1, 1])
                   env.action[3], # ranges from -1 to 1 (network predicts in [-1, 1])
                   env.action[4]] # ranges from -1 to 1 (network predicts in [-1, 1])
    
    _step!(env, next_action)
end


function _step!(env::VtolEnv, next_action)
        
    # caluclate wind impact
    v_in_wind_B = Flyonic.RigidBodies.vtol_add_wind(env.v_B, env.R_W, env.wind_W)
    # caluclate aerodynamic forces
    torque_B, force_B = Flyonic.VtolModel.vtol_model(v_in_wind_B, next_action, Flyonic.eth_vtol_param);
    # integrate rigid body dynamics for Δt
    env.x_W, env.v_B, _, env.R_W, env.ω_B, _, env.t = Flyonic.RigidBodies.rigid_body_simple(torque_B, force_B, env.x_W, env.v_B, env.R_W, env.ω_B, env.t, env.Δt, Flyonic.eth_vtol_param)
    

    # Visualize the new state 
    if env.visualization
        Flyonic.Visualization.set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        Flyonic.Visualization.set_actuators(env.name, next_action)
    end
 
    env.t += env.Δt

    if env.realtime
        sleep(env.Δt) # TODO: just a dirty hack. this is of course slower than real time.
    end
    
    # State space
    delta_rot = transpose(env.R_W) * env.target_rot
    v_W = env.R_W * env.v_B
    env.state[1:3] = env.x_W - env.target_pos
    env.state[4:6] = delta_rot[:, 1]
    env.state[7:9] = delta_rot[:, 2]
    env.state[10:12] = env.ω_B
    env.state[13:15] = v_W
    env.state[16:19] = env.last_action
    
    if eval_mode
        push!(plotting_position_errors, norm(env.state[1:3]))	
        push!(plotting_rotation_errors, norm(delta_rot))
        push!(plotting_actions, next_action)
        push!(plotting_return, env.Return)
    end

    # Termination criteria
    env.done =
        env.state[3] < 0.0 || # crashed
        env.t > 20 || # stop after 20 seconds
        env.done
    nothing
end;


seed = 123  
rng = StableRNG(seed)
N_ENV = 12
UPDATE_FREQ = 512

    
# define multiple environments for parallel training
env = MultiThreadEnv([
    # use different names for the visualization
    VtolEnv(; rng = StableRNG(hash(seed+i)), name = "vtol$i") for i in 1:N_ENV
])


# Define the function approximator
ns, na = length(state(env[1])), length(action_space(env[1]))
approximator = ActorCritic(
    actor = GaussianNetwork(
        pre = Chain(
        Dense(ns, 32, relu; initW = glorot_uniform(rng)),
        Dense(32, 32, relu; initW = glorot_uniform(rng)),
        ),
        μ = Chain(Dense(32, na, tanh; initW = glorot_uniform(rng))),
        logσ = Chain(Dense(32, na, tanh; initW = glorot_uniform(rng))),
        max_σ = Float32(1_000_000.0)
    ),
    critic = Chain(
        Dense(ns, 32, relu; initW = glorot_uniform(rng)),
        Dense(32, 16, relu; initW = glorot_uniform(rng)),
        Dense(16, 1; initW = glorot_uniform(rng)),
    ),
    optimizer = ADAM(5e-4),
);

# Define the agent
agent = Agent( # A wrapper of an AbstractPolicy
    # AbstractPolicy: the policy to use
    policy = PPOPolicy(;
                approximator = approximator |> gpu,
                update_freq=UPDATE_FREQ,
                dist = Normal,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.03f0,
                clip_range= 0.15f0
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

# callback functions for training
function saveModel(t, agent, env)
    model = cpu(agent.policy.approximator)   
    f = joinpath("./src/experiments/exp06_landing3D/runs/", "landing3D_$t.bson")
    @save f model
    println("parameters at step $t saved to $f")
end

function loadModel()
    # f = joinpath("./src/experiments/exp06_landing3D/runs/landing3D_59800000.bson")
    f = joinpath("./src/experiments/exp06_landing3D/runs/landing3D_14200000.bson")
    @load f model
    return model
end

function validate_policy(t, agent, env)
    run(agent.policy, test_env, StopAfterEpisode(1), episode_test_reward_hook)
    # the result of the hook
    println("\nTest reward at step $t: $(episode_test_reward_hook.rewards[end])")
end;

episode_test_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
# create a env only for reward test
test_env = VtolEnv(;name = "testVTOL", visualization = true, realtime = true);

agent.policy.approximator = loadModel()|>gpu;

plotting_position_errors = []
plotting_rotation_errors = []
plotting_actions = []
plotting_return = []



if eval_mode
    # Override the prob function to set the sigma to almost zero during evaluation
    function RLBase.prob(
        p::PPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
        state::AbstractArray,
        mask,
    )
        if p.update_step < p.n_random_start
            @error "todo"
        else
            μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host 
            logσ = log.(ones(size(logσ)) * 0.0000001)  # Uncomment during test time
            StructArray{Normal}((μ, exp.(logσ)))
        end
    end

    model = loadModel()
    model = Flux.gpu(model)
    agent.policy.approximator = model;
    
    for i = 1:1
        run(
            agent.policy, 
            VtolEnv(;name = "evalVTOL", visualization = true, realtime = true), 
            StopAfterEpisode(1), 
            episode_test_reward_hook
        )
    end
else
    function RLBase.prob(
        p::PPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
        state::AbstractArray,
        mask,
    )
        if p.update_step < p.n_random_start
            @error "todo"
        else
            μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host
            StructArray{Normal}((μ, exp.(logσ)))
        end
    end
    run(
        agent,
        env,
        StopAfterStep(100_000_000),
        ComposedHook(
            DoEveryNStep(saveModel, n=100_000), 
            DoEveryNStep(validate_policy, n=100_000),
            DoEveryNStep(n=4_096) do  t, agent, env
                Base.with_logger(logger) do
                    @info "reward" action_rate_penalty = mean([sub_env.action_rate_penalty for sub_env in env])
                    @info "reward" rotation_rate_penalty = mean([sub_env.rotation_rate_penalty for sub_env in env])
                    @info "reward" stay_alive = mean([sub_env.stay_alive for sub_env in env])
                    @info "reward" distance_reward = mean([sub_env.distance_reward for sub_env in env])
                    @info "reward" inside_cylinder_reward = mean([sub_env.inside_cylinder_reward for sub_env in env])
                    @info "reward" rotation_reward = mean([sub_env.rotation_reward for sub_env in env])
                    @info "reward" slow_descend_reward = mean([sub_env.slow_descend_reward for sub_env in env])
                    @info "reward" landed_reward = mean([sub_env.landed_reward for sub_env in env])
                    @info "reward" Return = mean([sub_env.Return for sub_env in env])
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