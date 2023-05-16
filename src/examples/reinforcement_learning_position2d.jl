include("../Flyonic.jl");
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
logger = TBLogger("logs/landing", tb_increment)

Flyonic.Visualization.create_visualization();

mutable struct VtolEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv # Parametric Constructor for a subtype of AbstractEnv
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
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

    close_to_target::T
    not_upright_orientation::T
    not_still::T
    fast_rotation::T

    target_x::T
    target_y::T
    target_z::T
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
            0.0..3.0, # thrust
            -1.0..1.0, # flaps 
            ], 
    )

    
    state_space = Space( # Three continuous values in state space.
        ClosedInterval{T}[
            typemin(T)..typemax(T), # rotation arround y
            typemin(T)..typemax(T), # rotation velocity arround y
            typemin(T)..typemax(T), # world position along x
            typemin(T)..typemax(T), # world position along z
            typemin(T)..typemax(T), # world velocity along x
            typemin(T)..typemax(T), # world velocity along z
            typemin(T)..typemax(T), # target position along x
            typemin(T)..typemax(T), # target position along y
            typemin(T)..typemax(T), # target position along z
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
        zeros(T, 6), # current state, needs to be extended.
        rand(action_space),
        false, # episode done ?
        0.0, # time
        rng, # random number generator  
        name,
        visualization, # visualization
        realtime, # realtime visualization
        Array{T}([0.0; 0.0; 0.0]), # x_W
        Array{T}([0,0,0]), # v_B
        Array{T}(Matrix(Rotations.UnitQuaternion(RotY(-pi/2.0)*RotX(pi)))), # Float64... so T needs to be Float64
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        T(0.025), # Δt  
        0.0, # reward part for close_to_target
        0.0, # reward part for not_upright_orientation
        0.0, # reward part for not_still
        0.0, # reward part for fast_rotation
        rand(Uniform(-10, 10)), # target position x
        0.0, # target position y
        rand(Uniform(5, 10))  # target position z
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
    
    stay_alive = 1.0
    angle_transformed = env.state[1] + pi/2
    if angle_transformed > pi
        angle_transformed -= pi
    end

    success_reward = 0
    
    l2_dist = ((env.state[3] - env.target_x)^2 + (env.state[4] - env.target_z)^2)
    if l2_dist < 0.1
        if abs(env.state[5]) < 0.05 && abs(env.state[6]) < 0.05
            success_reward += 10
        end
        success_reward += 5
    end
    
    close_to_target = (1 - (l2_dist / 10) ^ 0.4) * 0.2
    not_upright_orientation = abs(angle_transformed) * 0.01 * (5 - min(5, abs(env.state[3])))
    not_still = (abs(env.state[5]) + abs(env.state[6])) * 0.005 * (3 - min(3, l2_dist))
    fast_rotation = max(0, (abs(env.state[2]) - pi/6) ^ 2) * 0.005 # 30° per second is admittable
    
    env.close_to_target += close_to_target
    env.not_upright_orientation -= not_upright_orientation
    env.not_still -= not_still
    env.fast_rotation -= fast_rotation

    return stay_alive + success_reward + close_to_target - not_upright_orientation - not_still - fast_rotation
end
RLBase.reward(env::VtolEnv{A,T}) where {A,T} = computeReward(env)

function RLBase.reset!(env::VtolEnv{A,T}) where {A,T}
    # Visualize initial state
    if env.visualization
        Flyonic.Visualization.set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        Flyonic.Visualization.set_actuators(env.name, [0.0; 0.0; 0.0; 0.0])
    end
    
    env.x_W = [0.0; 0.0; 0.0];
    env.v_B = [0.0; 0.0; 0.0];
    env.R_W = Matrix(UnitQuaternion(RotY(-pi/2.0)*RotX(pi)));
    env.ω_B = [0.0; 0.0; 0.0];
    env.wind_W = [0.0; 0.0; 0.0];
    v_W = env.R_W * env.v_B
 
    env.state = [Rotations.params(RotYXZ(env.R_W))[1]; env.ω_B[2]; env.x_W[1]; env.x_W[3]; v_W[1]; v_W[3]; rand(Uniform(-10, 10)); 0.0; rand(Uniform(5, 10))]
    env.t = 0.0
    env.action = [0.0]
    env.done = false


    env.close_to_target = 0.0
    env.not_upright_orientation = 0.0
    env.not_still = 0.0
    env.fast_rotation = 0.0
    nothing
end;

# defines a methods for a callable object.
# So when a VtolEnv object is created, it has this method that can be called
function (env::VtolEnv)(a)

    # set the propeller trust and the two flaps 2D case
    next_action = [a[1], a[1], a[2], a[2]]
   
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
    rot = Rotations.params(RotYXZ(env.R_W))[1]
    env.state[1] = rot # rotation arround y
    env.state[2] = env.ω_B[2] # rotation velocity arround y
    env.state[3] = env.x_W[1] # world position along x
    env.state[4] = env.x_W[3] # world position along z
    v_W = env.R_W * env.v_B
    env.state[5] = v_W[1] # world velocity along x
    env.state[6] = v_W[3] # world velocity along z
    env.state[7] = env.target_x - env.x_W[1] # relative target position along x
    env.state[8] = env.target_y - env.x_W[2] # relative target position along y
    env.state[9] = env.target_z - env.x_W[3] # relative target position along z
    
    angle_transformed = env.state[1] + pi/2
    
    # Termination criteria
    env.done =
        #norm(v_B) > 2.0 || # stop if body is to fast
        env.x_W[3] < -1.0 || # stop if body is below -1m
        abs(env.state[7]) > 15 || abs(env.state[8]) > 15 || abs(env.state[9]) > 15 ||
   
        # -pi > rot || # Stop if the drone is pitched 90°.
        # rot > pi/2 || # Stop if the drone is pitched 90°.
        env.t > 30 # stop after 30s
    nothing
end;



seed = 123    
rng = StableRNG(seed)
N_ENV = 64
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
        Dense(ns, 16, relu; initW = glorot_uniform(rng)),#
        Dense(16, 16, relu; initW = glorot_uniform(rng)),
        ),
        μ = Chain(Dense(16, na; initW = glorot_uniform(rng))),
        logσ = Chain(Dense(16, na; initW = glorot_uniform(rng))),
    ),
    critic = Chain(
        Dense(ns, 16, relu; initW = glorot_uniform(rng)),
        Dense(16, 16, relu; initW = glorot_uniform(rng)),
        Dense(16, 1; initW = glorot_uniform(rng)),
    ),
    optimizer = ADAM(1e-3),
);



agent = Agent( # A wrapper of an AbstractPolicy
    # AbstractPolicy: the policy to use
    policy = PPOPolicy(;
                approximator = approximator |> gpu,
                update_freq=UPDATE_FREQ,
                dist = Normal,
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
    f = joinpath("./src/examples/RL_models_position/", "vtol_ppo_2_rot_cap_$t.bson")
    @save f model
    println("parameters at step $t saved to $f")
end


function loadModel()
    f = joinpath("./src/examples/RL_models_position/", "vtol_ppo_2_1500000.bson")
    @load f model
    return model
end

function validate_policy(t, agent, env)
    run(agent.policy, test_env, StopAfterEpisode(1), episode_test_reward_hook)
    println("Static target: test reward at step $t: $(episode_test_reward_hook.rewards[end])")
    run(agent.policy, VtolEnv(;name = "testVTOL", visualization = true, realtime = true), StopAfterEpisode(1), episode_test_reward_hook)
    # the result of the hook
    println("Random target: test reward at step $t: $(episode_test_reward_hook.rewards[end])")
    
end

episode_test_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
# create a env only for reward test
test_env = VtolEnv(;name = "testVTOL", visualization = true, realtime = true);


eval_mode = true

if eval_mode
    model = loadModel()
    model = Flux.gpu(model)
    agent.policy.approximator = model;
    for i = 1:10
        run(
            agent.policy, 
            VtolEnv(;name = "testVTOL", visualization = true, realtime = true), 
            StopAfterEpisode(1), 
            episode_test_reward_hook
        )
    end
else
    run(
        agent,
        env,
        StopAfterStep(1_500_000),
        ComposedHook(
            DoEveryNStep(saveModel, n=100_000), 
            DoEveryNStep(validate_policy, n=50_000),
            DoEveryNStep(n=1_000) do  t, agent, env
                Base.with_logger(logger) do
                    close_to_target = mean([sub_env.close_to_target for sub_env in env])
                    not_upright_orientation = mean([sub_env.not_upright_orientation for sub_env in env])
                    not_still = mean([sub_env.not_still for sub_env in env])
                    fast_rotation = mean([sub_env.fast_rotation for sub_env in env])
                    @info "reward" close_to_target = close_to_target
                    @info "reward" not_upright_orientation = not_upright_orientation
                    @info "reward" not_still = not_still
                    @info "reward" fast_rotation = fast_rotation
                    @info "reward" total = close_to_target + not_upright_orientation + not_still + fast_rotation 
                end
            end
        ),
    )
end