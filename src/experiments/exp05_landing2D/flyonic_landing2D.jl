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

using Plots;
using Statistics;
using CUDA;

using TensorBoardLogger
using Logging
using BSON: @save, @load # save mode

logger = TBLogger("logs/landing2d/landing", tb_increment)

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

    target::Vector{T}

    stay_alive::T
    distance_reward::T
    success_reward::T
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
            typemin(T)..typemax(T), # target_position along z
            ], 
    )
    
    if visualization
        Flyonic.Visualization.create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
        Flyonic.Visualization.set_transform(name, [-10.0; 0.0; 5.0] , QuatRotation(Rotations.UnitQuaternion(RotX(pi))));
        Flyonic.Visualization.set_actuators(name, [0.0; 0.0; 0.0; 0.0])
    end


    environment = VtolEnv(
        action_space,
        state_space,
        zeros(T, 8), # current state, needs to be extended.
        rand(action_space),
        false, # episode done ?
        0.0, # time
        rng, # random number generator  
        name,
        visualization, # visualization
        realtime, # realtime visualization
        Array{T}([-10.0; 0.0; 5.0]), # x_W
        Array{T}([4.0,0,0]), # v_B
        Matrix(Rotations.UnitQuaternion(RotX(pi))), # Float64... so T needs to be Float64
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        T(0.01), # Δt  
        [0.0; 0.5],# target position
        0.0, # reward part for stay_alive
        0.0, # penalty part for distance
        0.0, # reward part for success
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
    APPROACH_RADIUS = 5 # radius where the drone should transition to hovering
    L = 100 # weighting function is smoothed at r/l with a parabola
    weighting_fun_raw = (x, r) -> (1 - (abs(x / r)) ^ 0.4)
    weighting_fun = (x, r) ->   if x < -r/L 
                                    weighting_fun_raw(x, r) 
                                elseif x > r/L 
                                    weighting_fun_raw(x, r) 
                                else 
                                    -0.2*x^2*L^1.6/r^2 + 1 - 1/L^0.4 + 0.2*L^1.6/L^2
                                end
    masking_fun = (x, r) -> (max(0, weighting_fun(x, r)))

    stay_alive = 0.05
    delta_angle = env.state[1] - pi/2
    if delta_angle > pi
        delta_angle -= pi
    elseif delta_angle < -pi
        delta_angle += pi
    end

    distance_reward = weighting_fun(norm(env.state[3:4] - env.state[7:8]), APPROACH_RADIUS) * 0.2

    # model of the landing procedure
    success_reward = 0.0
    cylinder_radius = 0.5
    cylinder_height = 3.0
    angle_radius = 0.5 # allowed deviation from target angle (= 0.0)
    target_descend_rate = -0.3
    descend_rate_radius = 0.1 # allowed deviation from target descend rate
    elevation_radius = 0.05 # once being this close to the ground, the drone is considered having landed
    # drone is inside cylinder
    delta_height = env.state[4] - env.state[8]
    delta_radius = norm(env.state[3:3] - env.state[7:7])
    # reward being inside landing cylinder
    if (0.0 < delta_height && delta_height < cylinder_height) 
        success_reward += masking_fun(delta_radius, cylinder_radius) * 0.2
    end
    # reward being upright
    if delta_height < cylinder_height && delta_radius < cylinder_radius
        success_reward += masking_fun(delta_angle, angle_radius) * 0.3
        # reward having the right descend rate
        if abs(delta_angle) < angle_radius
            delta_descend_rate = env.state[6] - target_descend_rate
            success_reward += masking_fun(delta_descend_rate, descend_rate_radius) * 0.4
            # reward being close to the ground
            if abs(delta_descend_rate) < descend_rate_radius
                elevation = env.state[4]
                if elevation < elevation_radius
                    success_reward += 20.0
                    env.done = true
                end
            end
        end
    end

    env.stay_alive += stay_alive
    env.distance_reward += distance_reward
    env.success_reward += success_reward
    reward = stay_alive + distance_reward + success_reward
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
    
    env.x_W = [-10.0; 0.0; 5.0];
    env.v_B = [4.0; 0.0; 0.0];
    env.R_W = Matrix(Rotations.UnitQuaternion(RotX(pi)))
    env.ω_B = [0.0; 0.0; 0.0];
    env.wind_W = [0.0; 0.0; 0.0];
    v_W = env.R_W * env.v_B

    env.stay_alive = 0.0
    env.distance_reward = 0.0
    env.success_reward = 0.0

    env.target = [0.0; 0.5]
 
    env.state = [Rotations.params(RotYXZ(env.R_W))[1]; env.ω_B[2]; env.x_W[1]; env.x_W[3]; v_W[1]; v_W[3]; env.target[1]; env.target[2]]
    env.t = 0.0
    env.action = [0.0]
    env.done = false
    nothing
end;


# defines a methods for a callable object.
# So when a VtolEnv object is created, it has this method that can be called
function (env::VtolEnv)(a)

    # set the propeller trust and the two flaps 2D case
    next_action = [max(range(env.action_space[1])[1], min(range(env.action_space[1])[end],  a[1])), 
                   max(range(env.action_space[1])[1], min(range(env.action_space[1])[end],  a[1])),
                   max(range(env.action_space[2])[1], min(range(env.action_space[2])[end],  a[2])),
                   max(range(env.action_space[2])[1], min(range(env.action_space[2])[end],  a[2]))]

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
 
    env.t += env.Δt

    if env.realtime
        sleep(env.Δt) # TODO: just a dirty hack. this is of course slower than real time.
    end
    
    # State space
    rot = Rotations.params(RotYXZ(env.R_W))[1]
    env.state[1] = rot # rotation arround y
    env.state[2] = env.ω_B[2] # rotation velocity arround y
    env.state[3] = env.x_W[1] # world position along x
    env.state[4] = env.x_W[3] # world position along z
    v_W = env.R_W * env.v_B
    env.state[5] = v_W[1] # velocity along world x
    env.state[6] = v_W[3] # velocity along world z
    env.state[7] = env.target[1]
    env.state[8] = env.target[2]
    
    if eval_mode
        push!(plotting_position_errors, norm(env.state[3:4] - env.state[7:8]))	
        push!(plotting_rotation_errors, rot)
        push!(plotting_actions, next_action)
        push!(plotting_return, env.Return)
    end

    # Termination criteria
    env.done =
        env.state[4] < 0.0 || # crashed
        env.t > 15 # stop after 15 seconds
    nothing
end;



seed = 123    
rng = StableRNG(seed)
N_ENV = 16
UPDATE_FREQ = 1024
    
    
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
        Dense(ns, 16, relu; initW = glorot_uniform(rng)),
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
    f = joinpath("./src/experiments/exp05_landing2D/runs/", "vtol_ppo_2_$t.bson")
    @save f model
    println("parameters at step $t saved to $f")
end


function loadModel()
    f = joinpath("S:/Lenny/.UNI/RCI Sem 3/ADL4R/ADLR_S23/src/experiments/exp05_landing2D/runs/vtol_ppo_2_6000000.bson")
    @load f model
    return model
end

function validate_policy(t, agent, env)
    run(agent.policy, test_env, StopAfterEpisode(1), episode_test_reward_hook)
    # the result of the hook
    println("\nTest reward at step $t: $(episode_test_reward_hook.rewards[end])")
    println(test_env.state[1] - pi/2)
    
end;

episode_test_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
# create a env only for reward test
test_env = VtolEnv(;name = "testVTOL", visualization = true, realtime = true);

# agent.policy.approximator = loadModel();

eval_mode = false
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
            VtolEnv(;name = "evalVTOL", visualization = true, realtime = true), 
            StopAfterEpisode(5), 
            episode_test_reward_hook
        )
    end
else
    run(
        agent,
        env,
        StopAfterStep(30_000_000),
        ComposedHook(
            DoEveryNStep(saveModel, n=500_000), 
            DoEveryNStep(validate_policy, n=50_000),
            DoEveryNStep(n=3_000) do  t, agent, env
                Base.with_logger(logger) do
                    stay_alive = mean([sub_env.stay_alive for sub_env in env])
                    distance_reward = mean([sub_env.distance_reward for sub_env in env])
                    success_reward = mean([sub_env.success_reward for sub_env in env])
                    @info "reward" stay_alive = stay_alive
                    @info "reward" distance_reward = distance_reward
                    @info "reward" success_reward = success_reward
                    @info "reward" total = stay_alive + distance_reward + success_reward
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