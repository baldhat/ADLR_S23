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

    Return::T
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
            0.0..2.0, # thrust left
            0.0..2.0, # thrust right
            -1.0..1.0, # flaps left
            -1.0..1.0, # flaps right
            ], 
    )

    
    state_space = Space( # Twelve continuous values in state space.
        ClosedInterval{T}[
            typemin(T)..typemax(T) # time
            typemin(T)..typemax(T) # thrust left
            typemin(T)..typemax(T) # thrust right
            typemin(T)..typemax(T) # flaps left
            typemin(T)..typemax(T) # flaps right
            ], 
    )
    
    if visualization
        Flyonic.Visualization.create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
        Flyonic.Visualization.set_transform(name, [0.0; 0.0; 0.0] ,QuatRotation([1 0 0; 0 1 0; 0 0 1]));
        Flyonic.Visualization.set_actuators(name, [0.0; 0.0; 0.0; 0.0])
    end


    environment = VtolEnv(
        action_space,
        state_space,
        zeros(T, 5), # current state, needs to be extended.
        rand(action_space),
        zeros(4), # previous action
        false, # episode done ?
        0.0, # time
        rng, # random number generator  
        name,
        visualization, # visualization
        realtime, # realtime visualization
        
        Array{T}([0;0;0]), # x_W
        Array{T}([0.0; 0.0; 0.0]), # v_B
        Array{T}(Matrix(RotY(-90*pi/180))), # R_W
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        T(0.01), # Δt  

        0.0, # Return
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
    t = env.state[1]
    target_action = [sin(2 * pi * t / 2.) + 1.;
                     cos(2 * pi * t / 2.) + 1.;
                     sin(2 * pi * t / 4.);
                     cos(2 * pi * t / 4.)]
    # target_action = [1.0;
    #                  1.0;
    #                  0.0;
    #                  0.0]
    reward = -sum(abs2, env.state[2:5] - target_action)
    env.Return += reward
    return reward
end
RLBase.reward(env::VtolEnv{A,T}) where {A,T} = computeReward(env)


function RLBase.reset!(env::VtolEnv{A,T}) where {A,T}
    # Visualize initial state
    if env.visualization
        Flyonic.Visualization.set_actuators(env.name, [0.0; 0.0; 0.0; 0.0])
    end
    
    env.t = 0.0
    env.action = rand(env.action_space)
    env.state = [env.t;
                 env.action]
    env.done = false

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
    # Visualize the new state 
    # TODO: Can be removed for real trainings
    if env.visualization
        Flyonic.Visualization.set_actuators(env.name, next_action)
    end
    
    if env.realtime
        sleep(env.Δt) # TODO: just a dirty hack. this is of course slower than real time.
    end
 
    env.t += env.Δt

    env.state[1] = env.t
    env.state[2:5] = next_action

    if eval_mode
        push!(plotting_actions, next_action)
        push!(plotting_return, env.Return)
    end
    
    # Termination criteria
    env.done = env.t > 15 # stop after 15s
    nothing
end;


seed = 123    
rng = StableRNG(seed)
N_ENV = 8
UPDATE_FREQ = 1024
    
# define multiple environments for parallel training
env = MultiThreadEnv([
    # use different names for the visualization
    VtolEnv(; rng = StableRNG(hash(seed+i)), name = "vtol$i", visualization = false, realtime = false) for i in 1:N_ENV
])



# Define the function approximator
ns, na = length(state(env[1])), length(action_space(env[1]))
approximator = ActorCritic(
    actor = GaussianNetwork(
        pre = Chain(
        Dense(ns, 32, relu; initW = glorot_uniform(rng)),
        Dense(32, 32, relu; initW = glorot_uniform(rng)),
        ),
        μ = Dense(32, na; initW = glorot_uniform(rng)),
        logσ = Dense(32, na; initW = glorot_uniform(rng))
    ),
    critic = Chain(
        Dense(ns, 32, relu; initW = glorot_uniform(rng)),
        Dense(32, 32, relu; initW = glorot_uniform(rng)),
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
    f = joinpath("./src/experiments/exp05_landing2D/OF_RL_models/", "flyonic_landing2D_chkpt_$t.bson")
    @save f model 
    println("parameters at step $t saved to $f")
end

function loadModel()
    f = joinpath("S:/Lenny/.UNI/RCI Sem 3/ADL4R/ADLR_S23/src/experiments/exp05_landing2D/OF_RL_models/flyonic_landing2D_chkpt_3500000.bson")
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

model = loadModel()
model = Flux.gpu(model)
agent.policy.approximator = model

eval_mode = false
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
                    @info "reward" Return = mean([sub_env.Return for sub_env in env])
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
    x = range(0, length(plotting_return), length(plotting_return))
    p_actions = plot(x, plotting_actions, title="Actions", label=["thrust_L, thrust_R, flap_L, flap_R"], legend=true)
    p_rewards = plot(x, plotting_return, xlabel="time step", title="Return")
    plot(p_actions, p_rewards, layout=(2,1), legend=false, size=(1200, 500))
end