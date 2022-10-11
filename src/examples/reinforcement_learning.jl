include("../Flyonic.jl");
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

using BSON: @save, @load # save mode


create_visualization();

mutable struct VtolEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv # Parametric Constructor for a subtype of AbstractEnv
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    done::Bool
    t::T
    rng::R

    name::String #for multible environoments
    
    # Everything you need aditionaly can also go in here.
    x_W::Vector{T}
    v_B::Vector{T}
    R_W::Matrix{T}
    ω_B::Vector{T}
    wind_W::Vector{T}
    Δt::T
end


# define a keyword-based constructor for the type declared in the mutable struct typedef. 
# It could also be done with the macro Base.@kwdef.
function VtolEnv(;
     
    #continuous = true,
    rng = Random.GLOBAL_RNG, # Random number generation
    name = "vtol",
    kwargs... # let the function take an arbitrary number of keyword arguments 
)
    
    T = Float64; # explicit type which is used e.g. in state. Cannot be altered due to the poor matrix defininon.

    #action_space = Base.OneTo(21) # 21 discrete positions for the flaps
    
    action_space = Space(
        ClosedInterval{T}[
            0.0..2.0, # thrust
            -1.0..1.0, # flaps
            ], 
    )

    
    state_space = Space( # Three continuous values in state space.
        ClosedInterval{T}[
            typemin(T)..typemax(T), # rotation arround y
            typemin(T)..typemax(T), # rotation velocity arround y
            typemin(T)..typemax(T), # world position along x
            typemin(T)..typemax(T), # world position along z
            ], 
    )
    
    create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
    set_transform(name, [0.0; 0.0; 0.0] ,QuatRotation(UnitQuaternion(RotY(-pi/2.0)*RotX(pi))));
    set_actuators(name, [0.0; 0.0; 0.0; 0.0])


    environment = VtolEnv(
        action_space,
        state_space,
        zeros(T, 3), # current state, needs to be extended.
        rand(action_space),
        false, # episode done ?
        0.0, # time
        rng, # random number generator  
        name,
        zeros(T, 3), # x_W
        zeros(T, 3), # v_B
        Matrix(UnitQuaternion(RotY(-pi/2.0)*RotX(pi))), # Float64... so T needs to be Float64
        zeros(T, 3), # ω_B
        zeros(T, 3), # wind_W
        T(0.025), # Δt  
    )
    
    
    reset!(environment)
    
    return environment
    
end;


Random.seed!(env::VtolEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::VtolEnv) = env.action_space
RLBase.state_space(env::VtolEnv) = env.observation_space
RLBase.is_terminated(env::VtolEnv) = env.done
RLBase.state(env::VtolEnv) = env.state


function computeReward(env::VtolEnv{A,T}) where {A,T}
    
    stay_alive = 3.0
    not_upright_orientation = abs(env.state[1]-pi*0.5)*10.0
    not_centered_position = abs(env.state[3])*10.0
    hight = env.state[4]*100.0
    
    return stay_alive - not_upright_orientation - not_centered_position + hight
end


RLBase.reward(env::VtolEnv{A,T}) where {A,T} = computeReward(env)


function RLBase.reset!(env::VtolEnv{A,T}) where {A,T}
    
    # Visualize initial state
    set_transform(env.name, env.x_W,QuatRotation(env.R_W));
    set_actuators(env.name, [0.0; 0.0; 0.0; 0.0])
    
    env.x_W = [0.0; 0.0; 0.0];
    env.v_B = [0.0; 0.0; 0.0];
    env.R_W = Matrix(UnitQuaternion(RotY(-pi/2.0)*RotX(pi)));
    env.ω_B = [0.0; 0.0; 0.0];
    env.wind_W = [0.0; 0.0; 0.0];

 
    env.state = [env.ω_B[2]; Rotations.params(RotYXZ(env.R_W))[1]; env.x_W[1]; env.x_W[3]]
    env.t = 0.0
    env.action = [0.0]
    env.done = false
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
    v_in_wind_B = vtol_add_wind(env.v_B, env.R_W, env.wind_W)
    # caluclate aerodynamic forces
    torque_B, force_B = vtol_model(v_in_wind_B, next_action, eth_vtol_param);
    # integrate rigid body dynamics for Δt
    env.x_W, env.v_B, env.R_W, env.ω_B, time = rigid_body_simple(torque_B, force_B, env.x_W, env.v_B, env.R_W, env.ω_B, env.t, env.Δt, eth_vtol_param)


    # Visualize the new state 
    # TODO: Can be removed for real trainings
    set_transform(env.name, env.x_W, QuatRotation(env.R_W));
    set_actuators(env.name, next_action)
 
    env.t += env.Δt
    
    # State space
    rot = Rotations.params(RotYXZ(env.R_W))[1]
    env.state[1] = rot # rotation arround y
    env.state[2] = env.ω_B[2] # rotation velocity arround y
    env.state[3] = env.x_W[1] # world position along x
    env.state[4] = env.x_W[3] # world position along z
    
    
    # Termination criteria
    env.done =
        #norm(v_B) > 2.0 || # stop if body is to fast
        env.x_W[3] < -1.0 || # stop if body is below -1m
        0.0 > rot || # Stop if the drone is pitched 90°.
        rot > pi || # Stop if the drone is pitched 90°.
        env.t > 10 # stop after 10s
    nothing
end;



seed = 123    
rng = StableRNG(seed)
    N_ENV = 8
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
        f = joinpath("./RL_models/", "vtol_ppo_2_$t.bson")
        @save f model
        println("parameters at step $t saved to $f")
    end


    function loadModel()
        f = joinpath("./src/examples/RL_models/", "vtol_ppo_2_9320000.bson")
        @load f model
        return model
    end



    agent.policy.approximator = loadModel();


    run(
           agent,
           env,
           StopAfterStep(100_000),
           DoEveryNStep(saveModel, n=40_000)
       )