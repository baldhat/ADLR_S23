# Examples

## velocity_GlobalControll.ipynb
A simple example showing the interface to the drone model and the rigid body dynamic. The controller used is the one presented by Robin Ritz and Raffaello D’Andrea in paper "[A global controller for flying wing tailsitter vehicles](https://www.flyingmachinearena.ethz.ch/wp-content/uploads/ritzIEEE17.pdf)" in 2017. The controller is not suitable for large attitude deviations. Since it is not fully implemented.

```@raw html
<div class="container" style="width: 100%; height: 0; padding-bottom: 56.25%;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-ZuRPlvRdDM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen></iframe>
</div>
```

## reinforcement_learning.ipynb
Shows an example of how the environment can be used for learning. A constant thrust is set and both flaps are controlled identically. The drone should stay in the air as long as possible. The code is taken from the [JuliaRL_BasicDQN_CartPole experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/DQN/JuliaRL_BasicDQN_CartPole/#JuliaRL\\_BasicDQN\\_CartPole).

The [Julia Reinforcement Learning](https://juliareinforcementlearning.org/docs/) package is used to achieve fast results. A good introduction can be found [here](https://juliareinforcementlearning.org/blog/an_introduction_to_reinforcement_learning_jl_design_implementations_thoughts/).

There is ```reinforcement_learning_simple_mechanics.ipynb``` and ```reinforcement_learning.ipynb```. 


```@raw html
<div class="container" style="width: 100%; height: 0; padding-bottom: 56.25%;">
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/mU562NhUQcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen></iframe>
</div>
```
VTOL learning curve.
![learning curve](../assets/examples/learning_curve.png)


## 2d\_optimization\_problem.ipynb

This example contains a reduced 2-dimensional optimisation problem on which the loss landscape of the VTOL model can be viewed. It can also be used to try out different loss functions.

Zygote is used to calculate a gradient. The gradient is used to find a minimum by conjugate gradient descent.

A fixed setting for thrust and flaps was set for the entire duration of 0.2 s.
The aim was to keep the drone as motionless as possible at the start position and to find the best parameters (thrust and flaps) for this. For this purpose, the
deviation of the position, speed, rotation speed and orientation from the target values was used as loss. The actuators values were also taken into account.

![2D optimization loss landscape](../assets/examples/2d_optimization_problem.png)


## model\_predictive\_control.ipynb

Model Predictive Control Example

It is applied from support point to support point. The controller is about 10 times slower than real time.

```@raw html
<div class="container" style="width: 100%; height: 0; padding-bottom: 56.25%;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/iceSWRD9V78" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen></iframe>
</div>
```


## trajectory\_optimization.ipynb

A starting and an ending state is given. In between, the actuator inputs are optimised using a loss function and a conjugate gradient method.

Loss Function
```julia
loss += norm(x_W - x_W_desired)*10.0
loss += norm(v_B - v_B_desired) 
loss += norm(ω_B - ω_B_desired)/100.0
    
rotation_error = transpose(R_W) * R_W_desired
rot_ang = 3.0 - rotation_error[1,1] - rotation_error[2,2] - rotation_error[3,3]
loss += max(0.0, rot_ang)*2.0
```


Optimisation for 300 time steps.
```
 * Status: success

 * Candidate solution
    Final objective value:     3.444486e+00

 * Found with
    Algorithm:     Conjugate Gradient

 * Convergence measures
    |x - x'|               = 0.00e+00 ≤ 0.0e+00
    |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 2.40e+02 ≰ 1.0e-08

 * Work counters
    Seconds run:   30  (vs limit 800)
    Iterations:    175
    f(x) calls:    406
    ∇f(x) calls:   264
```


```@raw html
<div class="container" style="width: 100%; height: 0; padding-bottom: 56.25%;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/QDeZBX-MfSI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen></iframe>
</div>
```
