# Execute Julia script from the command line

Find out which Julia installation you have.
```
$ which julia
> /usr/local/bin/julia
```

Compile and run programme with four threads
```
$ /usr/local/bin/julia --threads=4 ./src/examples/your_script.jl
```

## Or use [tmux](https://tmuxguide.readthedocs.io/en/latest/tmux/tmux.html) to run in the background

start tmux
```
$ tmux new -s vtol
```

detached-session
```
ctr - b
d
```

attach the detached-session
```
$ tmux a -t vtol
```


vertical split windos
```
ctrl - b
%
```