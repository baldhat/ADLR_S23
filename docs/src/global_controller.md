## Global Controller

[Paper - A global controller for flying wing tailsitter vehicles](https://www.flyingmachinearena.ethz.ch/wp-content/uploads/ritzIEEE17.pdf)

[VTOL - Hardware](https://robohub.org/idsc-tailsitter-flying-robot-performs-vertical-loops-and-easily-transitions-between-hover-and-forward-flight/)

[first VTOL Video](https://www.youtube.com/watch?v=JModZfnVAv4), [second VTOL Video](https://www.youtube.com/watch?v=wfmf-eJ89T4)

Implementation only suitable for small attitude errors because of the attitude controller.

!!! warning "only small attitude errors"
    The Attidtue controller presented in the paper is not fully implemented, instead only an approximation for small deviations was used.
    ```math
    \bm{ω}_{des}^\mathcal{B} ≈ - \frac{1}{τ_α} \bm{n}_{err} σ_{err}
    ```