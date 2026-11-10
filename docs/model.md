# Numerical model

## Governing equation and units

HeatLab models in-plane conduction in a rectangle of width $L_x$, height $L_y$ and thickness $d$.

$$
\rho c_p\frac{\partial T}{\partial t}
  = \nabla\cdot(k\nabla T) + \dot q,
\qquad
\mathbf q'' = -k\nabla T.
$$

The steady equation sets the time derivative to zero. Lengths are in metres, temperatures in kelvin, volumetric generation in W/m³ and flux in W/m². Temperature displays can use °C. Positive prescribed boundary flux points into the solid. Cartesian flux output is signed in the positive $x$ and $y$ directions.

| Condition | Model on each side |
| --- | --- |
| Fixed temperature | $T=T_s$ |
| Prescribed flux | $-\mathbf q''\cdot\mathbf n=q''_{\rm in}$ with outward unit normal $\mathbf n$ |
| Insulated | $q''_{\rm in}=0$ |
| Convection | $q''_{\rm in}=h(T_\infty-T_s)$ |

Properties are constant, homogeneous and isotropic. There is no heat transfer across the two thickness faces. In-plane conditions are uniform along each side and fixed in time. The Python API accepts a spatially varying initial temperature and volumetric source as `(ny, nx)` arrays. The dashboard accepts uniform values for these two inputs. Radiation, contact resistance and variable properties are outside this model.

## Spatial and time discretization

The uniform mesh has one unknown per cell centre. Face area includes thickness $d$. Neighboring cells share equal and opposite conductive face powers. A corner receives two separate boundary-face terms.

| Face | Contribution |
| --- | --- |
| Interior | $G=kA/\Delta$ |
| Fixed temperature | $G=2kA/\Delta$ |
| Convection | $G=A/(\Delta/(2k)+1/h)$ |
| Prescribed flux | $Q_{\rm in}=q''_{\rm in}A$ |

Each cell adds generation $\dot q V$ and has heat capacity $C=\rho c_p V$. The sparse steady operator is $K$. Writing $\theta$ for temperature relative to a numerical offset, the steady system is $K\theta=b$. The backward-Euler update solves

$$
(K + C I/\Delta t)\Delta\theta = b-K\theta_{\rm old}.
$$

One sparse LU factorization is reused across the transient time levels. For a steady case with all Neumann or insulated boundaries, total boundary input plus generation must be zero. A reference mean removes the additive nullspace.

## Conservation and error

The dashboard reports absolute sensible energy relative to 0 K. To keep small changes resolvable, the solver accumulates energy changes from solved temperature increments before rounding the absolute energy. Stepwise power and cumulative energy are audited using the right-endpoint convention of backward Euler.

Errors use matched cell centres. The benchmark reports MAE, RMSE and maximum absolute error against analytical reference fields. Spatial refinement doubles both mesh dimensions and compares colocated values by bilinear interpolation. Temporal refinement holds the grid fixed and compares identical final times. The exact-in-time discrete cosine mode isolates temporal error, while the continuum solution also reveals the remaining spatial error. Acceptance thresholds belong to the documented benchmark.

## Implementation map

- [solver.py](../heatlab/solver.py) assembles the sparse operator and computes temperatures, face powers and energy.
- [visualization.py](../heatlab/visualization.py) handles comparisons, fixed scales and energy histories.
- [scientific_validation.py](../heatlab/scientific_validation.py) runs analytical and refinement studies and audits energy.
- [scientific_reporting.py](../heatlab/scientific_reporting.py) produces the PDF and matching data and figures.

Reference: T. L. Bergman, A. S. Lavine, F. P. Incropera and D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 8th ed., Wiley, §§1.3.1, 2.1, 2.3–2.4, 3.5.1, 4.2, 4.4.3, 4.5.2 and Chapter 5. [Publisher contents](https://catalogimages.wiley.com/images/db/pdf/9781119320432.toc.pdf)
