"""Recreation of Loquercio et al. 2021, *Learning high-speed flight in the wild* (Sci. Robot. 6:abg5810).

Pieces (paper section → module):

* simulated stereo camera + Semi-Global Matching depth (Methods C)      → ``stereo``
* training / test environments: Poisson forests, convex shapes, narrow gap → ``envs``
* privileged expert: global collision-free reference + Metropolis-Hastings
  sampling of cubic B-splines on the true geometry (Methods A, S6)        → ``expert``
* trajectories: B-splines, order-5 polynomial projection, input cost      → ``trajectory``
* student policy: MobileNet-V3 + MLPs, M = 3 hypotheses, R-WTA loss (B)   → ``policy``
* closed loop, DAgger and evaluation                                     → ``rollout``, ``dagger``

See docs/research/R18_agile_autonomy.md for what is faithful to the paper and what is not.
"""
