# The b90 close-encounter cutoff is exact: proof and numerical verification

`regularization-exactness.pdf` (source: `regularization-exactness.tex`,
compiles standalone with `latexmk -pdf regularization-exactness.tex`) proves
the statement referenced in the paper's footnote in sec:point-mass:

> For a point-mass perturber, the ultraviolet divergence of the impulsive
> kick covariance is regularized by the 90-degree-deflection radius
> b_90,i = G(m_i + m_*)/V^2 **with no free constant**: the Coulomb logarithm
> eqn:logLambda and the accompanying constants (-2/3 and -8/3 in Q_r, Q_t)
> are the exact leading-order result in the impulsive limit, with corrections
> suppressed by powers of b_90/r.

Three physical points the proof makes precise: (i) the recoil of the struck
star during the encounter is essential -- pinning the stars would change the
cutoff to G m_i/V^2; (ii) the companion deflects each close trajectory by an
O(1) amount, but this survives the impact-parameter integration only at
O(b_90/r) because it is an area-preserving relabeling (its direct impulse
cancels exactly in each body's kick); (iii) when the binary axis is nearly
aligned with the flight direction, the upstream star acts as a gravitational
lens: an annulus at its Einstein radius sqrt(2 G m_j r)/V focuses onto the
downstream star (a caustic), which dominates the close-approach measure there
but occupies too small a solid angle to affect the orientation averages.

## Verification scripts (self-contained; print PASS/FAIL)

| script | verifies (note section) | runtime |
|---|---|---|
| `exact_deflection_check.py` | the exact two-body kick split, the clean single-body self-energy 2 pi log(Lambda/b90) (Sec. 2), the angular isotropy of the close contribution, and 12-16M-sample Monte Carlos showing exact-deflection vs sharp-b90 differences vanish as a power of b90/r (Sec. 6-7) | ~1 min |
| `three_body_check.py` | the two-body superposition ansatz against genuine (frozen-binary) three-body integrations: worst-case single trajectories and the Monte-Carlo halving test dQ_r -> 0 (Sec. 3-4, 7) | ~45 min |
| `deflection_map_check.py` | the per-trajectory lemmas: recoil vs pinned cutoffs, the incoming deflection map b -> beta* (O(1) shift, area-preserving Jacobian at O(eta)), the osculating-element kick fidelity, the exact cancellation of the companion's direct impulse in the body kick (residual = the O(eta^{3/2}) tidal distortion, not the O(eta) impulse), and the Einstein-radius lensing caustic behind the aligned upstream star (Sec. 3-4) | ~3 min |
| `cutoff_check.py` | the Fourier-space origin of the two per-body cutoffs and the scheme dependence of alternative (sharp-k) regulators (Sec. 2, 5) | ~1 min |

The closed forms these scripts test (the in-plane tensor and the Q_r, Q_t
constants) are independently re-derived symbolically in
`../symbolic/check_kick_covariance.py` and
`../symbolic/check_impulsive_point_mass.py`.
