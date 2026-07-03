"""stochastic-binaries symbolic package (sbx).

Shared machinery for the SymPy verification scripts:

* :mod:`sbx.symbols`     -- shared symbols and conventions
* :mod:`sbx.averaging`   -- fast table-driven orbit averaging
* :mod:`sbx.anomalies`   -- phi(E) substitutions and fixed-phase derivatives
* :mod:`sbx.gauss`       -- Gauss planetary equations and force tensors
* :mod:`sbx.correlators` -- isotropic 4-tensors and contraction helpers
* :mod:`sbx.frames`      -- rotations, body-frame (ehat, qhat, Jhat) transforms
* :mod:`sbx.ito`         -- Ito change-of-variables for (B, D)
* :mod:`sbx.targets`     -- every draft target expression, keyed by \\label
* :mod:`sbx.verify`      -- equality assertions with numeric fallback
"""
