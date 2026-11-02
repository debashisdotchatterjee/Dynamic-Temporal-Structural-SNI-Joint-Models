# Landmark-temporal structural joint models: computational study

**Purpose.** This package is the *computational phase only* for a research manuscript on continuous temporal residual dependence and skew-normal/independent (SNI) joint models of longitudinal biomarkers and survival. The manuscript itself is intentionally **not rewritten** here; the empirical results should be inspected before results, discussion, and theory are finalized.

## Files delivered

- `HJ_Joint_Model_Colab.ipynb`: **single-file Google Colab notebook** with the complete executable Python code embedded; open it in Colab, select Runtime → Run all. No separate `.py` upload is needed.
- `hj_joint_study.py`: equivalent standalone Python implementation (`python hj_joint_study.py`).
- `README.md`: methodological and reproducibility details.
- `requirements.txt`: Python dependencies.

The notebook automatically **downloads a ZIP** named `HJ_structural_joint_outputs.zip`. The ZIP includes publication-oriented PDF/PNG figures, CSV and LaTeX tables, held-out individual-level risk predictions, a copy of the actual public real-data source, parameter estimates, optimizer diagnostics, dataset provenance/checksum, exact source code, environment versions, and a status file. Results are not shipped as precomputed or manufactured values.

## Run in Colab

1. Open `HJ_Joint_Model_Colab.ipynb` at <https://colab.research.google.com/> (File → Upload notebook).
2. Execute all cells in order. The notebook installs common Python libraries if needed and executes the included script.
3. It downloads a **genuine public clinical dataset**, performs model fitting, simulations, validation checks, produces figures/tables, then triggers the output ZIP download.
4. If a network firewall blocks the data source, the program **fails explicitly for real data** and logs the error; it does not silently create fake empirical results. Upload `pbcseq.csv` to the current Colab working directory and rerun if necessary. The simulation study is independent of the external download.

The source is the 1,945-row sequential Mayo Clinic primary biliary cirrhosis dataset (`survival::pbcseq`, 312 patients):

- Primary: https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv
- Mirror: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbcseq.csv
- Secondary repository mirror: https://github.com/TheAxeC/cirrhosis-competing-risks/blob/main/data/pbcseq.csv
- Dataset documentation: https://search.r-project.org/CRAN/refmans/survival/html/pbcseq.html
- Clinical joint-model precedent: https://drizopoulos.github.io/JMbayes2/

## Why the manuscript's current likelihood requires adjustment

The uploaded manuscript proposes a conditional covariance `c_j(T*) = κ exp{-λ(T*−t_j)}` where **the same random event time** `T*` is part of the joint Gaussian vector. Such a self-referential, event-dependent Gaussian covariance does not, without an appropriate normalizing construction, specify the claimed Gaussian conditional distribution. In addition, the original minimum-spacing theorem needs a strictly positive lower bound on inter-visit spacing; simply adding exponential decay does **not** guarantee positive definiteness when visit density diverges in a finite time interval. Likewise, an expression `Normal(m(T*),s²(T*))` is generally not an ordinary Gaussian distribution and its right-tail probability is not the naïve Gaussian CDF.

For the computational study, the structural model is *prospective at a prespecified fixed landmark* `ell`, so that only times `t_ij <= ell` enter the predictor. Define

`k_ij = exp(-lambda*(ell-t_ij))` and `w_ij = k_ij / sqrt(sum_j k_ij^2)`.

Let `b_i` be random intercept and slope, `e_ij` measurement innovations, `V_i` residual scale, `U_i` random-effect scale, and `H_i` standard half-normal. The exact generative specification is

```
b_i = U_i^(-1/2) * (Delta * H_i + G_i) - a * Delta
G_i ~ N_2(0, Gamma),   a = E[U^(-1/2)] sqrt(2/pi)
Y_ij = beta0 + beta1*t_ij + beta_sex*sex_i + beta_trt*trt_i
       + b_i0 + b_i1*t_ij + e_ij
e_ij | V_i ~ N(0, sigma_e^2/V_i), independently over j
L_i = log(T_i* - ell)
    = alpha0 + alpha_age*age_i + alpha_sex*sex_i + alpha_trt*trt_i
      + nu*b_i1 + (sigma_T*rho/sigma_e)*sum_j w_ij*e_ij
      + sigma_T*sqrt(1-rho^2)*Z_i/sqrt(V_i)
Z_i ~ N(0,1)
```

with independent `U_i`, `V_i`, `H_i`, `G_i`, `Z_i`. In the Gaussian models `U_i=V_i=1, Delta=0`; in the full SNI model `U_i~Gamma(3,3)` and independently `V_i~Gamma(3,3)`, with degrees of freedom fixed at 6 for feasibility. The Gaussian core is **analytically marginalized over all subject-specific random intercepts and slopes**, not estimated in a two-stage mixed-model plug-in. The SNI extension uses Gaussian analytical integration conditional on `H,U,V` and deterministic Gaussian-Legendre quadrature to approximate the remaining low-dimensional integrals. Thus, the full SNI fit is a **quadrature-approximated marginal joint likelihood**, *not MCMC*; the eventual paper must describe the inference actually performed.

Conditional on `V_i=v`, the residual covariance between the longitudinal innovation vector and the AFT innovation has block form

```
Sigma_i(v) = v^(-1) * [[sigma_e^2 I, sigma_e*sigma_T*rho*w],
                        [sigma_e*sigma_T*rho*w^T, sigma_T^2]]
```

Because `||w||²=1`, the Schur complement equals `sigma_T²*(1-rho²)>0` whenever `|rho|<1`. This holds for **every n>=1**, even increasingly dense observation schedules, and is directly tested numerically against the original unnormalized flat and exponential kernels.

## Real-data design

- Longitudinal marker: `log(bili)` (natural log of serum bilirubin in mg/dL).
- Event outcome: **composite death or transplantation** (`status` 1 or 2), with administrative/lost follow-up (`status` 0) treated as right censored. This avoids incorrectly treating transplant as noninformative censoring in a death-only analysis.
- Default landmark: **two years after enrollment**. Only patients alive/under follow-up after this date, with **at least two distinct pre-landmark recorded bilirubin values**, are eligible. All post-landmark biomarker data are excluded from the predictors. Follow-up is analyzed as positive residual survival time `T*−ell`.
- Random subject train/test splits, stratified by event status; no patient appears in both sets in one split.
- Longitudinal and event model parameters fit only on training subjects. Age scaling is fixed a priori as `(age−50)/10` to prevent test-set preprocessing leakage.
- Primary evaluation: **held-out conditional event-time log score**; complementary scores: proper joint log score, IPCW Brier at 2 and 4 years, and Harrell C-index. Censoring weights are estimated from **training** subjects only.
- Paired subject-bootstrap comparisons of held-out conditional log score; descriptive uncertainty, not a substitute for external validation.
- Landmark sensitivity at 1 and 3 years.

These are **conditional landmark joint models**, not unconditional original-enrollment time-to-event joint models. Clinical interpretation is restricted to patients eligible at the landmark. The noninformative censoring and observation-schedule assumptions require scrutiny, and informative clinical visit scheduling may affect generalization.

## Four prespecified model comparisons

1. `Shared-Gaussian`: conventional shared random-effect slope association, no residual structural coupling (`rho=0`).
2. `Flat-Gaussian`: shared slope plus normalized constant residual weights (`lambda=0`).
3. `Decay-Gaussian`: shared slope plus normalized **exponentially decaying** residual weights (`rho` and `lambda` estimated).
4. `Decay-SNI`: decaying kernel with two-component skew-t random effects and independent Student-t residual scale mixtures (fixed df=6).

The objective is a weakly regularized joint log likelihood (MAP-type estimation under explicit Gaussian penalties), optimized with L-BFGS-B and automatic derivatives from PyTorch. Optimizer convergence flags, bounds, and fitted values are all retained. Neither parameter identification nor numerical convergence is presumed guaranteed.

## Three simulation regimes

- `Gaussian-decay`: correctly specified decay + Gaussian errors, used for bias/RMSE and comparison against omitted or flat coupling.
- `SkewT-decay`: skewed heavy-tailed random effects **and** independent heavy-tailed residuals, assessing robustness.
- `Null-residual`: `rho=0`, assessing false association and overfit in the more flexible models. **Lambda is unidentifiable under this null**, and lambda recovery is purposefully not summarized for it.

Each replicate creates fresh, irregular visit times, independent biomarker innovations, right-censored residual event times, and independent subject-level train/test sets. The simulation is not manipulated to force the proposed model to win. For publication-level Monte Carlo standard errors, increase `HJ_SIM_REPS` to at least 100 or more after a computational feasibility check, and inspect nonconvergence counts. The Colab defaults prioritize reasonable turnaround for **initial verification** and should **not** be reported as definitive simulation evidence.

## Runtime settings

Set environment variables in the notebook's configuration cell *before* executing:

| Variable | Meaning | Notebook starting value |
|---|---|---|
| `HJ_SIM_REPS` | Independent repetitions per scenario | `12` |
| `HJ_SIM_N` | Subjects per simulated replicate | `240` |
| `HJ_REAL_REPEATS` | Independent 72/28 train/test splits | `3` |
| `HJ_MAX_ITER` | L-BFGS-B maximum iterations | `160` |
| `HJ_ROBUST_QUAD` | Half-normal x U gamma x V gamma nodes | `5,5,4` |
| `HJ_LANDMARK` | Main landmark in years | `2.0` |
| `HJ_SEED` | Random seed | `20261103` |

The standalone `.py` uses lighter defaults (`6` reps, `200` subjects, `2` splits, `115` iterations, `4,4,3` nodes). The notebook explicitly overrides these for a stronger initial run. Increase the settings for finalized paper simulations, review model convergence, and repeat with finer quadrature nodes (e.g. `7,7,5`) to verify the SNI numerical approximation.

## Verifiability and limitations

The script verifies the **exact Gaussian likelihood identity** against a directly assembled multivariate normal covariance, checks differentiation and probability ranges, checks actual eigenvalues against the Schur formula, and records failure logs. Reproducibility includes source-file SHA-256, random seed, run settings, versions, and held-out predictions.

**Do not interpret one split, a few simulation replications, or conditional log scores as proof of a universally superior procedure.** This is a genuinely computed verification study whose outputs may support, weaken, or falsify assumptions and claims. In particular, weak identification of `lambda` is expected if every subject has nearly identical visit schedules or if residual association is negligible. Further work before a top-journal submission would include more replications, Monte Carlo standard errors, full numerical integration sensitivity, formal uncertainty quantification, possible informative observation processes, and competing-risk extensions if clinically required. The mathematical results in the eventual manuscript must be compatible with this corrected, fixed-landmark generative formulation.
