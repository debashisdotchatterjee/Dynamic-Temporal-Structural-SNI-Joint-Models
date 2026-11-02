# Landmark Structural Joint Models with Normalized Temporal Decay and Skew-Normal/Independent Random Effects

**Research and reproducibility repository**  
Debashis Chatterjee and Arindom Chakraborty  
Department of Statistics, Visva-Bharati University, Santiniketan, India

This repository accompanies the research article *Landmark Structural Joint Models with Normalized Temporal Decay and Skew-Normal/Independent Random Effects*. It contains a complete LaTeX manuscript, mathematical proofs, Python implementations, a self-contained Google Colab notebook, publication figures and tables, and the machine-readable results of the computational study.

## Overview

The framework combines longitudinal biomarker measurements with residual survival after a **prespecified prediction landmark**. It separates two sources of prognostic association: persistent subject-specific trajectory variation and time-weighted measurement innovations. Skew-normal/independent (SNI) random effects and scale mixtures extend the Gaussian formulation to asymmetric and heavy-tailed data.

For a subject with recorded pre-landmark visits $t_{ij}\leq\ell$, the normalized recency weights are

$$
w_{ij}(\lambda;\ell)=\frac{\exp[-\lambda(\ell-t_{ij})]}{\{\sum_{k=1}^{n_i}\exp[-2\lambda(\ell-t_{ik})]\right\}^{1/2}\},\qquad \sum_{j=1}^{n_i}w_{ij}^{2}=1.
$$

Writing $R_i=T_i^*-\ell>0$ for residual survival time in the landmark-eligible cohort, the longitudinal and accelerated failure-time components are

$$
\mathbf Y_i=\mathbf X_i\boldsymbol\beta+\mathbf Z_i\mathbf b_i+\boldsymbol\varepsilon_i,
\qquad
\log R_i=\mathbf q_i^\mathsf T\boldsymbol\alpha+\nu b_{1i}+\eta_i,
$$

with the innovation covariance, conditional on its positive scale $V_i=v$,

$$
\operatorname{Cov}\!\left(\begin{bmatrix}\boldsymbol\varepsilon_i\\\eta_i\end{bmatrix}\Bigm|V_i=v\right)
=\frac1v\begin{bmatrix}
\sigma_e^2 I_{n_i}&\rho\sigma_e\sigma_T\mathbf w_i\\
\rho\sigma_e\sigma_T\mathbf w_i^\mathsf T&\sigma_T^2
\end{bmatrix}.
$$

Its Schur complement is $\sigma_T^2(1-\rho^2)>0$ for $|\rho|<1$, irrespective of the finite visit count or irregular sampling schedule. The manuscript gives the formal proofs, moment decomposition, likelihood construction, and the conditional structural-identification result. The model is **landmark- and finite-design-specific**; its normalized covariance matrices are not claimed to be the projective marginals of one visit-schedule-invariant continuous-time process.

## Statistical methodology

- **Random effects:** centered SNI random intercept and slope, $\mathbf b_i=U_i^{-1/2}(\boldsymbol\Delta H_i+\mathbf G_i)-a_U\boldsymbol\Delta$, with $H_i$ half-normal and $U_i$ an independent positive mixing variable.
- **Residual innovations:** an additional independent scale $V_i$ induces heavy-tailed measurement and event innovations; the reported SNI model takes $U_i,V_i\stackrel{\mathrm{iid}}\sim\operatorname{Gamma}(3,3)$ (shape–rate, six degrees of freedom).
- **Likelihood:** all Gaussian random-intercept/slope terms are analytically integrated out. The SNI fit then uses deterministic probability-scale Gauss–Legendre quadrature over $(H_i,U_i,V_i)$.
- **Estimation:** penalized marginal joint likelihood, automatic differentiation, and bounded L-BFGS-B optimization; **no MCMC** is used.
- **Prediction:** conditional event probabilities from the joint distribution of pre-landmark biomarker history and residual survival, with right-censored contributions handled through the conditional survivor function.

Four specified models are compared: `Shared-Gaussian` (no residual association), `Flat-Gaussian` (normalized flat loading), `Decay-Gaussian` (normalized exponential recency loading), and `Decay-SNI` (temporal loading plus SNI mixtures).

## Computational experiments

The simulation study considers: Gaussian temporal association, skewed/heavy-tailed temporal association, and a null-residual-association control. It evaluates held-out conditional survival log score, IPCW Brier score, concordance, and parameter recovery. Reported experiments contain **12 independent replicates per scenario, each with 240 simulated subjects**, and the random seed `20261103`.

The empirical study uses the Mayo Clinic **primary biliary cirrhosis** longitudinal dataset, `survival::pbcseq` (312 trial patients, 1,945 sequential visits). Log serum bilirubin is the longitudinal outcome; death or liver transplantation forms a composite survival endpoint. At the two-year landmark, the defined risk set contains **266 eligible participants**, with **132 observed subsequent composite events** and **134 censored observations**. The study reports three subject-level stratified 72/28 training/test partitions and supplementary landmark analyses at one and three years.

Treatment in the distributed analysis CSV is coded `1 = D-penicillamine`, `0 = placebo`. Only measurements collected no later than each prediction landmark are used. The study makes no causal claim about treatment effects.

**Data documentation:** [Mayo PBC sequential dataset (`survival`)](https://therneau.r-universe.dev/survival/doc/page/pbcseq) · [public CSV](https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv). Source information and checksums are recorded in `provenance/source.json`.

## Repository contents

```text
.
├── README.md
├── LICENSE
├── main.tex                      # Article and mathematical appendices
├── ref.bib                       # References
├── main.pdf                      # Compiled manuscript
├── figures/                      # Vector PDF research figures
├── tables/                       # Machine-generated LaTeX table data
├── analysis_code/
│   ├── HJ_Joint_Model_Colab.ipynb
│   ├── hj_joint_study.py
│   ├── quadrature_refinement.py
│   └── requirements.txt
└── provenance/                   # Simulation / PBC results and run metadata
```

## Reproducing the analyses

**Google Colab.** Open `analysis_code/HJ_Joint_Model_Colab.ipynb` in [Google Colab](https://colab.research.google.com/), and execute all cells. The notebook includes the analysis source and configuration and creates `HJ_structural_joint_outputs.zip` containing plots, tables, raw results, fit diagnostics, code, and environment metadata. Colab may require access to the public source CSV; if network access is unavailable, upload the genuine `pbcseq.csv` to the current working directory.

**Local Python.** Python 3.10+ is recommended. Install the dependencies and set the article's computational settings:

```bash
python -m pip install -r analysis_code/requirements.txt
export HJ_SEED=20261103
export HJ_LANDMARK=2.0
export HJ_SIM_REPS=12
export HJ_SIM_N=240
export HJ_REAL_REPEATS=3
export HJ_MAX_ITER=160
export HJ_ROBUST_QUAD=5,5,4
python analysis_code/hj_joint_study.py
```

On Windows PowerShell, use `$env:HJ_SIM_REPS="12"` (similarly for the other variables), then run the Python command. The standalone script uses smaller exploratory defaults if these variables are not set. Re-running the full experiments is computationally more demanding than reading the included results.

For a fixed-parameter assessment of integration accuracy in the clinical application, run:

```bash
python analysis_code/quadrature_refinement.py
```

This diagnostic evaluates already fitted SNI parameter vectors using `5×5×4`, `7×7×6`, and `9×9×7` integration nodes. It does **not** replace refitting each model under refined quadrature. The published numerical results and their precise model settings are available in `provenance/run_config.json` and the corresponding CSV files.

## Mathematical and numerical checks

The program checks positive-definiteness through the analytic Schur complement and direct eigenvalues, verifies the Gaussian marginalized event likelihood against a directly assembled multivariate Gaussian density, checks finite automatic derivatives, and validates predicted-probability ranges. Both optimizer diagnostics and test-subject predictions are preserved. The positive-definiteness proof establishes covariance admissibility, not strong finite-sample parameter identification.

The clinical results are interpreted as **exploratory predictive evidence**: residual-association models improve some held-out scores compared with the shared-effects-only baseline, while the data do not establish consistent superiority of temporal decay over normalized flat association. Several fitted association parameters lie on optimization bounds; consequently they should not be given precise clinical effect interpretations. The limited simulation replication count, possible informative visit schedules, and absence of external validation are discussed in the article.

## Compiling the manuscript

Upload the repository's manuscript files as a new [Overleaf](https://www.overleaf.com/) project, choose `main.tex` as the root document, and compile with pdfLaTeX and BibTeX. Alternatively:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The paper contains the derivations, labeled proofs in appendices, simulation and real-data tables, vector figures, and declarations.

## Citation and licensing

If you use the methodology or implementation, please cite the accompanying manuscript, *Landmark Structural Joint Models with Normalized Temporal Decay and Skew-Normal/Independent Random Effects*, by Debashis Chatterjee and Arindom Chakraborty. A journal DOI can be added upon publication.

Source code is distributed under the **[MIT License](LICENSE)**. The historical clinical dataset is attributed to its original study and maintainers; its provenance and any applicable source terms are separate from the source-code license. No competing interests are declared in the manuscript.
