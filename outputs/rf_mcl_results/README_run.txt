RF-MCL phase diagram numerical experiment
=========================================

d=64, p=192, psi_p=3, mu=1.0, sigma0=0.4
activation=tanh, seed=0
n_train=12000, n_eval=5000, power_iters=35

Outputs:
  A_population_tau_inf.csv / A_beta_spec_population_tau_inf.png
  B_population_finite_tau.csv / B_beta_spec_finite_tau.png
  C_empirical_REM_glass.csv / C_beta_glass_vs_tau.png / C_entropy_collapse_beta_scaled.png

Interpretation:
  A checks beta_spec=1/(2 lambda_max) and top-mode alignment with the GMM class direction.
  B checks the finite-training spectral filter in beta_spec(t,tau).
  C checks beta_glass(t,tau)=sqrt(2 alpha / v(t,tau)) with v(t,tau)=v_inf R(t,tau).
