// Hidden Markov Model for Market Regime Detection
// Implements a 2-state HMM with:
//   State 1: "Calm/Normal" market regime
//   State 2: "Crisis" regime
// Uses the Forward Algorithm (Rao-Blackwellization) in log-space
// for numerical stability and O(T) complexity.

data {
  int<lower=1> T;           // Number of weekly observations
  vector[T] Y;              // Weekly log returns: Y_t = ln(P_t / P_{t-1})
}

parameters {
  // --- Emission Parameters ---
  // Mean log return for each state
  array[2] real mu;

  // Volatility parameterization: sigma_2 = sigma_1 + sigma_diff
  // ensures sigma_2 > sigma_1 (Crisis is more volatile than Calm)
  real<lower=0> sigma_1;
  real<lower=0> sigma_diff;

  // --- Transition Probabilities ---
  real<lower=0, upper=1> P11;  // P(stay in Calm  | currently Calm)
  real<lower=0, upper=1> P22;  // P(stay in Crisis | currently Crisis)
}

transformed parameters {
  // Derived volatility for Crisis state
  real<lower=0> sigma_2;
  sigma_2 = sigma_1 + sigma_diff;

  // Collect sigmas into an array for indexing
  array[2] real<lower=0> sigma;
  sigma[1] = sigma_1;
  sigma[2] = sigma_2;

  // Transition matrix (in log-space for numerical stability)
  // P = | P11      1-P11 |
  //     | 1-P22    P22   |
  matrix[2, 2] log_P;
  log_P[1, 1] = log(P11);
  log_P[1, 2] = log1m(P11);
  log_P[2, 1] = log1m(P22);
  log_P[2, 2] = log(P22);
}

model {
  // -------------------------------------------------------
  // Priors
  // -------------------------------------------------------

  // Volatility priors (as specified in report)
  sigma_1    ~ exponential(1);
  sigma_diff ~ exponential(1);

  // Transition probability priors: Beta(1,1) = Uniform(0,1)
  P11 ~ beta(1, 1);
  P22 ~ beta(1, 1);

  // Weakly informative priors on means
  mu[1] ~ normal(0, 0.05);
  mu[2] ~ normal(0, 0.05);

  // -------------------------------------------------------
  // Forward Algorithm (log-space) — Rao-Blackwellization
  // Marginalizes out the discrete latent state sequence X_{1:T}
  // Reduces complexity from O(2^T) to O(T)
  // -------------------------------------------------------

  // --- Initial state distribution (uniform: pi = [0.5, 0.5]) ---
  vector[2] log_alpha;

  // Base case (t = 1):
  // log alpha_1(j) = log(pi_j) + log p(Y_1 | mu_j, sigma_j^2)
  for (j in 1:2) {
    log_alpha[j] = log(0.5) + normal_lpdf(Y[1] | mu[j], sigma[j]);
  }

  // Recursive step (t = 2, ..., T):
  for (t in 2:T) {
    vector[2] log_alpha_new;
    for (j in 1:2) {
      vector[2] accumulator;
      for (i in 1:2) {
        accumulator[i] = log_alpha[i] + log_P[i, j];
      }
      log_alpha_new[j] = normal_lpdf(Y[t] | mu[j], sigma[j])
                         + log_sum_exp(accumulator);
    }
    log_alpha = log_alpha_new;
  }

  // Termination: add marginalized log-likelihood to target
  target += log_sum_exp(log_alpha);
}

generated quantities {
  // -------------------------------------------------------
  // Filtered state probabilities P(X_t = Crisis | Y_{1:t})
  // -------------------------------------------------------

  matrix[T, 2] log_alpha_full;
  vector[T] prob_crisis;

  // Base case
  for (j in 1:2) {
    log_alpha_full[1, j] = log(0.5) + normal_lpdf(Y[1] | mu[j], sigma[j]);
  }

  // Recursive step
  for (t in 2:T) {
    for (j in 1:2) {
      vector[2] accumulator;
      for (i in 1:2) {
        accumulator[i] = log_alpha_full[t-1, i] + log_P[i, j];
      }
      log_alpha_full[t, j] = normal_lpdf(Y[t] | mu[j], sigma[j])
                              + log_sum_exp(accumulator);
    }
  }

  // Normalize to get filtered probabilities
  for (t in 1:T) {
    real log_normalizer = log_sum_exp(log_alpha_full[t, 1],
                                      log_alpha_full[t, 2]);
    prob_crisis[t] = exp(log_alpha_full[t, 2] - log_normalizer);
  }
}


