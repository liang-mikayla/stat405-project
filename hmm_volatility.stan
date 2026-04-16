data {
  int<lower=1> T;
  vector[T] Y;
}

parameters {
  real<lower=0> sigma_1;
  real<lower=0> sigma_diff;
  real<lower=0, upper=1> P11;
  real<lower=0, upper=1> P22;
}

transformed parameters {
  real<lower=0> sigma_2;
  array[2] real<lower=0> sigma;
  matrix[2, 2] log_P;
  matrix[T, 2] log_alpha_full;

  sigma_2 = sigma_1 + sigma_diff;
  
  sigma[1] = sigma_1;
  sigma[2] = sigma_2;

  log_P[1, 1] = log(P11);
  log_P[1, 2] = log1m(P11);
  log_P[2, 1] = log1m(P22);
  log_P[2, 2] = log(P22);

  for (j in 1:2)
    log_alpha_full[1, j] = log(0.5) + normal_lpdf(Y[1] | 0, sigma[j]);

  for (t in 2:T) {
    for (j in 1:2) {
      vector[2] accumulator;
      for (i in 1:2)
        accumulator[i] = log_alpha_full[t-1, i] + log_P[i, j];
      log_alpha_full[t, j] = normal_lpdf(Y[t] | 0, sigma[j]) + log_sum_exp(accumulator);
    }
  }
}

model {
  sigma_1 ~ exponential(1);
  sigma_diff ~ exponential(1);

  P11 ~ beta(50, 2);
  P22 ~ beta(20, 2);

  target += log_sum_exp(log_alpha_full[T, 1], log_alpha_full[T, 2]);
}

generated quantities {
  matrix[T, 2] log_beta;
  matrix[T, 2] log_gamma;
  vector[T] prob_crisis;

  for (j in 1:2) {
    log_beta[T, j] = 0.0;
  }

  for (t_rev in 1:(T - 1)) {
    int t = T - t_rev;
    
    for (i in 1:2) {
      vector[2] accumulator;
      for (j in 1:2) {
        accumulator[j] = log_P[i, j] + normal_lpdf(Y[t+1] | 0, sigma[j]) + log_beta[t+1, j];
      }
      log_beta[t, i] = log_sum_exp(accumulator);
    }
  }

  for (t in 1:T) {
    log_gamma[t, 1] = log_alpha_full[t, 1] + log_beta[t, 1];
    log_gamma[t, 2] = log_alpha_full[t, 2] + log_beta[t, 2];

    prob_crisis[t] = exp(log_gamma[t, 2] - log_sum_exp(log_gamma[t, 1], log_gamma[t, 2]));
  }
  
real y_predict;
  
  real p_state1_T = exp(log_alpha_full[T, 1] - log_sum_exp(log_alpha_full[T, 1], log_alpha_full[T, 2]));
  real p_state2_T = exp(log_alpha_full[T, 2] - log_sum_exp(log_alpha_full[T, 1], log_alpha_full[T, 2]));
  
  real p_state1_next = p_state1_T * P11 + p_state2_T * (1 - P22);
  
  int predicted_state = bernoulli_rng(1 - p_state1_next) + 1; 
  
  y_predict = normal_rng(0, sigma[predicted_state]);
}