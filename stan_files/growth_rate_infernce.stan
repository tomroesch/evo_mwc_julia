functions {
    vector gp_pred_rng(real[] x2,
                         vector y1, real[] x1,
                         real alpha, real rho, real sigma, real delta) {
        int N1 = rows(y1);
        int N2 = size(x2);
        vector[N2] f2;
        {
          matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
                             + diag_matrix(rep_vector(square(sigma), N1));
          matrix[N1, N1] L_K = cholesky_decompose(K);

          vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
          vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
          matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
          vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
          matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
          matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
                                  + diag_matrix(rep_vector(delta, N2));
          f2 = multi_normal_rng(f2_mu, cov_f2);
        }
        return f2;
      }

  vector gp_pred_der_rng(real[] x2,
                     vector y1, real[] x1,
                     real alpha, real rho, real sigma, real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] g2;
    {     
      matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
                             + diag_matrix(rep_vector(square(sigma), N1));

      matrix[N1, N1] L_K = cholesky_decompose(K);

      vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
      vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
      
      matrix[N1, N2] K_1;
      for (i in 1:N1){
        for (j in 1:N2){
          K_1[i, j] = (x1[i] - x2[j])/rho^2 * k_x1_x2[i,j];
        }
      }
      matrix[N2, N2] k_x2_x2 = cov_exp_quad(x2, alpha, rho);
      matrix[N2, N2] K_2;
      for (i in 1:N2){
        for (j in 1:N2){
          K_2[i, j] = (1/rho^2 - (x2[i] - x2[j])^2 /rho^4) * k_x2_x2[i, j];
        }
      }
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, K_1);
      vector[N2] g2_mu = (K_1' * K_div_y1);
      matrix[N2, N2] cov_g2 =   K_2 - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));

      g2 = multi_normal_rng(g2_mu, cov_g2);
    }
    return g2;
  }
}

data {
  int<lower=1> N;
  real x[N];
  vector[N] y;

  int<lower=1> N_predict;
  real x_predict[N_predict];
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

model {

  rho ~ normal(8000, 1000);
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 1);

  matrix[N, N] cov_exp = cov_exp_quad(x, alpha, rho);
  matrix[N, N] cov = cov_exp + diag_matrix(rep_vector(square(sigma), N));
                     
  matrix[N, N] L_cov = cholesky_decompose(cov);

  
  y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
}

generated quantities {
  vector[N_predict] f_predict = gp_pred_rng(x_predict, y, x, alpha, rho, sigma, 1e-8);
  vector[N_predict] g_predict = gp_pred_der_rng(x_predict, y, x, alpha, rho, sigma, 1e-8);
  vector[N_predict] y_predict;
  vector[N_predict] y_p_predict;
  for (n in 1:N_predict){
    y_predict[n] = normal_rng(f_predict[n], sigma);
    y_p_predict[n] = normal_rng(g_predict[n], 1e-5);
}
}