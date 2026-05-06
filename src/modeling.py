
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import joblib
import datetime

from sklearn.gaussian_process.kernels import Kernel, GenericKernelMixin

#
class PrecomputedKernel(Kernel, GenericKernelMixin):
    """
    class to allow precomuted kernel (G + K + GxK)
    """
    def __init__(self, K):
        self.K = K

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError('gradients not supported for precomputed kernels')

        # x and y will be passed as idx in the gpr to look up value in precomputed mat
        idx_x = X.flatten().astype(int)
        if Y is None:
            return self.K[idx_x][:, idx_x]
        else:
            idx_y = Y.flatten().astype(int)
            return self.K[idx_x][:, idx_y]

    def diag(self, X):
        return np.diag(self.K)[X.flatten().astype(int)]

    def is_stationary(self):
        return False



def load_train_data(config_file, ROOT):
    """
    Loads three dataset. SNP file, ENV file, Target file
    :param config_file: custom file
    :param ROOT: home path
    :return: NP file, ENV file, Target file
    """
    target_col = config_file['data_specs']['target']
    processed_path = ROOT / config_file['inputs']['processed_data_dir']
    idx_col = config_file['data_specs']['index']

    out_snp = ROOT / processed_path / config_file['outputs']['processed_snp']
    out_env = ROOT / processed_path / config_file['outputs']['processed_env']
    out_target = ROOT / processed_path / config_file['outputs']['processed_target']

    snp_df = pl.read_parquet(out_snp)
    env_df = pl.read_parquet(out_env)
    target_df = pl.read_parquet(out_target)

    # print(snp_df.shape, env_df.shape, target_df.shape)

    id_drop = idx_col
    # print(id_drop)
    X_snp = snp_df.drop(id_drop).to_numpy().astype(np.float32)
    X_env = env_df.drop(id_drop).to_numpy().astype(np.float32)
    y = target_df[target_col].to_numpy().flatten()

    print("Training datasets loaded successfully")

    return X_snp, X_env, y


def g_matrix(snp_arr, test_snp_arr=None):
    """
    Genetic relationship matrix construction, uses the VanRaden method
    :param snp_arr: np.ndarray. snp array of N_train_ind x N_snp
    :param test_snp_arr: np.ndarray. snp array of N_test_ind x N_snp
    :return: relationship matrix. if test_snp_arr=None, N_train x N_train else N_test x N_train
    """

    # G = ZZ'/sum_pq
    # or G = Z_test Z.T/sum_pq
    # allele freq
    p = np.mean(snp_arr, axis=0) / 2

    P = 2 * p
    Z = snp_arr - P

    sum_pq = 2 * np.sum(p * (1 - p))
    if test_snp_arr is None:
        G = np.dot(Z, Z.T) / sum_pq
    else:
        Z_test = test_snp_arr - P
        G = np.dot(Z_test, Z.T) / sum_pq

    G = G.astype(np.float32)

    print('G matrix constructed successfully')
    return G


def combine_kernels(G_subset, env_arr_scaled_train, env_arr_scaled_test=None, gamma = None, gxe_norm=None):
    """
    Computes K and GxK kernels as well as total kernels
    :param G_subset: np.ndarray. relationship matrix N_train x N_train or N_test x N_train
    :param env_arr_scaled_train: np.ndarray. train env data of scaled engineered features. N_train_ind x N_env
    :param env_arr_scaled_test: np.ndarray. test env data of scaled engineered features. N_test_ind x N_env
    :param gamma: float. for kernel scaling to determine similarity
    :param gxe_norm: float, norm val obtained from train GxE
    :return: kernel total N_train x N_train or N_test x N_train, gamma, and gxe_norm
    """

    # K kernel as similarity matrix
    # euclidean sq and the guaissian kernel - already normalized (diag already 1).
    if env_arr_scaled_test is None: # only train data
        dist = squareform(pdist(env_arr_scaled_train, 'sqeuclidean'))
        if gamma is None:
            gamma = 1.0 / np.median(dist) if np.median(dist) > 0 else 1.0
        K = np.exp(-gamma * dist)
        # calc norm value up here and use for test
        GxE = np.multiply(G_subset, K)
        gxe_norm = np.mean(np.diag(GxE))
        GxE_norm = GxE / gxe_norm
    else:
        dist = cdist(env_arr_scaled_test, env_arr_scaled_train, 'sqeuclidean')
        # gamma = 1.0 / np.median(dist) if np.median(dist) > 0 else 1.0
        K = np.exp(-gamma * dist)
        # print(K.shape)
        GxE = np.multiply(G_subset, K)
        GxE_norm = GxE / gxe_norm

    # print(gxe_norm)
    # print(np.mean(np.diag(G_subset)), np.mean(np.diag(K)), np.mean(np.diag(GxE_norm))) # expected ~1, ~1, ~1

    kernel_total = G_subset + K + GxE_norm
    # print('G, K, GxE matrices combined')
    return kernel_total, gamma, gxe_norm
    # return G_subset, K, GxE_norm


def gblup_solver_with_gpr(train_kernels, y, alpha):
    """
    Proxy for GBLUP model
    :param train_kernels: np.ndarray. N_train x N_train or N_test x N_train
    :param y: np.ndarray. target
    :param alpha: expected noise in y, similar to Ve
    :return: fitted model and y_scaler for use in prediction
    """
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # idx for len of kernel
    train_idx = np.arange(train_kernels.shape[0]).reshape(-1, 1)
    # kernel is precomputed
    # zero_kern = DotProduct(sigma_0=1e-10) # log(0) will throw a warning, hence the small number
    # gpr_model = GaussianProcessRegressor(kernel=zero_kern, alpha=alpha, optimizer=None)
    kernel = PrecomputedKernel(train_kernels)

    gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None)

    gpr_model.fit(train_idx, y_train_scaled)

    return gpr_model, y_scaler

def gblup_make_new_prediction(model, val_train_kernels, y_scaler):
    """
    Predict y for new genotypes
    :param model: model obj from gblup_solver_with_gpr
    :param val_train_kernels: np.ndarray. N_val x N_train
    :param y_scaler: from gblup_solver_with_gpr
    :return: predicted valued for new genotypes
    """

    test_idx = np.arange(val_train_kernels.shape[0]).reshape(-1, 1)

    # update model kernel to val_train
    model.kernel_.K = val_train_kernels

    y_pred_scaled = model.predict(test_idx)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return y_pred


def do_cross_validation(X_snp, X_env, y, config_file):
    """
    Does cross validation
    :param X_snp: np.ndarray. snp array of N_ind x N_snp
    :param X_env: np.ndarray. env array of N_ind x N_env
    :param y: np.ndarray. N_ind x 1
    :param config_file: custom file
    :return: dict containing cross validation metrics
    """

    print('Doing Cross-validation')

    cv_rmp = []
    cv_rmse = []
    cv_bias = []

    n_folds = config_file['training_params']['cv_folds']
    random_seed = config_file['training_params']['random_seed']
    alpha_val = config_file['training_params']['gpr_alpha']

    G = g_matrix(X_snp)
    kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
    for train_idx, val_idx in kf.split(X_env):
        # print(len(train_idx), len(val_idx))
        y_train, y_val = y[train_idx], y[val_idx]
        X_env_train, X_env_val = X_env[train_idx], X_env[val_idx]

        # normalize env for K
        scaler = StandardScaler()
        X_env_train_sc = scaler.fit_transform(X_env_train)
        X_env_val_sc = scaler.transform(X_env_val)

        # subset G
        G_train = G[train_idx][:, train_idx]
        # print(np.mean(np.diag(G_train)))
        G_val_train = G[val_idx][:, train_idx]
        # norm_val = np.mean(np.diag(G_train))
        # G_train_norm = G_train / norm_val
        # print(np.mean(np.diag(G_train_norm)))
        # G_val_train_norm = G_val_train / norm_val

        # build G + K + GxE kernel
        train_kernels, train_gamma, train_gxe_norm = combine_kernels(G_train, X_env_train_sc)

        # print(train_gamma, train_gxe_norm)
        val_train_kernels, _, _ = combine_kernels(
            G_val_train, X_env_train_sc,
            X_env_val_sc, gamma=train_gamma,
            gxe_norm=train_gxe_norm)

        # model with train
        gpr_model, y_scaler = gblup_solver_with_gpr(train_kernels, y_train, alpha=alpha_val)
        # predict val
        y_pred = gblup_make_new_prediction(gpr_model, val_train_kernels, y_scaler)

        rmp, _ = pearsonr(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        bias = LinearRegression().fit(y_pred.reshape(-1, 1), y_val).coef_[0]
        cv_rmp.append(rmp)
        cv_rmse.append(rmse)
        cv_bias.append(bias)
    # put results in a dict
    metrics = summarize_cv_metrics(cv_rmp, cv_rmse, cv_bias)

    print('Cross-validation completed')
    return metrics


def summarize_cv_metrics(cv_rmp, cv_rmse, cv_bias):
    """
    Summarizes results for cross validation
    :param cv_rmp: list
    :param cv_rmse: list
    :param cv_bias: list
    :return: dict
    """
    metrics = {}
    rmp_mean, rmp_std = float(np.round(np.mean(cv_rmp), 2)), float(np.round(np.std(cv_rmp), 3))
    rmse_mean, rmse_std = float(np.round(np.mean(cv_rmse), 2)), float(np.round(np.std(cv_rmse), 3))
    bias_mean, bias_std = float(np.round(np.mean(cv_bias), 2)), float(np.round(np.std(cv_bias), 3))

    metrics = {
        'rMP': {'mean': rmp_mean, 'std': rmp_std},
        'RMSE': {'mean': rmse_mean, 'std': rmse_std},
        'Bias': {'mean': bias_mean, 'std': bias_std}
    }
    # metrics['CV_rMP'] = {'mean': rmp_mean, 'std': rmp_std}
    # metrics['CV_RMSE'] = {'mean': rmse_mean, 'std': rmse_std}
    # metrics['CV_Bias'] = {'mean': bias_mean, 'std': bias_std}

    return metrics



def train_and_save_model(y, X_snp, X_env, config_file, ROOT):
    """
    Trains model on full training data. Saves to path
    :param y: np.ndarray. N_train_ind x 1
    :param X_snp: np.ndarray. snp array of N_train_ind x N_snp
    :param X_env: np.ndarray. env array of N_train_ind x N_env
    :param config_file: custom file
    :param ROOT: home path
    :return: None
    """

    alpha = config_file['training_params']['gpr_alpha']

    model_path = ROOT / config_file['outputs']['model_dir']
    model_path.mkdir(parents=True, exist_ok=True)

    # y scaling is done in gblub solver
    G = g_matrix(X_snp)

    # env scaling
    env_scaler = StandardScaler()
    X_env_sc = env_scaler.fit_transform(X_env)

    train_kernels, gamma, gxe_norm = combine_kernels(G, X_env_sc)

    model, y_scaler = gblup_solver_with_gpr(train_kernels, y, alpha)


    print('Training completed')

    model_data = {
        'gpr_model': model,
        'y_scaler': y_scaler,
        'gamma': gamma,
        'gxe_norm': gxe_norm,
        # 'g_norm': g_norm,
        'training_config': config_file['training_params']
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    full_path = model_path / f"gpr_model_{timestamp}.pkl"
    joblib.dump(model_data, full_path)
    print(f'Model saved in {model_path}')

