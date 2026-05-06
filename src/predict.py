import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import joblib
import json



def process_new_lines(new_data_path, config_file, ROOT):
    """
    First func to run new data on. It does preprocessing, imputation and feature engineering based on metadata from train
    :param new_data_path: str. path of new data.
    :param config_file: custom file
    :param ROOT: home path
    :return: Three data set that has gone through processing and QC. SNP recoded, ENV with eng features, and Target
    (will contain null if not available)
    """
    from src.feature_engineering import split_frame, process_env_data, update_env_data
    from src.processor import snp_recode_impute, quality_checks

    processed_dir = config_file['inputs']['processed_data_dir']
    processed_path = ROOT / processed_dir
    target_col = config_file['data_specs']['target']

    idx_col = config_file['data_specs']['index']

    test_df = pl.read_parquet(new_data_path)
    # snps
    test_df_recode = snp_recode_impute(test_df, config_file, ROOT)
    test_df_qc = quality_checks(test_df_recode, config_file)
    # print(test_df_qc.shape)

    # split data
    snp_df, env_df, y_df =  split_frame(test_df_qc, config_file)
    # print(snp_df.shape)
    # env
    env_proc_df = process_env_data(env_df, config_file)
    env_updated_df = update_env_data(env_proc_df, config_file)

    # normalize with metadata
    # metadata_path = processed_path / 'env_metadata.json'
    # env_norm = normalize_env_vars_for_new_lines(env_updated_df, metadata_path)

    # convert to numpy
    id_drop = idx_col
    X_test_snp = snp_df.drop(id_drop).to_numpy().astype(np.float32)
    X_test_env = env_updated_df.drop(id_drop).to_numpy().astype(np.float32)
    y_test = y_df[target_col].to_numpy().flatten()

    print('New data processed successfully')
    print(f'SNP: {X_test_snp.shape}, ENV: {X_test_env.shape}, Y: {y_test.shape}')

    return X_test_snp, X_test_env, y_test


def latest_model_path(config_file, ROOT):
    """
    Loads the most recent model
    :param config_file: custom file
    :param ROOT: home path
    :return: str. path to latest model
    """
    model_dir = config_file['outputs']['model_dir']
    full_model_dir = ROOT / model_dir
    name_pattern = config_file['outputs']['model_name_pattern']
    model_files = list(full_model_dir.glob(name_pattern))

    if not model_files:
        raise FileNotFoundError(f'No models found in {full_model_dir}')
    # sort by time modified, pick max
    latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model_path


# gamma = data['gamma']
def build_kernels_predict_new_genotypes(new_snp, new_env, train_snp, train_env, config_file, ROOT):
    """
    Loads model. Creates N_test x N_train total kernel as input for model. Predict new genotypes
    :param new_snp: np.ndarray. snp array N_test_ind x N_snp
    :param new_env: np.ndarray. env array N_test_ind x N_env
    :param train_snp: np.ndarray. snp array N_train_ind x N_snp
    :param train_env: np.ndarray. env array N_train_ind x N_env
    :param config_file: custom file
    :param ROOT: home path
    :return: Two outputs, predicted data of new genotypes and N_test x N_train total kernel matrix
    """
    from src.modeling import g_matrix, combine_kernels, gblup_make_new_prediction

    model_path = latest_model_path(config_file, ROOT)
    model_data = joblib.load(model_path)
    print(f'Model loaded successfully: {model_path}')

    model = model_data['gpr_model']
    y_scaler = model_data['y_scaler']
    gamma, gxe_norm = model_data['gamma'], model_data['gxe_norm']

    G_test_train = g_matrix(train_snp, new_snp)
    # print(G_test_train.shape)

    env_scaler = StandardScaler()
    X_env_train_sc = env_scaler.fit_transform(train_env)
    X_env_test_sc = env_scaler.transform(new_env)

    test_train_kernels, _, _ = combine_kernels(G_test_train, X_env_train_sc, X_env_test_sc, gamma=gamma, gxe_norm=gxe_norm)
    print(G_test_train.shape, test_train_kernels.shape)

    # make predictions
    y_pred = gblup_make_new_prediction(model, test_train_kernels, y_scaler)

    print('Prediction of new genotypes completed')
    return y_pred, test_train_kernels


def summarize_metrics_for_new_lines(y_pred, y_score=None):
    """
    Summarize metrics for new genotypes. Will return NA if true data is not available
    :param y_pred: output from build_kernels_predict_new_genotypes
    :param y_score: optional, but will return NA if unavailable
    :return: dict of metrics
    """

    print(y_pred.shape)
    y_score = np.array(y_score).flatten()
    y_pred = np.array(y_pred).flatten()

    if y_score is None or np.all(np.isnan(y_score)):
        return {
            'rMP': 'N/A', 'RMSE': 'N/A', 'Bias': 'N/A'
        }

    rmp_val, _ = pearsonr(y_score, y_pred)
    rmp = float(rmp_val)
    rmse = float(np.round(np.sqrt(mean_squared_error(y_score, y_pred)), 2))
    bias = float(np.round(LinearRegression().fit(y_pred.reshape(-1, 1), y_score).coef_[0], 2))

    results = {'rMP': np.round(rmp, 2), 'RMSE': rmse, 'Bias': bias}
    return results


def combine_and_save_metrics(cv_metrics, test_y_pred, config_file, ROOT, test_y_score=None):
    """
    Combines metrics from cross validation and new genotype prediction
    :param cv_metrics: dict. output from do_cross_validation
    :param test_y_score: optional, will return NA in test but CV results are valid
    :param test_y_pred: output from build_kernels_predict_new_genotypes
    :param config_file: custom file
    :param ROOT: home path
    :return: dict. CV and Test metrics
    """

    metrics_path = ROOT / config_file['outputs']['metrics_dir']
    metrics_path.mkdir(parents=True, exist_ok=True)

    test_summary = summarize_metrics_for_new_lines(test_y_pred, test_y_score)

    metrics = {
        'Cross_validation': cv_metrics,
        'External_test': test_summary
    }

    metrics_path_full = metrics_path / 'metrics.json'
    with open(metrics_path_full, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Model metrics saved in {metrics_path}')

    return metrics
