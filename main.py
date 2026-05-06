
# 1. Load Config to get paths and training parameters
# 2. Separate out test data
# 2. Do some EDA if true, and then preprocessing data
# 3. Do feature engineering
# 4. Train model and do cross val
# 5. Load model to predict new data


import numpy as np
import polars as pl
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.modeling import PrecomputedKernel
from src.eda import (target_distribution, env_features_target_corr_plots,
                     feature_timestamp_boxplots, feature_correlation_plots,
                     get_all_correlation_plot, make_prediction_plots)
from src.processor import snp_processing, snp_recode_impute, quality_checks
from src.feature_engineering import split_frame, process_env_data, update_env_data, save_processed_files
from src.modeling import load_train_data, do_cross_validation, train_and_save_model
from src.predict import process_new_lines, build_kernels_predict_new_genotypes, combine_and_save_metrics


def run_pipeline():

    print('===== STARTING PIPELINE ====')
    ROOT = Path(__file__).resolve().parent

    config_path = ROOT / 'config.yaml'

    with open(config_path, 'r') as f:
        config_file = yaml.safe_load(f)

    df_path = ROOT / config_file['inputs']['raw_data_path']
    df = pl.read_csv(df_path)
    idx_col = config_file['data_specs']['index']
    df = df.with_row_index(idx_col)

    test_ratio = config_file['training_params']['test_split_ratio']
    random_seed = config_file['training_params']['random_seed']
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_seed)

    processed_path = ROOT / config_file['inputs']['processed_data_dir']
    processed_path.mkdir(parents=True, exist_ok=True)
    train_out_path = processed_path / config_file['outputs']['train_file']
    test_out_path = processed_path / config_file['outputs']['test_file']

    plot_path = ROOT / config_file['outputs']['plots_dir']
    plot_path.mkdir(parents=True, exist_ok=True)

    train_df.write_parquet(train_out_path)
    test_df.write_parquet(test_out_path)

    null_counts = train_df.null_count()
    null_counts.sum_horizontal().item()


    # EDA
    if config_file['execution_flags']['save_plot']:
        print(f'===== RUNNING EDA. PLOTS WILL BE SAVED IN {plot_path} ====')

        # get y distribution
        target_distribution(train_df, config_file, save_path=plot_path / 'train_target_dist.png')
        # environmental correlations
        env_features_target_corr_plots(train_df, config_file, save_path=plot_path / 'timestamp_corr_with_target.png')
        feature_timestamp_boxplots(df, config_file, save_path=plot_path / 'timestamp_boxplot.png')
        feature_correlation_plots(df, config_file, add_var=None, save_path=plot_path / 'timestamp_corr.png')

    print('==== DOING PREPROCESSING ====')
    # PREPROCESSING
    snp_processing(train_df, config_file, ROOT)
    train_df_recoded = snp_recode_impute(train_df, config_file, ROOT)
    train_qc = quality_checks(train_df_recoded, config_file)
    snp_data, env_data, target_data = split_frame(train_qc, config_file)

    env_processed = process_env_data(env_data, config_file)
    env_df_updated = update_env_data(env_processed, config_file)

    if config_file['execution_flags']['save_plot']:
        feature_correlation_plots(env_processed, config_file,add_var=config_file['data_specs']['engineered_variables'],
                                  save_path=plot_path / 'engineered_corr.png')
        get_all_correlation_plot(env_df_updated, target_data, save_path=plot_path / 'final_env_target_corr.png')


    save_processed_files(snp_data, env_df_updated, target_data, config_file, ROOT)

    print('==== RUNNING MODELING AND PREDICTION ====')
    # MODELING AND PREDICTION
    X_train_snp, X_train_env, y_train = load_train_data(config_file, ROOT)

    # cross validation
    cv_metrics = do_cross_validation(X_train_snp, X_train_env, y_train, config_file)
    # build and save model on full train data
    train_and_save_model(y_train, X_train_snp, X_train_env, config_file, ROOT)

    # predict on unseen test data

    # get the test file
    test_file = config_file['outputs']['test_file']
    test_path = ROOT / processed_path / test_file
    # run some preprocessing
    test_snp_df, test_env, test_y_df = process_new_lines(test_path, config_file, ROOT)

    # check distribution of test target
    if config_file['execution_flags']['save_plot'] and not np.all(np.isnan(test_y_df)):
        target_distribution(test_y_df, config_file, save_path=plot_path / 'test_target_dist.png')

    # predict cucumber weight for new lines
    y_pred, _ = build_kernels_predict_new_genotypes(test_snp_df, test_env, X_train_snp, X_train_env, config_file, ROOT)

    metrics = combine_and_save_metrics(cv_metrics, y_pred, config_file, ROOT, test_y_df)

    if config_file['execution_flags']['save_plot']:
        make_prediction_plots(y_pred, test_y_df, metrics, save_path=plot_path / 'test_prediction_plots.png')

    print('==== PIPELINE RUN COMPLETE ====')


if __name__ == '__main__':
    run_pipeline()


