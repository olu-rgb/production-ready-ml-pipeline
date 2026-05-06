import polars as pl



def split_frame(df, config_file):
    """
    Splits data into stand-alone components
    1. SNP data. 2. Env data. 3. Target data.
    If target_data not in file ie truly new data without target, it still creates target_data but with null values
    :param df: full df, train or test. pl.Dataframe
    :param config_file: custom file
    :return: Three files. 1. SNP data. 2. Env data. 3. Target data.
    """
    snp_prefix = config_file['data_specs']['snp_prefix']
    idx_col = config_file['data_specs']['index']
    snp_data = df.select(pl.col(idx_col, f'^{snp_prefix}.*$'))

    env_vars = ['sample'] + config_file['data_specs']['env_variables']
    # print(env_vars)
    env_data = df.select([pl.col(f'^{var}_.*$') for var in env_vars])

    target_name = config_file['data_specs']['target']
    if target_name in df.columns:
        target_data = df[[idx_col, target_name]]
    else:
        target_data = df.select([idx_col]).with_columns(pl.lit(None).alias(target_name))
        print(f'{target_name} not found, returned null placeholder')

    print('Data splitting successful')

    return snp_data, env_data, target_data


def process_env_data(df, config_file):

    """
    Process env data file output from split_frame. Does feature engineering based on the timestamp variables from raw
    :param df: pl.Dataframe. Env data
    :param config_file: custom file
    :return: pl.Dataframe with engineered columns. Drops timestamp variables.
    """
    env_vars = config_file['data_specs']['env_variables']
    max_timestamp = config_file['data_specs']['time_points']

    # handle null data column wise
    impute_strategy = config_file['env_preprocessing']['impute_strategy']
    t_cols = [var for var in df.columns if var in env_vars]

    if impute_strategy == 'median':
        df = df.with_columns([
            pl.col(t_col).fill_null(pl.col(t_col).median()) for t_col in t_cols
        ])
    else: # mean, mode, min, max
        df = df.with_columns([
            pl.col(t_col).fill_null(strategy=impute_strategy) for t_col in t_cols
        ])

    # do engineering on non missing data
    df = df.with_columns([
        (pl.col(f'irrigation_ml_{t}') / pl.col(f'temp_{t}') )
        .alias(f'irr_temp_ratio_{t}')
        for t in range(1, max_timestamp+1)
    ])

    df = df.with_columns([
        (pl.col(f'temp_{t}') * (100 - pl.col(f'humidity_{t}')))
        .alias(f'vpd_{t}')
        for t in range(1, max_timestamp+1)
    ])

    df = df.with_columns([
        (pl.col(f'co2_ppm_{t}') * pl.col(f'light_hours_{t}'))
        .alias(f'co2_light_{t}')
        for t in range(1, max_timestamp+1)
    ])

    # mean_df = df.select([pl.mean_horizontal(pl.col(f'^{var}_.*$').alias(f'{var}_mean')) for var in env_vars])
    engineered_vars = config_file['data_specs']['engineered_variables']
    env_vars_to_process = env_vars + engineered_vars
    # print(env_vars_to_process)
    aggs = []
    for var in env_vars_to_process:
        pattern = f'^{var}_.*$'
        # print(pattern)
        aggs.extend([
            # pl.col('sample_id').alias('sample_id'),
            pl.mean_horizontal(pl.col(pattern)).alias(f'{var}_mean'), # auto ignores missing
            pl.min_horizontal(pl.col(pattern)).alias(f'{var}_min'),
            pl.max_horizontal(pl.col(pattern)).alias(f'{var}_max'),
            # df.select(pl.col(pattern)).map_rows(lambda: row np.std(row)).to_series().alias(f'{var}_std')
            pl.concat_list(pl.col(pattern)).list.drop_nulls().list.std().alias(f'{var}_std') # deals with missing
            # (pl.std_horizontal(pl.col(pattern)).alias(f'f{var}_std'))

        ])

    clean_df = df.select(pl.col('sample_id'), *aggs)

    print('New env features engineered from timestamp. Timestamp columns dropped')

    return clean_df


def update_env_data(df, config_file):
    """
    This one updates the env_data file based on the EDA results like colinearity
    :param df: pl.Dataframe of cleaned env_data from process_env_data
    :param config_file: custom file
    :return: pl.Dataframe with final engineered columns
    """
    keep = config_file['data_specs']['stats_to_keep']
    drop_env = config_file['data_specs']['env_to_drop']
    # keep = ['id', 'mean', 'std']
    new_df = df.select([pl.col(f'^*._{var}$') for var in keep])
    # drop humidity but keep vpd
    update_df = new_df.drop(drop_env)
    return update_df
    # return new_df


# def env_metadata(df, config_file):
#     """
#     Creates and save env metadata for future use
#     :param df: pl.Dataframe from update_env_data
#     :param config_file: custom file
#     :return: None
#     """
#     # clean_env = pl.DataFrame()
#     env_metadata = {}
#     env_vars = config_file['data_specs']['env_variables'] + config_file['data_specs']['engineered_variables']
#     # print(env_vars)
#     # print(f'Input shape: {df.shape}')
#     for var in env_vars: # gets temp, irrigation ...
#         cols = [feat for feat in df.columns if feat.startswith(var)] # gets temp_mean, temp_std ...
#         for col in cols: # gets temp_mean ...
#             # clean_env = df[col].drop_nulls() # handled in process_env_data
#
#             env_metadata[col] = {
#                 'mean': df[col].mean(),
#                 'std': df[col].std()
#             }
#
#     # print(f'Cleaned shape for stats calc: {df.shape}')
#     out_path = Path(processed_path / 'env_metadata.json')
#     with open(out_path, 'w') as f:
#         json.dump(env_metadata, f, indent=4)
#
#     print(f'Env Metadata saved to {out_path}')



def save_processed_files(df_snp, df_env, df_target, config_file, ROOT):

    """
    Saves all processed data.
    :param df_snp: pl.Dataframe. output from split_frame
    :param df_env: pl.Dataframe. output from update_env_data
    :param df_target: pl.Dataframe. output from split_frame
    :param config_file: custom file
    :param ROOT: home path
    :return: None
    """

    processed_path = config_file['inputs']['processed_data_dir']
    df_snp = df_snp.sort('sample_id')
    df_env = df_env.sort('sample_id')
    df_target = df_target.sort('sample_id')

    assert df_snp.shape[0] == df_env.shape[0] == df_target.shape[0]

    out_snp = ROOT / processed_path / config_file['outputs']['processed_snp']
    out_env = ROOT / processed_path / config_file['outputs']['processed_env']
    out_target = ROOT / processed_path / config_file['outputs']['processed_target']

    df_snp.write_parquet(out_snp)
    df_env.write_parquet(out_env)
    df_target.write_parquet(out_target)

    print(f'Training files saved in {processed_path}. Ready for modeling.')


