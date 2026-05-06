import json
import polars as pl


def snp_processing(df, config_file, ROOT):

    """
    Creates and store metadata of snps for recoding and imputation. Calc MAF for recoding
    :param df: full train df. pl.Dataframe
    :param config_file: custom file
    :param ROOT: home path
    :return: None but it is saved for reuse
    """
    snp_prefix = config_file['data_specs']['snp_prefix']
    snp_cols = [mk_name for mk_name in df.columns if mk_name.startswith(snp_prefix)]
    # print(snp_cols)

    snp_metadata = {}
    for col in snp_cols:
        counts = df[col].value_counts().sort('count', descending=True)
        mapping = {val: i for i, val in enumerate(counts[col].to_list())}
        # recode_map[col] : mapping

        mode_value = 0 # assumption is that missing allele is the major allele

        snp_metadata[col] = {
            'recode_map': mapping,
            'impute_value': mode_value
        }

    processed_path = config_file['inputs']['processed_data_dir']
    out_path = ROOT / processed_path / config_file['outputs']['snp_metadata']

    with open(out_path, 'w') as f:
        json.dump(snp_metadata, f, indent=4)
    print(f'SNP metadata saved to {out_path}')
    # return snp_metadata



def snp_recode_impute(df, config_file, ROOT):

    """
    Uses the snp metadata to recode/impute SNPs
    :param df: pl.Dataframe. full train df
    :param config_file: custom file
    :param ROOT: home path
    :return: same df but recoded as 0, 1, 2
    """

    processed_path = config_file['inputs']['processed_data_dir']
    snp_path = ROOT / processed_path / config_file['outputs']['snp_metadata']
    print(snp_path)
    if snp_path.exists():
        with open(snp_path, 'r') as f:
            snp_metadata = json.load(f)
    print(f'SNP metadata loaded successfully: {len(snp_metadata)} items')

    for col, meta in snp_metadata.items():
        df = df.with_columns(pl.col(col)
                                 .replace_strict(meta['recode_map'], default=None)
                                 .fill_null(meta['impute_value'])
                                 .cast(pl.Int8)
                                 )
    print('SNP recoded to 0, 1, 2')
    return df


def quality_checks(df, config_file):

    """
    Does QC on data.
    1. Calc and prints total missing data. 2. Filter rows by target bounds set in config
    3. Sets out of bounds to null for env set in config. To be imouted with median.
    4. Removes columns with near zero variance
    5. Removes columns with missing data above set threshold in config
    :param df: pl.Dataframe
    :param config_file: custom file
    :return: cleaned/QC'ed df
    """
    target_col = config_file['data_specs']['target']
    targ_min, targ_max = config_file['data_specs']['target_bounds']
    nzv_threshold = config_file['data_specs']['nzv_threshold']
    max_miss = config_file['data_specs']['max_miss']
    # env_cols = config_file['data_specs']['env_variables']
    idx_col = config_file['data_specs']['index']
    no_nzv_list = [idx_col, target_col]

    total_missing_data = df.null_count().sum_horizontal().item()
    print(f'Total missing data: {total_missing_data}')

    dropped=0
    # allows for new individual without target data
    if target_col in df.columns:
        initial_n = df.height
        df = df.filter(
            (pl.col(target_col) >= targ_min) & (pl.col(target_col) <= targ_max)
        )
        # print(df.height)
        final_n = df.height
        dropped = initial_n - final_n

    if dropped > 0:
        print(f'Dropped {dropped} rows due to target bounds')

    env_lim = config_file['env_preprocessing']['env_bounds']
    # print(env_lim.items())
    # remove out of bounds in env variables
    for var, (var_min, var_max) in env_lim.items(): # temp, (5, 50)
        cols = [env_time for env_time in df.columns if env_time.startswith(var)] # gets all the temp timestamps

        df = df.with_columns(
            pl.when(pl.col(col).is_between(var_min, var_max))
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
            for col in cols
        )

    # remove cols with near zero var
    numeric_cols = [col for col in df.select(pl.col(pl.NUMERIC_DTYPES)).columns
                    if col not in no_nzv_list]
    # print(len(numeric_cols))

    # calc var for each col
    var_df = df.select([pl.col(col).var().alias(col) for col in numeric_cols])
    # get cols to drop
    cols_drop = []
    for col in numeric_cols:
        var_val = var_df[col].item()
        if var_val is None or var_val <= nzv_threshold:
            cols_drop.append(col)

    if cols_drop:
        df = df.drop(cols_drop)

        print(f'len{cols_drop} columns dropped due to NZV threshold: {nzv_threshold}')

    # remove columns with more than max_miss threshold
    null_counts = df.null_count()
    # print(null_counts)
    total_rows = df.height
    null_cols = [
        col for col in df.columns
        if (null_counts[col][0] / total_rows) > max_miss
    ]
    df = df.drop(null_cols)
    print(f'Dropped {len(null_cols)} columns: exceed max null threshold of {max_miss}')

    print('Quality checks completed')

    return df

