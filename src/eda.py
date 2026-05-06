
import math
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import seaborn as sns


def target_distribution(df, config_file, save_path=None):
    """
    Plots distribution of target (y)
    :param df: pl.Dataframe when train, np.ndaray when test
    :param config_file: custom file
    :param save_path: bool set in config. Saves plot if True
    """
    target_col = config_file['data_specs']['target']
    if isinstance(df, pl.DataFrame):
        vals = df[target_col]
    else:
        vals = df
    fig, ax = plt.subplots(1, 2, figsize = (12, 5))
    # ax = ax[0]
    sns.histplot(vals, kde=True, ax=ax[0])
    ax[0].set_xlabel(target_col.title())

    sns.boxplot(vals, ax=ax[1])
    ax[1].set_ylabel(target_col.title())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def env_features_target_corr_plots(df, config_file, save_path=None):
    """
    Line plot of env timestamps and feature correlation
    :param df: pl.Dataframe
    :param config_file: custom file
    :param save_path: bool set in config. Saves plot if True
    """
    target_col = config_file['data_specs']['target']
    env_vars = config_file['data_specs']['env_variables']

    num_of_vars = len(env_vars)
    cols = config_file['plotting']['cols']
    rows = math.ceil(num_of_vars/cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    ax_flat = ax.flatten()

    for  idx, var in enumerate(env_vars):
        var_lst = [
            (f'{var}_{i}') for i in range(1,6)
        ]

        temp_corr = [
            df.select(pl.corr(col, target_col)).item()
            for col in var_lst
        ]
        # for r in range(3):
        #     for c in range(2):
        ax_flat[idx].plot(range(1,6), temp_corr, marker='o', linestyle='--', color='orange')
        ax_flat[idx].set_xlabel(var)
        ax_flat[idx].set_ylabel('Correlation')
        ax_flat[idx].set_title(f'Corr: {var.title()} Timestamps vs {target_col.title()}')
        ax_flat[idx].set_xticks(range(1,6))

    # remv unused plots, start from where the loop ended
    for u in range(idx+1, len(ax_flat)):
        ax_flat[u].set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def feature_correlation_plots(df, config_file, add_var=None, save_path=None):

    """
    Plots correlation of features only
    :param df: pl.Dataframe
    :param config_file: custom file
    :param add_var: used more with engineered features. Passed as a list.
    :param save_path: bool set in config. Saves plot if True
    """
    # add_var: must be list when passed

    env_vars = config_file['data_specs']['env_variables']
    if not add_var:
        env_vars = env_vars
    else:
        env_vars = env_vars + add_var

    # adjust subplots by num of vars to plot
    num_of_vars = len(env_vars)
    cols = config_file['plotting']['cols']
    rows = math.ceil(num_of_vars/cols)

    # fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    ax_flat = ax.flatten()

    for idx, var in enumerate(env_vars):
        temp_cols = [t_stmp for t_stmp in df.columns if t_stmp.startswith(var)]
        temp_corr = np.round(df[temp_cols].to_pandas().corr(), 3)

        mask = np.triu(np.ones_like(temp_corr, dtype=bool), k=0)

        sns.heatmap(temp_corr.iloc[1:, :-1], mask = mask[1:, :-1], annot=True, cmap='RdBu_r', center=0, ax=ax_flat[idx])
        ax_flat[idx].set_title(f'Corr: {var.title()}')
        ax_flat[idx].tick_params(axis='x', labelrotation=70)
        ax_flat[idx].tick_params(axis='y', labelrotation=0)

    # remv unused plots, start from where the loop ended
    for u in range(idx+1, len(ax_flat)):
        ax_flat[u].set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def feature_timestamp_boxplots(df, config_file, save_path=None):
    """
    Plots boxplots of feature timestamps
    :param df: pl.Dataframe
    :param config_file: custom file
    :param save_path: bool set in config. Saves plot if True
    """

    env_vars = config_file['data_specs']['env_variables']
    num_of_vars = len(env_vars)
    cols = config_file['plotting']['cols']
    rows = math.ceil(num_of_vars/cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    ax_flat = ax.flatten()
    for idx, var in enumerate(env_vars):
        temp_cols = [t_stmp for t_stmp in df.columns if var in t_stmp]
        temp_data = df[temp_cols].to_pandas()

        sns.boxplot(temp_data, ax=ax_flat[idx])
        ax_flat[idx].tick_params(axis='x', labelrotation=70)
        ax_flat[idx].set_title(f'Distribution: {var.title()}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_all_correlation_plot(df, y, save_path=None):
    """
    Make correlation matrix of features with target and then plots
    :param df: pl.Dataframe
    :param y: pl.Dataframe
    :param save_path: bool set in config. Saves plot if True
    """

    temp_data = df.join(y, on='sample_id')
    temp_data = temp_data.drop('sample_id')
    # print(df.shape, temp_data.shape)
    temp_corr = temp_data.corr()
    temp_corr = np.round(temp_corr.to_pandas(), 2)
    # print(isinstance(temp_corr, pd.DataFrame))

    mask = np.ones_like(temp_corr, dtype=bool)
    mask_bool = np.triu(mask, k=0)

    temp_corr.index = temp_corr.columns

    plt.figure(figsize=(18, 9))
    sns.heatmap(temp_corr.iloc[1:, :-1], mask=mask_bool[1:, :-1], center=0, annot=True, cmap='RdBu_r')
    plt.title('Correlation across features and target')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def make_prediction_plots(y_pred, y_true, metrics, save_path=None):

    """
    Model evaluation on predicted data and makes three plots. Plot will only be made if y_true is available.
    1. Scatter of fitted vs true 2. Residual Distribution 3. Residual scatter with fitted
    :param y_pred: fitted values of new/test individuals. Results from build_kernels_predict_new_genotypes
    :param y_true: true scores of new/test individuals
    :param metrics: dict result from combine_and_save_metrics
    :param save_path: bool set in config. Saves plot if True
    """

    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten() if y_true is not None else None

    true_not_null = y_true is not None and not np.all(np.isnan(y_true))
    if true_not_null:
        rmp = metrics['External_test']['rMP']
        rmse = metrics['External_test']['RMSE']
        bias = metrics['External_test']['Bias']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]

        all_data = np.concat([y_true, y_pred])
        l_min, l_max = all_data.min(), all_data.max()
        ax.scatter(x=y_pred, y=y_true, alpha=0.6, edgecolor='w')
        ax.axline(xy1=(0, 0), slope=1, linestyle='-.', color='grey', label=('y=x'))
        ax.set_xlabel('Predicted Scores')
        ax.set_ylabel('True Scores')
        ax.set_xlim(l_min, l_max)
        ax.set_ylim(l_min, l_max)
        ax.set_title(f'True vs Predicted Scores\n(rMP: {rmp}, RMSE: {rmse}, Bias: {bias})')

        ax.legend()

        ax = axes[1]
        residual = y_true - y_pred
        sns.histplot(residual, bins=15, kde=True, ax=ax, color='skyblue')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Errors (Residuals)')

        ax = axes[2]
        ax.scatter(x=y_pred, y=residual, alpha=0.6, color='red', edgecolor='w')
        ax.axhline(y=0, linestyle='--', color='grey', label='Zero Residuals')
        ax.set_xlabel('Predicted Score')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted Scores')
        ax.legend()

        plt.suptitle(f'Model Evaluation on Test Data', fontsize=16, y=1.05)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
