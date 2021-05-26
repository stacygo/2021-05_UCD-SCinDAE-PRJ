import os
import pdftotext
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


def read_txt_splitlines(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()


def read_pdf(file_path):
    with open(file_path, "rb") as file:
        return pdftotext.PDF(file)


def write_pdf(output_pdf, output_file_name):
    output_file_dir = os.path.dirname(output_file_name)
    os.makedirs(output_file_dir, exist_ok=True)

    with open(output_file_name, "wb") as file:
        file.write(output_pdf)


def write_df_to_csv(output_df, output_file_name):
    output_file_dir = os.path.dirname(output_file_name)
    os.makedirs(output_file_dir, exist_ok=True)

    output_df.to_csv(output_file_name, index=False)


def print_pretty_table(table_to_print, format_float='.1f'):
    if isinstance(table_to_print, list):
        return tabulate(table_to_print, headers='firstrow', tablefmt='rst', stralign='l', floatfmt=format_float)

    if isinstance(table_to_print, pd.core.frame.DataFrame):
        return table_to_print.to_markdown(tablefmt="rst", stralign='l', floatfmt=format_float)

    return 'Sorry, I can only print lists and pandas DataFrames.'


def print_pretty_list(list_to_print):
    if isinstance(list_to_print, list):
        list_to_print = list(map(str, list_to_print))
        if len(list_to_print) > 50:
            list_to_print = list_to_print[0:5]
            list_to_print.append('(...more than 50 values)')
            return '\n'.join(list_to_print)
        if len(list_to_print) > 10:
            return '\n'.join(list_to_print)
        return ', '.join(list_to_print)

    return 'Sorry, I can only print lists.'


def show_extended_info(df_to_describe):
    if isinstance(df_to_describe, pd.core.frame.DataFrame):
        df_with_info = pd.DataFrame(df_to_describe.dtypes, columns=['dtype'])
        df_with_info = pd.concat([df_with_info, df_to_describe.describe()[1:].T], axis=1)
        df_with_info = df_with_info.where(df_with_info.notnull(), None)
        df_with_info['not_null'] = df_to_describe.count()
        df_with_info['is_null'] = df_to_describe.isnull().sum()
        df_with_info['unique'] = df_to_describe.nunique()
        df_with_info.index.name = 'column'
        return df_with_info.reset_index()

    return 'Sorry, I can only describe pandas DataFrames.'


def get_feature_importances(estimator):
    if hasattr(estimator, 'coef_'):
        return estimator.coef_
    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_

    return np.array([])


def set_plot_layout(plot_size):
    plt.rcParams['font.family'] = 'Tahoma'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelweight'] = 'bold'

    plt.figure(figsize=plot_size, dpi=200)
    plt.tight_layout()


def boxplot_df_to_png(output_df, output_file_name, output_plot, plot_size=(6.4, 4.8)):
    set_plot_layout(plot_size)

    ax = sns.boxplot(x=output_plot['df_x'], y=output_plot['df_y'], data=output_df,
                     color='lightgrey', linewidth=0.75, fliersize=2.0)
    sns.despine()

    plt.title(output_plot['title'])

    plt.xlabel(output_plot['x_label'])
    if len(ax.get_xticklabels()) > 15:
        plt.xticks(rotation=45, ha='right')
    plt.ylabel(output_plot['y_label'])

    if output_file_name == '':
        plt.show()
    else:
        output_file_dir = os.path.dirname(output_file_name)
        os.makedirs(output_file_dir, exist_ok=True)

        plt.savefig(output_file_name, bbox_inches='tight')


def barplot_df_to_png(output_df, output_file_name, output_plot, plot_size=(6.4, 4.8)):
    set_plot_layout(plot_size)

    ax = sns.barplot(x=output_plot['df_x'], y=output_plot['df_y'], data=output_df,
                     estimator=output_plot['estimator'],
                     color='darkgray', linewidth=0.75)
    if 'ylim' in output_plot:
        ax.set(ylim=output_plot['ylim'])
    sns.despine()

    plt.title(output_plot['title'])

    plt.xlabel(output_plot['x_label'])
    if len(ax.get_xticklabels()) > 15:
        plt.xticks(rotation=45, ha='right')
    plt.ylabel(output_plot['y_label'])

    if output_file_name == '':
        plt.show()
    else:
        output_file_dir = os.path.dirname(output_file_name)
        os.makedirs(output_file_dir, exist_ok=True)

        plt.savefig(output_file_name, bbox_inches='tight')


def heatmap_df_to_png(output_df, output_file_name, output_plot, plot_size=(6.4, 4.8)):
    set_plot_layout(plot_size)

    cmap = sns.dark_palette("#FFFFFF", as_cmap=True, reverse=True)  # "YlGnBu"
    ax = sns.heatmap(output_df, cmap=cmap, annot=True, fmt='.0f')
    sns.despine()

    plt.title(output_plot['title'])

    plt.xlabel(output_plot['x_label'])
    if len(ax.get_xticklabels()) > 15:
        plt.xticks(rotation=45, ha='right')
    plt.ylabel(output_plot['y_label'])

    if output_file_name == '':
        plt.show()
    else:
        output_file_dir = os.path.dirname(output_file_name)
        os.makedirs(output_file_dir, exist_ok=True)

        plt.savefig(output_file_name, bbox_inches='tight')
