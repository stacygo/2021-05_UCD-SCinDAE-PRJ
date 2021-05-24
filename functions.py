import os
import pdftotext
import pandas as pd
from tabulate import tabulate


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
