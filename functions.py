import os
import pdftotext


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
