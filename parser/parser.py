from functions import read_pdf, write_df_to_csv
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm


def generate_file_paths(file_dir):
    file_paths = []

    for root, dirs, files in os.walk(file_dir):
        for file_name in files:
            if file_name.endswith(".pdf"):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)

    return file_paths


def parse_pdf_to_marks(file_name):
    not_to_parse = {'../crawler/output/pdfs/2005/limerick/2005 COUNTY LIMERICK BROADFORD LIMK 510.pdf':
                        'input/not_to_parse/2005_limerick_broadford.csv'}

    if file_name in not_to_parse:
        pdf_parsed = pd.read_csv(not_to_parse[file_name],
                                 parse_dates=['date'],
                                 dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
    else:
        pdf = read_pdf(file_name)
        pdf_parsed = parse_pdf_page_to_marks(pdf[0])

    return pdf_parsed


def parse_pdf_page_to_marks(pdf_page):
    marks_dict = {}

    marks_town = re.search(r'Centre\s*:\s+(.+?)($|\s{3,}|\s-\s)', pdf_page)
    if marks_town is not None:
        marks_dict['town'] = marks_town.group(1).title()

    marks_county = re.search(r'County\s*:\s+(.+?)($|\s{3,}|\()', pdf_page)
    if marks_county is not None:
        marks_dict['county'] = marks_county.group(1).title()

    marks_category = re.search(r'Category\s*:\s+([A-H])', pdf_page)
    if marks_category is not None:
        marks_dict['category'] = marks_category.group(1)

    marks_dict['year'], marks_dict['date'] = parse_year_date(pdf_page)
    marks_df = parse_criteria(pdf_page, marks_dict)

    return marks_df


def parse_year_date(pdf_page):
    marks_year = re.search(r'Tidy Towns( Competition)?\s*([0-9]{4})', pdf_page)

    if marks_year is not None:
        marks_year = marks_year.group(2)
    else:
        marks_year = re.search(r'Year\s*:\s+([0-9]{4})', pdf_page)

        if marks_year is not None:
            marks_year = marks_year.group(1)
        else:
            marks_year = None

    marks_date = re.search(r'Date(\(s\)|\sof\sAdjudication)?\s*:\s+([0-9/-]+)', pdf_page)

    if marks_date is not None:
        # Date was found, let's check that it matches marks_year
        marks_date = marks_date.group(2).replace('/', '-')
        marks_date_split = marks_date.split('-')
        if len(marks_date_split) == 3:
            if marks_date_split[2][0] == '9':
                marks_date_split[2] = '19' + marks_date_split[2]
            else:
                marks_date_split[2] = str(marks_year)
            marks_date = '-'.join(marks_date_split)
        else:
            marks_date = None

    marks_year = int(marks_year) if marks_year is not None else np.nan
    marks_date = pd.to_datetime(marks_date, format='%d-%m-%Y') if marks_date is not None else pd.NaT

    return marks_year, marks_date


def parse_criteria(pdf_page, marks_dict):
    scenario, criteria = define_scenario_criteria(pdf_page, marks_dict['year'])

    # scenario = 1; criteria + 3 digit columns: total, previous, current
    if scenario == 1:
        search_type_dict = {'max': 1, 'previous': 2, 'current': 3}
    # scenario = 2; criteria + 3 digit columns: total, current, previous
    elif scenario == 2:
        search_type_dict = {'max': 1, 'current': 2, 'previous': 3}
    # scenario = 3; criteria + 2 digit columns: total, current
    elif scenario == 3:
        search_type_dict = {'max': 1, 'current': 2}

    marks_df = pd.DataFrame()

    if len(criteria) == 0:
        marks_dict_update = {'criteria': '-', 'mark': np.nan, 'max_mark': np.nan}
        marks_dict.update(marks_dict_update)
        marks_df = pd.DataFrame.from_dict(marks_dict)
    else:
        for marks_tuple in criteria:
            marks_dict_update = {'criteria': marks_tuple[0],
                                 'mark': int(marks_tuple[search_type_dict['current']]),
                                 'max_mark': int(marks_tuple[search_type_dict['max']])}
            marks_dict.update(marks_dict_update)
            marks_df = marks_df.append(marks_dict, ignore_index=True)

    return marks_df


def define_scenario_criteria(pdf_page, year):
    # Try to find strings with criteria + 3 digit columns
    criteria_re = r'^\s*([^0-9\n]*?)\s{3,}(\d{1,3})\s{3,}(\d{1,3})\s{3,}([0-9-]{1,3})$'
    criteria = re.findall(criteria_re, pdf_page, re.MULTILINE)
    # scenario = 0; default, no criteria has been found in pdf_page yet
    scenario = 0 if len(criteria) == 0 else 1

    if scenario == 1:
        # Define regex to find the sequence of columns with marks
        years_seq_regex = r'' + str(year - 1) + r'\D+' + str(year)
        years_seq = re.search(years_seq_regex, pdf_page)

        if years_seq is None:
            years_seq_regex = r'' + str(year) + r'\D+' + str(year - 1)
            years_seq = re.search(years_seq_regex, pdf_page)
            # scenario = 2; criteria + 3 digit columns: total, current, previous
            scenario = 0 if years_seq is None else 2
        else:
            # scenario = 1; criteria + 3 digit columns: total, previous, current
            scenario = 1
    else:
        # Try to find strings with criteria + 2 digit columns
        criteria_re = r'^\s*([^0-9\n]*?)\s{3,}(\d{1,3})\s{3,}(\d{1,3})$'
        criteria = re.findall(criteria_re, pdf_page, re.MULTILINE)
        # scenario = 3; criteria + 2 digit columns: total, current
        scenario = 0 if len(criteria) == 0 else 3

    return scenario, criteria


if __name__ == '__main__':
    file_paths = generate_file_paths('../crawler/output/pdfs')

    parser_marks_df = pd.DataFrame()

    for pdf_path in tqdm(file_paths):
        parser_marks = parse_pdf_to_marks(pdf_path)
        parser_marks['pdf_path'] = pdf_path
        parser_marks_df = parser_marks_df.append(parser_marks, ignore_index=True)

    # parser_marks_df = pd.melt(parser_marks_df, id_vars=['town', 'county', 'category', 'year', 'date'],
    #                           var_name='criteria', value_name='mark').dropna()
    write_df_to_csv(parser_marks_df, 'output/parser_marks_df.csv')
