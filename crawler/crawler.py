# https://www.google.com/search?q=site:https://www.tidytowns.ie/+2006+pdf&start=100

from functions import read_txt_splitlines, write_pdf, write_df_to_csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
import os


def get_pdfs_by_year_county(year, county):
    url = 'https://www.tidytowns.ie/reports/?report_year=' + year + '&report_county=' + county
    print(url)

    # Send the request; catch the response and extract the html
    reports_html = requests.get(url).text

    # Create a BeautifulSoup object from the HTML
    reports_soup = BeautifulSoup(reports_html, features="html.output")

    # Find <ul> element with class="reports-list"
    reports_soup_ul = reports_soup.find('ul', class_='reports-list')

    # Find links by looking for <li> elements
    reports_soup_ul_li = [] if reports_soup_ul is None else reports_soup_ul.find_all('li')

    year_county_reports_list = []

    if len(reports_soup_ul_li) > 0:
        for code in reports_soup_ul_li:
            code_a_href = code.find('a')['href']
            if re.search(r'\.pdf$', code_a_href) is not None:
                year_county_report = {'pdf_name': os.path.basename(code_a_href),
                                      'pdf_path': os.path.dirname(code_a_href),
                                      'pdf_found': 1,
                                      'pdf_success':
                                          download_pdf_report(code_a_href, 'output/pdfs', year, county)}
                year_county_reports_list.append(year_county_report)

    return year_county_reports_list


def download_pdf_report(url, output_dir, year, county):
    file_dir = output_dir + '/' + year + '/' + county + '/'
    file_name = os.path.basename(url)
    file_path = file_dir + file_name

    if not os.path.exists(file_path):
        print('Processing ' + url + '...')
        response = requests.get(url)
        time.sleep(3)
        if response.status_code == 200:
            write_pdf(response.content, file_path)
        else:
            return 0

    return 1


if __name__ == '__main__':
    year_start = 1996
    year_end = 2019
    year_range = range(year_start, year_end + 1)
    counties_list = read_txt_splitlines('input/counties.txt')

    crawler_pdfs_df = pd.DataFrame()

    for year in year_range:
        for county in counties_list:
            crawler_pdfs = pd.DataFrame(get_pdfs_by_year_county(str(year), county))
            crawler_pdfs['year'] = year
            crawler_pdfs['county'] = county
            crawler_pdfs_df = crawler_pdfs_df.append(crawler_pdfs)

    write_df_to_csv(crawler_pdfs_df, 'output/crawler_pdfs_df.csv')
