from functions import write_df_to_csv
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    year_start = 1996
    year_end = 2019
    year_range = range(year_start, year_end + 1)

    crawler_pdfs_df = pd.read_csv('output/crawler_pdfs_df.csv')

    # Explore crawler output: view the first 5 rows
    print('\ncrawler_pdfs_df: view the first 5 rows\n')
    print(crawler_pdfs_df.head().to_markdown(tablefmt="pretty", stralign='l'))

    # Explore crawler output: show dataframe info
    print('\ncrawler_pdfs_df: show dataframe info\n')
    print(crawler_pdfs_df.info())

    # Explore crawler output: show dataframe shape vs rows with pdf_found=1
    pdf_success_1 = crawler_pdfs_df['pdf_success'] == 1
    crawler_pdfs_pt = ([['crawler_pdfs_df', crawler_pdfs_df.shape],
                        ['rows with pdf_found=1', crawler_pdfs_df[pdf_success_1].shape]])

    print('\ncrawler_pdfs_df: show dataframe shape vs rows with pdf_found=1\n')
    print(tabulate(crawler_pdfs_pt, headers=['dataframe', 'shape'], tablefmt='pretty', stralign='l'))

    # Explore crawler output: check year mismatches
    crawler_pdfs_df['pdf_year'] = crawler_pdfs_df['pdf_name'].apply(lambda x: int(x[0:4]))
    year_mismatch = crawler_pdfs_df['pdf_year'] != crawler_pdfs_df['year']
    crawler_pdfs_df_year_mismatch = crawler_pdfs_df[year_mismatch]

    write_df_to_csv(crawler_pdfs_df_year_mismatch, 'output/crawler_pdfs_df_year_mismatch.csv')

    print('\ncrawler_pdfs_df: check year mismatches\n')
    show_columns = ['pdf_name', 'pdf_year', 'year', 'county', 'pdf_found', 'pdf_success']
    print(crawler_pdfs_df_year_mismatch[show_columns].to_markdown(tablefmt="pretty", stralign='l', index=False))

    # Explore crawler output: sort years by pdf_success
    choose_columns = ['year', 'pdf_found', 'pdf_success']
    crawler_years_df_agg = crawler_pdfs_df[choose_columns].groupby(by=['year']).sum().reset_index()
    crawler_years_df_agg['year'] = crawler_years_df_agg['year'].apply(lambda x: str(x))
    crawler_years_df_agg.sort_values(by=['pdf_success'], inplace=True)

    sns.boxplot(y=crawler_years_df_agg['pdf_success'], palette='Set3')
    plt.ylabel('Number of downloaded pdfs')
    plt.savefig('output/crawler_pdfs_df_years_distr.png')

    print('\ncrawler_pdfs_df: sort years by pdf_success\n')
    print(crawler_years_df_agg.to_markdown(tablefmt="pretty", stralign='l', index=False))

    # Explore crawler output: pivot by county vs year
    crawler_pdfs_df_pivot = \
        crawler_pdfs_df.pivot_table(index=['county'], columns=['year'], values='pdf_success',
                                    aggfunc=np.sum, fill_value=0, margins=True, margins_name='(total)')

    for year in year_range:
        if year not in crawler_pdfs_df_pivot:
            crawler_pdfs_df_pivot[year] = 0

    crawler_pdfs_df_pivot.rename(columns={'(total)': -1}, inplace=True)
    crawler_pdfs_df_pivot.sort_index(axis=1, inplace=True)
    crawler_pdfs_df_pivot.rename(columns={-1: '(total)'}, inplace=True)

    write_df_to_csv(crawler_pdfs_df_pivot.reset_index(),
                    'output/crawler_pdfs_df_pivot_county_year.csv')

    print('\ncrawler_pdfs_df: pivot by county vs year\n')
    show_columns = [2006, 2010, 2008, 2011, 1996]
    print(crawler_pdfs_df_pivot[show_columns].to_markdown(tablefmt="pretty", stralign='l'))
