from functions import write_df_to_csv, print_pretty_table, show_extended_info, boxplot_df_to_png
import pandas as pd
import numpy as np

if __name__ == '__main__':
    year_start = 1996
    year_end = 2019
    year_range = range(year_start, year_end + 1)

    df = pd.read_csv('output/crawler_pdfs_df.csv')

    # Explore crawler output: show dataframe info
    print('\nShow dataframe info\n')
    print(print_pretty_table(show_extended_info(df)))

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # Show dataframe stats
    output_pt = ([['dataframe', 'shape'], ['pdfs crawled', df.shape]])

    find_pdf_success = df['pdf_success'] == 1
    output_pt.append(['pdfs downloaded', df[find_pdf_success].shape])

    choose_cols = ['pdf_name', 'year', 'county']
    output_pt.append(['pdfs to parse', df[find_pdf_success][choose_cols].drop_duplicates().shape])

    print('\nShow dataframe stats\n')
    print(print_pretty_table(output_pt))

    # Check year mismatches
    df['pdf_year'] = df['pdf_name'].apply(lambda x: int(x[0:4]))
    find_year_mismatch = df['pdf_year'] != df['year']

    print('\nCheck year mismatches\n')
    choose_cols = ['pdf_name', 'pdf_year', 'year', 'county', 'pdf_found', 'pdf_success']
    print(print_pretty_table(df[find_year_mismatch][choose_cols]))

    write_df_to_csv(df[find_year_mismatch], 'output/crawler_pdfs_df_year_mismatch.csv')

    # Show distribution of pdf_success per county by years
    choose_cols = ['year', 'county', 'pdf_found', 'pdf_success']
    groupby_cols = ['year', 'county']
    output_df = df[choose_cols].groupby(by=groupby_cols).sum().reset_index()

    output_file_name = 'output/crawler_pdfs_df_years_distr.png'
    output_plot = {'df_x': 'year', 'df_y': 'pdf_success',
                   'title': 'Distribution of downloaded pdfs per county by years',
                   'x_label': 'Year', 'y_label': 'Downloaded pdfs per county'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (6.7, 3))

    # Sort years by pdf_success
    choose_cols = ['year', 'pdf_found', 'pdf_success']
    groupby_cols = ['year']
    output_df = df[choose_cols].groupby(by=groupby_cols).sum().reset_index()
    output_df = output_df.sort_values(by=['pdf_success'])

    print('\nSort years by pdf_success\n')
    print(print_pretty_table(output_df))

    # Pivot by county vs year, and show the outliers
    output_df = df.pivot_table(index=['county'], columns=['year'], values='pdf_success',
                               aggfunc=np.sum, fill_value=0, margins=True, margins_name='(total)')

    for year in year_range:
        if year not in output_df:
            output_df[year] = 0

    output_df = output_df.rename(columns={'(total)': -1}).sort_index(axis=1)
    output_df = output_df.rename(columns={-1: '(total)'})

    print('\nPivot by county vs year, and show the outliers\n')
    show_columns = [2006, 2010, 2008, 2005, 2011]
    print(print_pretty_table(output_df[show_columns]))

    write_df_to_csv(output_df.reset_index(), 'output/crawler_pdfs_df_pivot_county_year.csv')
