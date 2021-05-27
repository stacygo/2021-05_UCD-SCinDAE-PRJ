from functions import write_df_to_csv, print_pretty_table, show_extended_info, boxplot_df_to_png
import pandas as pd
import numpy as np

if __name__ == '__main__':
    year_start = 1996
    year_end = 2019
    year_range = range(year_start, year_end + 1)

    # Read the report with the results of crawling from a .csv file
    df = pd.read_csv('output/crawler_pdfs_df.csv')

    # Show dataframe info
    print('\nShow dataframe info\n')
    print(print_pretty_table(show_extended_info(df)))

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # 1. Check crawler stats

    # > Step 1: Check the number of pdfs crawled
    output_pt = ([['dataframe', 'shape'], ['pdfs crawled', df.shape]])

    # > Step 2: Check the number of pdfs downloaded / available for download
    find_pdf_success = df['pdf_success'] == 1
    output_df = df[find_pdf_success]
    output_pt.append(['pdfs downloaded', output_df.shape])

    # > Step 3: Check the number of pdfs to parse
    choose_cols = ['pdf_name', 'year', 'county']
    output_df = output_df[choose_cols].drop_duplicates()
    output_pt.append(['pdfs to parse', output_df.shape])

    # > Step 4: Print the results to a console
    print('\n1. Check crawler stats\n')
    print(print_pretty_table(output_pt))

    # 2. Check year mismatches

    # > Step 1: Add a column 'pdf_year' with a year extracted from 'pdf_name'
    df['pdf_year'] = df['pdf_name'].apply(lambda x: int(x[0:4]))

    # > Step 2: Create a dataframe with the filtered rows / cols
    find_year_mismatch = df['pdf_year'] != df['year']
    choose_cols = ['pdf_name', 'pdf_year', 'year', 'county', 'pdf_found', 'pdf_success']
    output_df = df[find_year_mismatch][choose_cols].reset_index(drop=True)

    # > Step 3: Print the results to a console
    print('\n2. Check year mismatches\n')
    print(print_pretty_table(output_df))

    # > Step 4: Write a report on year mismatches to a .csv file
    write_df_to_csv(output_df, 'output/crawler_pdfs_df_02_year_mismatch.csv')

    # 3. Sort years by pdf_success

    # > Step 1: Create a dataframe with the filtered rows / cols
    choose_cols = ['year', 'pdf_found', 'pdf_success']
    groupby_cols = ['year']
    output_df = df[choose_cols].groupby(by=groupby_cols).sum().reset_index()
    output_df = output_df.sort_values(by=['pdf_success'])

    # > Step 2: Print the results to a console
    print('\n4. Sort years by pdf_success\n')
    print(print_pretty_table(output_df))

    # 4. Show distribution of pdf_success per county by years

    # > Step 1: Create a dataframe with the filtered rows / cols
    choose_cols = ['year', 'county', 'pdf_found', 'pdf_success']
    groupby_cols = ['year', 'county']
    output_df = df[choose_cols].groupby(by=groupby_cols).sum().reset_index()

    # > Step 2: Make a plot to show the distribution, and write it to a .png file
    output_file_name = 'output/crawler_pdfs_df_03_box_years.png'
    output_plot = {'df_x': 'year', 'df_y': 'pdf_success',
                   'title': 'Distribution of downloaded pdfs per county by years',
                   'x_label': 'Year', 'y_label': 'Downloaded pdfs per county'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (6.7, 3))

    # 5. Pivot by county vs year, and show the outliers

    # > Step 1: Create a dataframe using table pivoting
    output_df = df.pivot_table(index=['county'], columns=['year'], values='pdf_success',
                               aggfunc=np.sum, fill_value=0, margins=True, margins_name='(total)')

    # > Step 2: Add columns with 0 values for the years without any pdfs found
    for year in year_range:
        if year not in output_df:
            output_df[year] = 0

    # > Step 3: Sort the columns
    output_df = output_df.rename(columns={'(total)': -1}).sort_index(axis=1)
    output_df = output_df.rename(columns={-1: '(total)'})

    # > Step 4: Print the results to a console
    print('\n5. Pivot by county vs year, and show the outliers\n')
    show_columns = [2006, 2010, 2008, 2005, 2011]
    print(print_pretty_table(output_df[show_columns]))

    # > Step 5: Write a report on county vs year pivoting to a .csv file
    write_df_to_csv(output_df.reset_index(), 'output/crawler_pdfs_df_05_pt_county_year.csv')
