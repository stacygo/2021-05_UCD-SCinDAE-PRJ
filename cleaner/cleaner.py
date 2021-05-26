from functions import write_df_to_csv, print_pretty_table, show_extended_info
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('../parser/output/parser_marks_df.csv',
                     parse_dates=['date'],
                     dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})

    # Explore parser output: show dataframe info
    print('\nShow dataframe info\n')
    print(print_pretty_table(show_extended_info(df)))

    # Clean 'category' column
    categories_df = pd.read_csv('input/categories.csv')
    df = df.merge(categories_df, how='left', on=['category'])

    # Clean 'county' column
    df['county_l1'] = df['county'].apply(lambda x: x.split(' ')[0])

    # Check & clean 'town' column

    # > Step 1: Add a column with a county from 'pdf_path'
    df['pdf_path_county'] = df['pdf_path'].apply(lambda x: x.split('/')[-1])

    # > Step 2: Process 'Ballingarry' to differentiate Ballingarry (North Tipperary) -- a civil parish
    # and a townland in North Tipperary with a population of 575 (as of 2006) from  Ballingarry
    # (North Tipperary) -- a village and civil parish in South Tipperary with a population of 269
    # (as of 2016)
    # See: https://en.wikipedia.org/wiki/Ballingarry,_North_Tipperary
    # See: https://en.wikipedia.org/wiki/Ballingarry,_South_Tipperary
    find_ballingarry = df['town'] == 'Ballingarry'

    find_tipperary_north = df['pdf_path_county'] == 'tipperary-north'
    df.loc[find_ballingarry & find_tipperary_north, 'town'] = 'Ballingarry (North)'

    find_tipperary_south = df['pdf_path_county'] == 'tipperary-south'
    df.loc[find_ballingarry & find_tipperary_south, 'town'] = 'Ballingarry (South)'

    # > Step 3: Make a csv file with a list of counties / towns for further post-processing of town names
    choose_cols = ['county_l1', 'town']
    output_df = df[choose_cols].drop_duplicates().sort_values(by=choose_cols)

    write_df_to_csv(output_df, 'output/cleaner_towns_df.csv')

    # > Step 4: Apply the results of manual post-processing of town names
    towns_df = pd.read_csv('input/towns.csv')
    df = df.merge(towns_df, how='left', on=['county_l1', 'town'])

    # Check & clean 'date' column

    # > Step 1: Add a column with a month
    df['date_month'] = df['date'].dt.month

    # > Step 2: Check the min / max / pd.NaT dates by years
    find_criteria_total = df['criteria'] == 'TOTAL MARK'
    output_df = df[find_criteria_total].groupby('year').agg(
        date_count=('date', len),
        date_min=('date', np.min),
        date_max=('date', np.max),
        date_nan=('date', lambda x: x.isnull().sum())
    )
    output_df['date_min'] = output_df['date_min'].dt.date
    output_df['date_max'] = output_df['date_max'].dt.date
    print('\nShow min / max / pd.NaT dates by years\n')
    print(print_pretty_table(output_df))

    # > Step 3: Make a csv file with a list of towns with the adjudication date out of May-September
    find_month_in = df['date_month'].isin([5, 6, 7, 8, 9])
    choose_cols = ['date', 'county_l1', 'town', 'pdf_path', 'pdf_name']
    output_df = df[~find_month_in][choose_cols].drop_duplicates()
    output_df = output_df.dropna(subset=['date']).sort_values(by=choose_cols).reset_index(drop=True)
    output_df['date'] = output_df['date'].dt.date
    print('\nShow a list of towns with the adjudication date out of May-September\n')
    print(print_pretty_table(output_df.head()))

    write_df_to_csv(output_df, 'output/cleaner_dates_df.csv')

    # > Step 4: Drop the rows without the date
    df = df.dropna(subset=['date'])

    # Check & clean 'criteria' column

    # > Step 1: Check the total mark dynamics by years and counties
    find_criteria_total = df['criteria'] == 'TOTAL MARK'
    output_df = \
        df[find_criteria_total].pivot_table(index=['county'], columns=['year'], values='max_mark',
                                            aggfunc=np.mean, fill_value=0, margins=True, margins_name='(total)')
    print('\nShow mean (total) max_mark by years\n')
    print(print_pretty_table(output_df.loc[['(total)']].T))

    # > Step 2: Keep in the dataset only years >= 2014
    find_years = (df['year'] >= 2014) & (df['year'] <= 2019)
    df = df[find_years]

    # > Step 3: Make a csv file with a list of criteria for further post-processing
    output_df = df.pivot_table(index=['criteria'], columns=['year'], values='max_mark',
                               aggfunc=np.mean, fill_value=0).reset_index()
    print('\nShow mean max_mark by years and criteria\n')
    print(print_pretty_table(output_df))

    write_df_to_csv(output_df, 'output/cleaner_criteria_2014_df.csv')

    # > Step 4: Apply the results of manual post-processing of criteria
    criteria_df = pd.read_csv('input/criteria_2014.csv')
    df = df.merge(criteria_df, how='left', on=['criteria'])

    # Drop some columns, and drop duplicates
    choose_cols = ['county', 'town', 'criteria', 'pdf_name', 'pdf_path', 'pdf_path_county', 'date_month']
    df = df.drop(choose_cols, axis=1).drop_duplicates()

    # Explore cleaner output: show dataframe info
    print('\nShow dataframe info after cleaning\n')
    print(print_pretty_table(show_extended_info(df)))

    # Make a csv file with a clean dataset
    write_df_to_csv(df, 'output/cleaner_marks_df_2014.csv')
