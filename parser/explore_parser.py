from functions import print_pretty_table, print_pretty_list, show_extended_info
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('output/parser_marks_df.csv',
                     parse_dates=['date'],
                     dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})

    # Explore parser output: show dataframe info
    print('\nShow dataframe info\n')
    print(print_pretty_table(show_extended_info(df)))

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.iloc[:, 0:8].head()))
    print(print_pretty_table(df.iloc[:, 8:].head()))

    # Check unique values in 'category', 'county', 'criteria'
    choose_cols = ['category', 'county', 'criteria']
    output_pt = [['column', 'unique values']]
    for col in choose_cols:
        show_col_unique = df[col].unique()
        show_col_unique = sorted(show_col_unique)
        show_col_unique = print_pretty_list(show_col_unique)
        output_pt.append([col, show_col_unique])

    print('\nheck unique values in \'category\', \'county\', \'criteria\'\n')
    print(print_pretty_table(output_pt))
