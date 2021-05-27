from functions import show_extended_info, print_pretty_table
from functions import boxplot_df_to_png, barplot_df_to_png, heatmap_df_to_png
import pandas as pd
import numpy as np


def explore_01_box_years_marks(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    output_df = df_to_explore[find_criteria_total].astype({'year': 'int32'})

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_01_box_years_marks.png'
    output_plot = {'df_x': 'year', 'df_y': 'mark',
                   'title': '\'TOTAL MARK\' by years',
                   'x_label': 'Year', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 2))


def explore_02_box_counties_marks(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['county_l1'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_02_box_counties_marks.png'
    output_plot = {'df_x': 'county_l1', 'df_y': 'mark',
                   'title': '\'TOTAL MARK\' by counties for 2019',
                   'x_label': 'County', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (6.6, 4))


def explore_02_bar_counties_count(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['county_l1'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_02_bar_counties_count.png'
    output_plot = {'df_x': 'county_l1', 'df_y': 'mark',
                   'estimator': np.count_nonzero,
                   'title': 'Adjudicated towns by counties for 2019',
                   'x_label': 'County', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (6.6, 1))


def explore_03_box_days_marks(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['date_dnn'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_03_box_days_marks.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'title': '\'TOTAL MARK\' by week days for 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 2))


def explore_03_bar_days_count(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['date_dnn'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_03_bar_days_count.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'estimator': np.count_nonzero, 'ylim': (50, 170),
                   'title': 'Adjudicated towns by week days for 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 1))


def explore_03_bar_days_marks_median(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['date_dnn'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_03_bar_days_marks_median.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'estimator': np.median, 'ylim': (270, 320),
                   'title': 'Median \'TOTAL MARK\' by week days for 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Median Mark'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 1))


def explore_04_box_months_marks(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['date'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_04_box_months_marks.png'
    output_plot = {'df_x': 'date_month', 'df_y': 'mark',
                   'title': '\'TOTAL MARK\' by months for 2019',
                   'x_label': 'Adjudication Month', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 2))


def explore_04_bar_months_count(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['date'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_04_bar_months_count.png'
    output_plot = {'df_x': 'date_month', 'df_y': 'mark',
                   'estimator': np.count_nonzero,
                   'title': 'Adjudicated towns by months for 2019',
                   'x_label': 'Adjudication Month', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 1))


def explore_05_box_categories_marks(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['category'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_05_box_categories_marks.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'title': '\'TOTAL MARK\' by categories for 2019',
                   'x_label': 'Categories', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 2))


def explore_05_bar_categories_count(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['category'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_05_bar_categories_count.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'estimator': np.count_nonzero,
                   'title': 'Adjudicated towns by categories for 2019',
                   'x_label': 'Categories', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 1))


def explore_05_bar_categories_marks_median(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].sort_values(by=['category'])

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_05_bar_categories_marks_median.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'estimator': np.median, 'ylim': (270, 350),
                   'title': 'Median \'TOTAL MARK\' by categories for 2019',
                   'x_label': 'Category', 'y_label': 'Median Mark'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.2, 1))


def explore_06_heat_weeks_days(df_to_explore):
    # Create a dictionary for mapping week day numbers into week day names
    wd_names = {0: 'Mo', 1: 'Tu', 2: 'We', 3: 'Th', 4: 'Fr', 5: 'Sa', 6: 'Su'}

    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].pivot_table(index=['date_week'], columns=['date_dnn'], values='mark',
                                                       aggfunc=np.count_nonzero, fill_value=np.nan)
    output_df = output_df.rename(columns=wd_names)

    # View the first 5 rows after transformation
    print('\n6. View the first 5 rows after transformation\n')
    print(print_pretty_table(output_df.head()))

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_06_heatmap_weeks_days.png'
    output_plot = {'title': 'Adjudicated towns by weeks / week days for 2019',
                   'x_label': 'Week Day', 'y_label': 'Week of Year'}
    heatmap_df_to_png(output_df, output_file_name, output_plot, (6.6, 3))


def explore_07_heat_criteria_categories(df_to_explore):
    # Transform the dataset for exploration
    find_criteria_total = df_to_explore['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df_to_explore['year'] == 2019
    choose_rows = ~find_criteria_total & find_year_2019
    output_df = df_to_explore[choose_rows].pivot_table(index=['criteria_tidy'], columns=['category'], values='mark',
                                            aggfunc=np.mean, fill_value=0)

    # View the first 5 rows after transformation
    print('\n7. View the first 5 rows after transformation\n')
    print(print_pretty_table(output_df.head()))

    # Make a plot, and write it to a .png file
    output_file_name = 'output/cleaner_marks_df_2014_07_heat_criteria_categories.png'
    output_plot = {'title': 'Mean marks by criteria / categories / 2019',
                   'x_label': 'Category', 'y_label': 'Criteria'}
    heatmap_df_to_png(output_df, output_file_name, output_plot, (6.6, 3))


if __name__ == '__main__':
    # Read the clean dataset from a .csv file
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv',
                     parse_dates=['date'],
                     dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})

    # Show dataframe info
    print('\nShow dataframe info\n')
    print(print_pretty_table(show_extended_info(df)))

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # 1. Explore 'TOTAL MARK' distribution dynamics by years
    explore_01_box_years_marks(df)

    # 2. Explore 'TOTAL MARK' distribution for 2019 by counties
    explore_02_box_counties_marks(df)
    explore_02_bar_counties_count(df)

    # Add a column 'date_dnn' with a week day number extracted from 'date'
    df['date_dnn'] = df['date'].dt.weekday

    # Add a column 'date_dn' with a week day name extracted from 'date' and shortened to 2 symbols
    df['date_dn'] = df['date'].dt.day_name().apply(lambda x: x[0:2])

    # 3. Explore 'TOTAL MARK' distribution for 2019 by week days
    explore_03_box_days_marks(df)
    explore_03_bar_days_count(df)
    explore_03_bar_days_marks_median(df)

    # Add a column 'date_month' with a month name extracted from 'date' and shortened to 3 symbols
    df['date_month'] = df['date'].dt.month_name().apply(lambda x: x[0:3])

    # 4. Explore 'TOTAL MARK' distribution for 2019 by months
    explore_04_box_months_marks(df)
    explore_04_bar_months_count(df)

    # 5. Explore 'TOTAL MARK' distribution for 2019 by categories
    explore_05_box_categories_marks(df)
    explore_05_bar_categories_count(df)
    explore_05_bar_categories_marks_median(df)

    # Add a column 'date_week' with a week number extracted from 'date'
    df['date_week'] = df['date'].dt.isocalendar().week

    # 6. Explore heatmap of adjudicated towns by weeks / week days for 2019
    explore_06_heat_weeks_days(df)

    # 7. Explore heatmap of marks by criteria / categories for 2019
    explore_07_heat_criteria_categories(df)
