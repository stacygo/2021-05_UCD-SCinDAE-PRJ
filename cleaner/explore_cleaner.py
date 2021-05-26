from functions import print_pretty_table
from functions import boxplot_df_to_png, barplot_df_to_png, heatmap_df_to_png
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv',
                     parse_dates=['date'],
                     dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})

    wd_names = {0: 'Mo', 1: 'Tu', 2: 'We', 3: 'Th', 4: 'Fr', 5: 'Sa', 6: 'Su'}

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    df['date_dnn'] = df['date'].dt.weekday
    df['date_dn'] = df['date'].dt.day_name().apply(lambda x: x[0:2])
    df['date_month'] = df['date'].dt.month_name().apply(lambda x: x[0:3])
    df['date_week'] = df['date'].dt.isocalendar().week
    df['date'] = df['date'].dt.date

    # Report 1: Explore dynamics by years

    # Transform the dataset for exploration
    find_criteria_total = df['criteria_tidy'] == 'TOTAL MARK'
    output_df = df[find_criteria_total]

    # View the first 5 rows after transformation
    print('\nReport 1: View the first 5 rows after transformation\n')
    print(print_pretty_table(output_df.iloc[:, 0:9].head()))
    print(print_pretty_table(output_df.iloc[:, 9:].head()))

    # Show distribution of marks per town by years
    output_df = output_df.sort_values(by=['date'])
    output_file_name = 'output/cleaner_marks_df_2014_total_dyn_years.png'
    output_plot = {'df_x': 'year', 'df_y': 'mark',
                   'title': 'Distribution of marks by years',
                   'x_label': 'Year', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 2))

    # Report 2: Explore 'TOTAL MARK' rows for 2019

    # Transform the dataset for exploration
    find_criteria_total = df['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df['year'] == 2019
    choose_rows = find_criteria_total & find_year_2019
    output_df = df[choose_rows]

    # View the first 5 rows after transformation
    print('\nReport 2: View the first 5 rows after transformation\n')
    print(print_pretty_table(output_df.iloc[:, 0:9].head()))
    print(print_pretty_table(output_df.iloc[:, 9:].head()))

    # Show distribution of marks per town by counties
    output_df = output_df.sort_values(by=['county_l1'])
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_counties.png'
    output_plot = {'df_x': 'county_l1', 'df_y': 'mark',
                   'title': 'Distribution of marks by counties / 2019',
                   'x_label': 'County', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (6.7, 4))

    # Show histogram of adjudications by counties
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_counties_hist.png'
    output_plot = {'df_x': 'county_l1', 'df_y': 'mark',
                   'estimator': np.count_nonzero, # 'ylim': (50, 170),
                   'title': 'Histogram of adjudications by counties / 2019',
                   'x_label': 'County', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (6.7, 1))

    # Show distribution of marks per town by week days
    output_df = output_df.sort_values(by=['date_dnn'])
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_adj_weekdays.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'title': 'Distribution of marks by week days / 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 2))

    # Show histogram of adjudications by week days
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_adj_weekdays_hist.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'estimator': np.count_nonzero, 'ylim': (50, 170),
                   'title': 'Histogram of adjudications by week days / 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 1))

    # Show histogram of median mark by week days
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_adj_weekdays_hist_median.png'
    output_plot = {'df_x': 'date_dn', 'df_y': 'mark',
                   'estimator': np.median, 'ylim': (270, 320),
                   'title': 'Histogram of median marks by week days / 2019',
                   'x_label': 'Adjudication Week Day', 'y_label': 'Median Mark'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 1))

    # Show distribution of marks per town by adjudication months
    output_df = output_df.sort_values(by=['date'])
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_adj_months.png'
    output_plot = {'df_x': 'date_month', 'df_y': 'mark',
                   'title': 'Distribution of marks by months / 2019',
                   'x_label': 'Adjudication Month', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 2))

    # Show histogram of adjudications by months
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_adj_months_hist.png'
    output_plot = {'df_x': 'date_month', 'df_y': 'mark',
                   'estimator': np.count_nonzero,
                   'title': 'Histogram of adjudications by months / 2019',
                   'x_label': 'Adjudication Month', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 1))

    # Show distribution of marks per town by category
    output_df = output_df.sort_values(by=['category'])
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_categories.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'title': 'Distribution of marks by categories / 2019',
                   'x_label': 'Categories', 'y_label': 'Marks'}
    boxplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 2))

    # Show histogram of adjudications by categories
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_categories_hist.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'estimator': np.count_nonzero,
                   'title': 'Histogram of adjudications by categories / 2019',
                   'x_label': 'Categories', 'y_label': 'Towns'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 1))

    # Show histogram of median marks by categories
    output_file_name = 'output/cleaner_marks_df_2014_total_2019_categories_hist_median.png'
    output_plot = {'df_x': 'category', 'df_y': 'mark',
                   'estimator': np.median, 'ylim': (270, 350),
                   'title': 'Histogram of median marks by categories / 2019',
                   'x_label': 'Category', 'y_label': 'Median Mark'}
    barplot_df_to_png(output_df, output_file_name, output_plot, (3.3, 1))

    # Show heatmap of adjudications by weeks / week days
    output_df = output_df.sort_values(by=['date'])
    output_df = output_df.pivot_table(index=['date_week'], columns=['date_dnn'], values='mark',
                                            aggfunc=np.count_nonzero, fill_value=np.nan)
    output_df = output_df.rename(columns=wd_names)
    print(print_pretty_table(output_df.iloc[:, 0:9].head()))
    output_file_name = 'output/cleaner_marks_df_2014_criteria_2019_heatmap_weeks.png'
    output_plot = {'title': 'Heatmap of adjudications by weeks / week days / 2019',
                   'x_label': 'Week Day', 'y_label': 'Week of Year'}
    heatmap_df_to_png(output_df, output_file_name, output_plot, (6.7, 3))

    # Report 3: Explore criteria rows for 2019

    # Transform the dataset for exploration
    find_criteria_total = df['criteria_tidy'] == 'TOTAL MARK'
    find_year_2019 = df['year'] == 2019
    choose_rows = ~find_criteria_total & find_year_2019
    output_df = df[choose_rows].pivot_table(index=['criteria_tidy'], columns=['category'], values='mark',
                                            aggfunc=np.mean, fill_value=0)

    # View the first 5 rows after transformation
    print('\nReport 3: View the first 5 rows after transformation\n')
    print(print_pretty_table(output_df.head()))

    # Show heatmap of mean marks by criteria / categories
    output_file_name = 'output/cleaner_marks_df_2014_criteria_2019_heatmap_criteria.png'
    output_plot = {'title': 'Heatmap of mean marks by criteria / categories / 2019',
                   'x_label': 'Category', 'y_label': 'Criteria'}
    heatmap_df_to_png(output_df, output_file_name, output_plot, (6.7, 3))

    # Show distribution of pdf_success per county by years
    # choose_cols = ['year', 'county', 'pdf_found', 'pdf_success']
    # groupby_cols = ['year', 'county']
    # output_df = df[choose_cols].groupby(by=groupby_cols).sum().reset_index()
    #
    # output_file_name = 'output/crawler_pdfs_df_years_distr.png'
    # output_plot = {'df_x': 'year', 'df_y': 'pdf_success',
    #                'title': 'Distribution of downloaded pdfs per county by years',
    #                'x_label': 'Year', 'y_label': 'Downloaded pdfs per county'}
    # boxplot_df_to_png(output_df, output_file_name, output_plot, (6.7, 3))

    # # Explore parser output: aggregate to show duplicates
    # choose_cols = ['category', 'county', 'date', 'town', 'year']
    # parser_marks_df_agg = parser_marks_df.groupby(choose_cols)['pdf_name'].nunique().reset_index()
    #
    # print('\nparser_marks_df: aggregate to show duplicates\n')
    # print(print_pretty_table(show_extended_info(parser_marks_df_agg)))
    #
    # # Explore parser output: pivot by county vs year
    # parser_marks_df_pivot = \
    #     parser_marks_df_agg.pivot_table(index=['county'], columns=['year'], values='pdf_name',
    #                                     aggfunc=np.sum, fill_value=0, margins=True, margins_name='(total)')
    #
    # for year in year_range:
    #     if year not in parser_marks_df_pivot:
    #         parser_marks_df_pivot[year] = 0
    #
    # parser_marks_df_pivot.rename(columns={'(total)': -1}, inplace=True)
    # parser_marks_df_pivot.sort_index(axis=1, inplace=True)
    # parser_marks_df_pivot.rename(columns={-1: '(total)'}, inplace=True)
    #
    # write_df_to_csv(parser_marks_df_pivot.reset_index(),
    #                 'output/parser_marks_df_pivot_county_year.csv')
