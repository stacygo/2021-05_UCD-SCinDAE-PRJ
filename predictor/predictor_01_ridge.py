from functions import write_df_to_csv, print_pretty_table, show_extended_info
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv')

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # Transform the dataset for predicting
    # Model 1: Predict 'TOTAL MARK' using only 'TOTAL MARK' dynamics by years for non-nan rows
    find_total_mark = df['criteria_tidy'] == 'TOTAL MARK'
    choose_cols = ['town_tidy', 'county_l1', 'criteria_tidy']
    df = df[find_total_mark].pivot_table(index=choose_cols, columns=['year'],
                                         values='mark', aggfunc=np.sum, fill_value=np.nan).reset_index()
    df = df.dropna()

    # View the first 5 rows after transformation
    print('\nView the first 5 rows after transformation\n')
    print(print_pretty_table(df.head()))

    y = df.dropna()[2019].values
    X = df.dropna().drop(['town_tidy', 'county_l1', 'criteria_tidy', 2019], axis=1).values

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    steps = [('ridge', Ridge())]
    parameters = {'ridge__alpha': np.linspace(500, 3000, 50)}
    pipeline_cv = GridSearchCV(Pipeline(steps), parameters, cv=5)

    # Fit to the training set, and predict for the test set
    pipeline_cv.fit(X_train, y_train)
    y_pred = pipeline_cv.predict(X_test)

    # Compute and print the tuned parameters and metrics
    r2 = pipeline_cv.best_score_
    rmse = mean_squared_error(y_test, y_pred) ** 1/2
    print('\nCompute and print the tuned parameters and metrics\n')
    print("Tuned Ridge Parameters: {}".format(pipeline_cv.best_params_))
    print("Tuned Ridge R squared: {}".format(r2))
    print("Tuned Ridge RMSE: {}".format(rmse))

    # Show the predictions for train and test sets
    print('\nView the first 50 rows of predictions in dataset\n')
    df['2019_pred'] = pipeline_cv.predict(X)
    print(print_pretty_table(df.head(50)))
