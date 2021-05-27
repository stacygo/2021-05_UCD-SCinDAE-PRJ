from functions import write_df_to_csv, show_extended_info, print_pretty_table, get_feature_importances
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    SEED = 42
    predictor_name = 'predictor_03'

    # Read the clean dataset from a .csv file
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv',
                       parse_dates=['date'],
                       dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # Approach 3: Predict 'TOTAL MARK' (2019) by 'criteria_tidy' (2018, % of max), 'date_dn', 'date_month', 'category'

    # Transform the dataset for predicting

    # Add a column 'date_dn' with a week day name extracted from 'date' and shortened to 2 symbols
    df['date_dn'] = df['date'].dt.day_name().apply(lambda x: x[0:2])

    # Add a column 'date_month' with a month name extracted from 'date' and shortened to 3 symbols
    df['date_month'] = df['date'].dt.month_name().apply(lambda x: x[0:3])

    # Add a column 'mark_rel' calculated as % of 'mark' from 'max_mark'
    df['mark_rel'] = df['mark'] / df['max_mark']

    # Transform the dataset to get a target column
    find_total_mark = df['criteria_tidy'] == 'TOTAL MARK'
    choose_rows = find_total_mark
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy']
    df_y = df[choose_rows].pivot_table(index=choose_cols, columns=['year'],
                                       values='mark', aggfunc=np.sum, fill_value=np.nan)
    df_y = df_y[[2019]].reset_index().dropna()

    # Transform the dataset to get features columns
    find_non_total_mark = df['criteria_tidy'] != 'TOTAL MARK'
    find_year_2018 = df['year'] == 2018
    choose_rows = find_non_total_mark & find_year_2018
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy', 'date_dn', 'date_month', 'category']
    df_Xy = df[choose_rows].pivot_table(index=choose_cols, columns=['criteria_tidy'],
                                        values='mark_rel', aggfunc=np.min, fill_value=np.nan)
    df_Xy = df_Xy.reset_index().dropna()

    # Add dummy variables for 'date_dn', 'date_month', 'category'
    df_Xy = pd.get_dummies(df_Xy, columns=['date_dn'], drop_first=True)
    df_Xy = pd.get_dummies(df_Xy, columns=['date_month'], drop_first=True)
    df_Xy = pd.get_dummies(df_Xy, columns=['category'], drop_first=True)
    print('\nShape with dummies: ' + str(df_Xy.shape))

    # Show dataframe info after transformation
    print('\nShow dataframe info after transformation\n')
    print(print_pretty_table(show_extended_info(df_Xy)))

    # Merge the transformed dataset with the target column
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy']
    df_Xy = df_Xy.merge(df_y, how='inner', on=choose_cols)
    print('\nShape after the merge: ' + str(df_Xy.shape))

    # Choose target 'y' and features 'X'
    y = df_Xy[2019].values
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy', 2019]
    X = df_Xy.drop(choose_cols, axis=1)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # List the models with their parameters
    models = [('linreg', LinearRegression(), {}),
              ('lasso', Lasso(), {'alpha': np.linspace(0.001, 0.1, 50)}),
              ('ridge', Ridge(), {'alpha': np.linspace(0.01, 0.1, 50)}),
              ('elasticnet', ElasticNet(), {'l1_ratio': np.linspace(0.9, 1, 30)}),
              ('knn', KNeighborsRegressor(), {'n_neighbors': [2, 3, 4, 5],
                                              'weights': ['uniform', 'distance'],
                                              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
              ('svr', SVR(), {'C': np.linspace(100, 300, 20),
                              'epsilon': np.linspace(0.01, 1, 10)}),
              ('treereg', DecisionTreeRegressor(), {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                                                    'max_depth': [3, 5, 9, 15, 25, None],
                                                    'min_samples_split': [2, 7, 20, 24],
                                                    'min_samples_leaf': [1, 2, 5, 8]}),
              ('randforest', RandomForestRegressor(), {'criterion': ['mse', 'mae'],
                                                       'n_estimators': [50, 100, 300]}),
              ('bagging', BaggingRegressor(), {'n_estimators': [10, 50, 100, 200],
                                               'max_samples': [0.3, 0.6, 1.0]}),
              ('adaboost', AdaBoostRegressor(), {'n_estimators': [10, 50, 100]}),
              ('gradboost', GradientBoostingRegressor(), {'n_estimators': [10, 50, 100],
                                                          'loss': ['ls', 'lad', 'huber', 'quantile'],
                                                          'max_depth': [3, 5, 15]}),
              ('gradboost_dft', GradientBoostingRegressor(), {})]

    # Compute the models, and save their scores / features / parameters
    model_scores_df = pd.DataFrame()
    model_features_df = pd.DataFrame()
    model_results_df = pd.DataFrame()

    print('\nCompute the models\n')
    for name, model, parameters in models:
        # Create a model, and search for the best performing parameters
        model_cv = GridSearchCV(model, parameters, cv=5,
                                scoring=['r2', 'neg_mean_squared_error'], refit='r2')

        # Fit to the training set, and predict for the test set
        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)

        # Save the tuned parameters and metrics
        model_cv_score_r2 = model_cv.cv_results_['mean_test_r2'][model_cv.best_index_]
        model_cv_score_nmse = model_cv.cv_results_['mean_test_neg_mean_squared_error'][model_cv.best_index_]
        model_cv_score = pd.DataFrame([{'name': name,
                                        'r2_train': model_cv_score_r2,
                                        'r2_test': r2_score(y_test, y_pred),
                                        'rmse_train': (-model_cv_score_nmse) ** (1/2),
                                        'rmse_test': mean_squared_error(y_test, y_pred) ** (1/2),
                                        'best_params_': model_cv.best_params_}])
        model_scores_df = model_scores_df.append(model_cv_score)

        # Save found regression coefficients / features importances
        model_cv_features = pd.DataFrame()
        model_cv_features[name] = get_feature_importances(model_cv.best_estimator_)
        model_features_df = pd.concat([model_features_df, model_cv_features], axis=1)

        # Save CV results
        model_cv_results = pd.DataFrame([model_cv.cv_results_])
        model_cv_results['name'] = name
        model_results_df = model_results_df.append(model_cv_results)

        # Make the predictions for 'X', and add them to 'df'
        df_Xy['pred_' + name] = model_cv.predict(X).round(2)
        print('> Done: ' + name)

    # Print the tuned parameters and metrics
    print('\nPrint the tuned parameters and metrics\n')
    print(print_pretty_table(model_scores_df, '.4f'))

    # Transform the dataframe with found regression coefficients / features importances
    model_features_df = model_features_df.T
    model_features_df.columns = X.columns.values
    model_features_df = model_features_df.reset_index()

    # Write the results to .csv files
    write_df_to_csv(model_scores_df, 'output/' + predictor_name + '_model_scores.csv')
    write_df_to_csv(model_features_df, 'output/' + predictor_name + '_model_features.csv')
    write_df_to_csv(model_results_df, 'output/' + predictor_name + '_model_results.csv')
    write_df_to_csv(df_Xy.sort_values(by=2019, ascending=False), 'output/' + predictor_name + '.csv')
