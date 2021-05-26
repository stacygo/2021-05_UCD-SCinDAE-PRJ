from functions import write_df_to_csv, print_pretty_table, get_feature_importances
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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
    predictor_name = 'predictor_03_2'
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv')

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # Model 3: Predict 'TOTAL MARK' (2019) by 'criteria_tidy' (2018, % of max), 'category_tidy'

    # Calculate 'mark_rel' as % of 'mark' from 'max_mark'
    df['mark_rel'] = df['mark'] / df['max_mark']

    # Transform the dataset to get a target column
    find_total_mark = df['criteria_tidy'] == 'TOTAL MARK'
    choose_rows = find_total_mark
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy']
    df_y = df[choose_rows].pivot_table(index=choose_cols, columns=['year'],
                                       values='mark_rel', aggfunc=np.min, fill_value=np.nan)
    df_y = df_y[[2019]].reset_index().dropna()

    # View the first 5 rows after transformation
    print('\nView the first 5 rows after transformation\n')
    print(print_pretty_table(df_y.head()))
    print(df_y.shape)

    # Add a column 'category_code' to shorten categories name
    # df['criteria_tidy'] = pd.Categorical(df['criteria_tidy'])
    # df['criteria_code'] = df['criteria_tidy'].cat.codes
    # df['criteria_code'] = df['criteria_code'].apply(lambda x: 'cri_' + str(x))

    # Transform the dataset for predicting
    find_non_total_mark = df['criteria_tidy'] != 'TOTAL MARK'
    find_year_2018 = df['year'] == 2018
    choose_rows = find_non_total_mark & find_year_2018
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy']
    df = df[choose_rows].pivot_table(index=choose_cols, columns=['criteria_tidy'],
                                     values='mark_rel', aggfunc=np.min, fill_value=np.nan)
    df = df.reset_index().dropna()

    # View the first 5 rows after transformation
    print('\nView the first 5 rows after transformation\n')
    print(print_pretty_table(df.iloc[:, 0:5].head(), '.4f'))
    print(df.shape)

    # Add a dummy variable for 'category_tidy'
    df['category_dummy'] = df['category_tidy']
    df = pd.get_dummies(df, columns=['category_dummy'], drop_first=True)
    print('\nShape with dummies: ' + str(df.shape))

    # Merge the transformed dataset with the target column
    df = df.merge(df_y, how='inner', on=choose_cols)
    print('\nShape after the merge: ' + str(df.shape))

    # Choose target 'y' and features 'X'
    y = df[2019].values
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy', 2019]
    X = df.drop(choose_cols, axis=1)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # List the models with their parameters
    models = [('linreg', LinearRegression(), {}),
              ('lasso', Lasso(), {'alpha': np.linspace(0.0001, 1, 50)}),
              ('ridge', Ridge(), {'alpha': np.linspace(0.0001, 1, 50)}),
              ('elasticnet', ElasticNet(), {'l1_ratio': np.linspace(0.0001, 1, 30)}),
              ('treereg', DecisionTreeRegressor(), {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                                                    'max_depth': [3, 5, 9, 15, 25, None],
                                                    'min_samples_split': [2, 5, 20, 24],
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
        df['pred_' + name] = model_cv.predict(X).round(2)
        print('> Done: ' + name)

    # Print the results
    print('\nPrint the tuned parameters and metrics\n')
    print(print_pretty_table(model_scores_df, '.4f'))

    print('\nPrint the features importances\n')
    model_features_df = model_features_df.T
    model_features_df.columns = X.columns.values
    print(print_pretty_table(model_features_df.iloc[:, 0:3], '.4f'))
    print(model_features_df.shape)

    write_df_to_csv(model_scores_df, 'output/' + predictor_name + '_model_scores.csv')
    write_df_to_csv(model_features_df, 'output/' + predictor_name + '_model_features.csv')
    write_df_to_csv(model_results_df, 'output/' + predictor_name + '_model_results.csv')
    write_df_to_csv(df.sort_values(by=2019, ascending=False), 'output/' + predictor_name + '.csv')
