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
    predictor_name = 'predictor_01'
    df = pd.read_csv('../cleaner/output/cleaner_marks_df_2014.csv')

    # View the first 5 rows
    print('\nView the first 5 rows\n')
    print(print_pretty_table(df.head()))

    # Transform the dataset for predicting
    # Model 1: Predict 'TOTAL MARK' using only 'TOTAL MARK' dynamics by years for non-nan rows
    find_total_mark = df['criteria_tidy'] == 'TOTAL MARK'
    choose_rows = find_total_mark
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy', 'criteria_tidy']
    df = df[choose_rows].pivot_table(index=choose_cols, columns=['year'],
                                     values='mark', aggfunc=np.sum, fill_value=np.nan).reset_index()
    print(df.shape)
    df = df.dropna()
    print(df.shape)

    # View the first 5 rows after transformation
    print('\nView the first 5 rows after transformation\n')
    print(print_pretty_table(df.head()))

    y = df[2019].values
    choose_cols = ['category_tidy', 'county_l1', 'town_tidy', 'criteria_tidy', 2019]
    X = df.drop(choose_cols, axis=1)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # List the models with their parameters
    models = [('linreg', LinearRegression(), {}),
              ('lasso', Lasso(), {'alpha': np.linspace(0.1, 10, 50)}),
              ('ridge', Ridge(), {'alpha': np.linspace(100, 3000, 50)}),
              ('elasticnet', ElasticNet(), {'l1_ratio': np.linspace(0.01, 1, 30)}),
              ('treereg', DecisionTreeRegressor(), {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                                                    'max_depth': [3, 5, 9, 15, 25, None],
                                                    'min_samples_split': [2, 5, 20, 24],
                                                    'min_samples_leaf': [1, 2, 5, 8]}),
              ('randforest', RandomForestRegressor(), {'criterion': ['mse', 'mae'],
                                                       'n_estimators': [100, 300, 500]}),
              ('bagging', BaggingRegressor(), {'n_estimators': [10, 50, 100, 200],
                                               'max_samples': [0.3, 0.6, 1.0]}),
              ('adaboost', AdaBoostRegressor(), {'n_estimators': [10, 50, 100, 200]}),
              ('gradboost', GradientBoostingRegressor(), {'n_estimators': [10, 50, 100, 200],
                                                          'loss': ['ls', 'lad', 'huber', 'quantile'],
                                                          'max_depth': [3, 5, 15]}),
              ('gradboost_dft', GradientBoostingRegressor(), {})]

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

        model_cv_features = pd.DataFrame()
        model_cv_features[name] = get_feature_importances(model_cv.best_estimator_)
        model_features_df = pd.concat([model_features_df, model_cv_features], axis=1)

        model_cv_results = pd.DataFrame([model_cv.cv_results_])
        model_cv_results['name'] = name
        model_results_df = model_results_df.append(model_cv_results)

        # Make the predictions for train and test sets, and save them
        df['pred_' + name] = model_cv.predict(X).round(2)
        print('> Done: ' + name)

    print('\nPrint the tuned parameters and metrics\n')
    print(print_pretty_table(model_scores_df, '.4f'))

    print('\nPrint the features importances\n')
    model_features_df = model_features_df.T
    model_features_df.columns = X.columns.values
    print(print_pretty_table(model_features_df, '.4f'))

    write_df_to_csv(model_scores_df, 'output/' + predictor_name + '_model_scores.csv')
    write_df_to_csv(model_features_df, 'output/' + predictor_name + '_model_features.csv')
    write_df_to_csv(model_results_df, 'output/' + predictor_name + '_model_results.csv')
    write_df_to_csv(df, 'output/' + predictor_name + '.csv')
