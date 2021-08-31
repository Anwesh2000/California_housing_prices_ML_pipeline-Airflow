import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


HOUSING_PATH = r'data_file'


def load_housing_data():

    print('Loading training data...')
    X= pd.read_csv(os.path.join(HOUSING_PATH, "X_train.csv"))
    Y= pd.read_csv(os.path.join(HOUSING_PATH, "Y_train.csv"))
    return X, Y


def model_tuning_training(X, Y):
    print('model fine-tuning & training...')
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()

    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=2)
    grid_search.fit(X, Y.values.ravel())
    print('Best parameters :'+ str(grid_search.best_params_))
    return grid_search.best_estimator_


def prepare_data(strat_test_set):
    print('preparing test data')
    Y_test = strat_test_set["median_house_value"].copy()
    housing = strat_test_set.drop("median_house_value", axis=1)

    cat_encoderr = OneHotEncoder(sparse=False)
    cat_encoderr.fit_transform(housing[['ocean_proximity']])
    ocean_proximity_categories = cat_encoderr.categories_[0]

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    X_test = full_pipeline.fit_transform(housing)

    return X_test, Y_test


def model_test_save(model):
    strat_test_set = pd.read_csv(os.path.join(HOUSING_PATH, "test_data.csv"))
    X_test, Y_test = prepare_data(strat_test_set)

    predictions = model.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)

    print('RMSE score of the model: ' + str(rmse))
    print('saving model...')
    joblib.dump(model, os.path.join(HOUSING_PATH, "model.pkl"))


def load():
    X, Y = load_housing_data()
    model = model_tuning_training(X, Y)
    model_test_save(model)



