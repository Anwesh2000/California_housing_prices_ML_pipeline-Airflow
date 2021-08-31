import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_housing_data():
    print('Loading Housing data...')
    housing_path = r'data_file'
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def create_train_test_set(housing):
    print('Creating training & testing sets...')
    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set, strat_test_set


def store_data(df_housing_prepared, housing_labels):
    print('storing prepared training data...')
    df_housing_prepared.to_csv(r'data_file/X_train.csv', header=True, index=False)
    housing_labels.to_csv(r'data_file/Y_train.csv', header=True, index=False)


def transform():
    housing = load_housing_data() 
    strat_train_set, strat_test_set = create_train_test_set(housing)

    strat_test_set.to_csv(r'data_file/test_data.csv', header=True, index=False)
    print('preparing data for training...')
    housing_labels = strat_train_set["median_house_value"].copy()
    housing = strat_train_set.drop("median_house_value", axis=1)

    cat_encoderr = OneHotEncoder(sparse=False)
    cat_encoderr.fit_transform(housing[['ocean_proximity']])
    ocean_proximity_categories =cat_encoderr.categories_[0]

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
    
    housing_prepared=full_pipeline.fit_transform(housing)
    df_housing_prepared = pd.DataFrame(housing_prepared, columns=list(housing_num)+ list(ocean_proximity_categories))

    store_data(df_housing_prepared, housing_labels)


