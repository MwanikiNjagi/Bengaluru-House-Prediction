import config
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    df = pd.read_csv(config.PREPROCESSED_FILE)
    training(df)
    print(df.head(20))


def training(df):
    X = df.drop(["price"], axis=1)
    y = df["price"]
    print(X.head())
    categorical_features_indices = np.where(X.dtypes !=np.float)[0]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size = 0.8, random_state = 12)
    model  = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', early_stopping_rounds=5)
    model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_validation, y_validation), plot=True)
    y_valid = model.predict(X_validation)
    print (y_valid)


if __name__== "__main__":
    main()