import numpy as np
import config
import preprocess
import pandas as pd
from catboost import CatBoostRegressor


def main():
    df = pd.read_csv(config.TEST_FILE)
    df_train = pd.read_csv(config.PREPROCESSED_FILE)
    preprocess.fill_vals(df)
    preprocess.availabilty(df)
    preprocess.area_type_encoded(df)
    preprocess.size_cleaned(df)
    preprocess.total_sqft_cleaned(df)
    df = df.drop(["area_type", "availability", "bath", "balcony", "size", "society"], axis=1)
    print(df.head(20))
    print(df.dtypes)
    #df = df.fillna(0)#Removes all NaN values to reduce training errors
    df.to_csv("./Input/preprocessed_test_data.csv", index=False)
    training(df, df_train) 
   

def training(df, df_train):
    X_train = df_train.drop(["price"], axis = 1)
    y_train = df_train["price"]
    X_test = df.drop(["price"], axis = 1)
    y_test = df["price"]
    print(X_test.head())

    categorical_features_indices = np.where(X_test.dtypes !=np.float)[0]
    model  = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', early_stopping_rounds=10)
    model.fit(X_train, y_train, cat_features=categorical_features_indices, plot=True)
    y_test = model.predict(X_test)
    submission = pd.DataFrame()
    submission["price"] = model.predict(X_test)
    submission.to_csv("./Input/Submission.csv", index = False) 
    print(y_test)
    

if __name__ == "__main__":
    main()
