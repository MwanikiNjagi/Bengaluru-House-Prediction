import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config
import numpy as np
from random import choice
def main():
    df = pd.read_csv(config.TRAINING_FILE)
    df_test =  pd.read_csv(config.TEST_FILE)
    fill_vals(df)
    #society_cleaned(df)
    availabilty(df)
    area_type_encoded(df)
    size_cleaned(df)
    total_sqft_cleaned(df)
    df = df.drop(["area_type", "availability", "bath", "balcony", "size", "society"], axis=1)
    df = df.dropna()#Removes all NaN values to reduce training errors
    df.to_csv("./Input/preprocessed_train_data.csv", index=False) #Removes indices
    print(df.head(30))
    print(df.dtypes)
    print(df_test.dtypes)
    return df
    #print("Data has been preprocessed and converted to a csv file")
    #print (df["society"].value_counts())


#Fills NaN values in balcony and bath and drops all rows with NaN values in the society column
def fill_vals(df, column_1 = "balcony", column_2 = "bath"):
    df[column_1].fillna(method='ffill', inplace=True)
    df[column_2].fillna(method='ffill', inplace=True)
    df[column_1] = df[column_1].astype(int)
    df[column_2] = df[column_2].astype(int)
    print(df.head())
    df["bath_processed"] =  df[column_2]
    df["balcony_processed"] = df[column_1]
    #df[column_3] = df[column_3].fillna(df.groupby("location")[column_3].apply(lambda x:x.mode()[0])) 
    #df = df.fillna(df.groupby("location")[column_4].mode()[0], inplace = True)
    return df
#    df[column] = df[column].fillna(df.groupby(["location"])[column].apply(lambda x:x.fillna(x.mode())))
   # return df

#Processing availability into three labels 0, 1, 2
def availabilty(df, column = "availability"):
    #Using Ready To Move as 0 and the rest as 1
    df[column] = df[column].apply(lambda x:str(x).replace("Ready To Move", "0"))
    df[column] = df[column].apply(lambda x:str(x).replace("Immediate Possession", "1"))
    df[column] = df[column].apply(lambda x:x.split("-")[0])
    df[column] = df[column].astype(int)
    df.loc[df[column]>0, column] = 2
    df["availability_encoded"] =  df[column]
    return df


def area_type_encoded(df, column = "area_type"):  
    LE = LabelEncoder()
    df["area_type_encoded"] = LE.fit_transform(df[column])
    return df   
#BHK means Bedroom, Hall and Kitchen and the number preceding BHK means the number of bedrooms
def size_cleaned(df, column="size"):
    df[column] = df[column].apply(lambda x:str(x).split(' ')[0])
    df[column] = df[column].apply(lambda x:str(x).replace("GrrvaGr", "None"))
    df[column] = pd.to_numeric(df[column], errors = "coerce")
    df[column] = df[column].fillna(df[column].mean()) 
    #df[column] = df[column].apply(lambda x:x.replace("None", "0"))
    df[column] = df[column].astype(int)
    df["size_cleaned"] = df[column]
  

def total_sqft_cleaned(df, column = "total_sqft"):
    df[column] = df[column].apply(lambda x:x.split('-')[0])
    df[column] = pd.to_numeric(df[column], errors = "coerce")#Drops Sq.Meters and Sq.Yards
    df[column] = df[column].fillna(df[column].mean())
    df[column] =  df[column].astype(float)
    

if __name__ == "__main__":
    main()
   