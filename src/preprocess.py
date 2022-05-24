from pydoc import describe
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config

def main():
    df = pd.read_csv(config.TRAINING_FILE)
    fill_vals(df)
    availabilty(df)
    #society_encoded(df)
    area_type_encoded(df)
    size_cleaned(df)
    df.to_csv("./Input/preprocessed_data.csv")
    df = df.drop(["area_type", "availability", "bath", "balcony", "size"], axis=1)
    print(df.head(20))
    print(df["location"].describe())
    print(df["society"].describe())
    #print("Data has been preprocessed and converted to a csv file")
    #print (df["society"].value_counts())


#Fills NaN values in balcony and bath and drops all rows with NaN values in the society column
def fill_vals(df, column_1 = "balcony", column_2 = "bath"):
    df[column_1].fillna(method='ffill', inplace=True)
    df[column_2].fillna(method='ffill', inplace=True)
    df[column_1] = df[column_1].astype(int)
    df[column_2] = df[column_2].astype(int)
    df["bath_processed"] =  df[column_2]
    df["balcony_processed"] = df[column_1]
    df.dropna(subset=["society", "size"], inplace=True)
    return df

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

#def society_encoded(df, column="society"):

def area_type_encoded(df, column = "area_type"):  
    LE = LabelEncoder()
    df["area_type_encoded"] = LE.fit_transform(df[column])
    return df   
#BHK means Bedroom, Hall and Kitchen and the number preceding BHK means the number of bedrooms
def size_cleaned(df, column="size"):
    df[column] = df[column].apply(lambda x:str(x).split(' ')[0])
    df[column] = df[column].astype(int)
    df["size_cleaned"] = df[column]

#def 
if __name__ == "__main__":
    main()
   