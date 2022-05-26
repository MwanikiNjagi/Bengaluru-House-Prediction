import config
import pandas as pd

def main():
    df = pd.read_csv(config.PREPROCESSED_FILE)
    print(df.head())

if __name__== "__main__":
    main()