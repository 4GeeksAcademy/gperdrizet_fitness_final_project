'''Helper functions for EDA/cleaning of continuous variables.'''

import pandas as pd

def standard_scaler(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Takes a dataframe and standard scales is, returns the
    scaled data frame.'''

    print("I'm the scaler!")

    return data_df