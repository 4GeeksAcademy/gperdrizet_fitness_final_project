'''Helper functions for EDA/cleaning of categorical variables.'''

import pandas as pd

def encode_variables(data_df:pd.DataFrame) -> pd.DataFrame:
    '''Takes dataframe, one-hot encodes categorical variables,
    returns encoded dataframe'''

    return data_df