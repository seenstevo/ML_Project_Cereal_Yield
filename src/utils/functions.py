import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.impute import KNNImputer



def world_bank_data_load_melt(path, skiprows = 4):
    '''
    Create a df from the world bank datasets
    Uses melt to generate long format with year in one column
    '''
    
    df = pd.read_csv(path, skiprows = skiprows)
    
    indicator_name = df['Indicator Name'][0]
    
    df = df.drop(columns = ['Indicator Code', 'Indicator Name'], axis = 1)
    
    df = pd.melt(df, id_vars= ['Country Name', 'Country Code'], var_name = 'Year', value_name = indicator_name)
    
    return df
    
    
def drop_too_many_missing(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    '''
    Given a cutoff, we will drop columns that are deemed to contain too many missing values
    '''
    missing_df = pd.DataFrame(df.isnull().sum() * 100 / len(df)).sort_values(by = 0, ascending = False)
    cols_to_drop = missing_df[missing_df[0] > cutoff].index
    
    return df.drop(columns = cols_to_drop)

    
def log_transform_data(df: pd.DataFrame, col_ignore: list):
    '''
    Log transform the numeric columns and recombine back with the non-numeric
    '''
    df_ignore = df[col_ignore]
    df_rest = df.drop(columns = col_ignore)
    numeric = df_rest.select_dtypes(include=np.number).apply(np.log1p)
    non_numeric = df_rest.select_dtypes(exclude=np.number)
    
    return pd.concat([numeric, non_numeric, df_ignore], axis=1)


def extract_lat_lon(row):
    '''
    Simply extract the latitude and longitude for each country
    '''
    lat = row['Latitude']
    long = row['Longitude']
    country = row['Country']
    return country, lat, long


def haversine_calculation(row, countries_with_data: list, df: pd.DataFrame):
    '''
    To be applied to each row in a dataframe we will find the nearest country by haversine distance that also contains the information we want
    '''
    country, lat, long = extract_lat_lon(row)
    country_distance = []
    for row in df.itertuples():
        if country != row.Country:
            country_distance.append([row.Country, haversine((lat, long), (row.Latitude, row.Longitude), unit = 'km')])
    country_distance = sorted(country_distance,key=lambda l:l[1])
    for cd in country_distance:
        if cd[0] in countries_with_data:
            return cd[0], cd[1]
        
        
def knn_impute(cols_to_impute: list, k: int, df: pd.DataFrame):
    
    to_impute_df = df.drop(columns = cols_to_impute)
    imputer = KNNImputer(n_neighbors = k)
    knn_imputed = imputer.fit_transform(to_impute_df)
    
    knn_imputed_df = pd.DataFrame(knn_imputed, columns = to_impute_df.columns).reset_index(drop = True)
    
    return pd.concat([df[cols_to_impute].reset_index(drop = True), knn_imputed_df], axis = 1), imputer