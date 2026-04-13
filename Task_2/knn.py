from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np           

def impute_knn_data(df, n_neighbors=5):
    
    df_knn = df.copy()
    
    # Store original unique values (excluding NaNs) to map them later
    # This prevents creating non-existent categories like '4' in thal
    valid_ca = sorted([v for v in df['ca'].unique() if not np.isnan(v)])
    valid_thal = sorted([v for v in df['thal'].unique() if not np.isnan(v)])
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Impute missing values
    df_filled = pd.DataFrame(imputer.fit_transform(df_knn), columns=df_knn.columns)
    
    def match_closest(value, valid_list):
        # Finds the element in valid_list that is closest to the imputed 'value'
        return min(valid_list, key=lambda x: abs(x - value))

    # pick the closest REAL category from the medical list
    df_filled['ca'] = df_filled['ca'].apply(lambda x: match_closest(x, valid_ca))
    df_filled['thal'] = df_filled['thal'].apply(lambda x: match_closest(x, valid_thal))
    
    return df_filled

def show_imputed_values(df_with_nan, df_after_knn):
    print("\n" + "="*40)
    print("VERIFYING IMPUTED VALUES (KNN)")
    print("="*40)
    # Identify the exact row indices that contained at least one NaN before imputation
    nan_indices = df_with_nan[df_with_nan.isnull().any(axis=1)].index
  
    cols_to_show = ['slope', 'ca', 'thal', 'target']
    
    print("Values inserted by KNN in the missing places:")
    print(df_after_knn.loc[nan_indices, cols_to_show])