import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def plot_cluster(df_cluster, save = None):
    '''Plot geographical positions of a dataframe of points supposed to be a cluster.
    df_cluster : a df containing geometry
    '''
    change_type_map = {
        "Demolition": 0,
        "Road": 1,
        "Residential": 2,
        "Commercial": 3,
        "Industrial": 4,
        "Mega Projects": 5,
    }
    colors = ['r','b','g','y','c','w']

    # plt.style.use('dark_background')
    for category, i in change_type_map.items():
        if 'change_type' in df_cluster:
            X_feature = df_cluster[df_cluster['change_type'] == category]
            stopping = False
        else:
            X_feature = df_cluster
            stopping = True
        L_x = X_feature['centroid_x']
        L_y = X_feature['centroid_y']
        plt.scatter(L_x, L_y, s = 0.1, color = colors[i])
        if stopping: break
    
    if save is not None: plt.savefig(save)
    else: plt.show()
    


def save_list(L, name):
    A = np.array(L)
    np.save(name, A)
def load_list(name):
    A = np.load(name)
    return A.tolist()


def load_data(add_knn_mean = False, 
              add_knn_concat = False, 
              add_polynomial = False, 
              add_dates = False, 
              n_data_max = 99999999, shuffle = False):
    print("Loading data...")
    name = 'train'
    train_df = pd.read_csv(f'{name}_df.csv')
    idx = np.random.permutation(train_df.index) if shuffle else train_df.index  #Shuffle train dataset

    Y = train_df["change_type"]
    X = train_df.drop("change_type", 1)
    
    X = X[[
       'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4', 'change_status_date5',
       'diff1', 'diff2', 'diff3', 'diff4', 
       'season_date1', 'season_date2', 'season_date3', 'season_date4', 'season_date5',
       'year_date1', 'year_date2', 'year_date3', 'year_date4', 'year_date5',
       
        'area', 'length', 'area/length**2',
        'elongation', 'centroid_x',
        'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
        'centroid_dist', 'length/width', 
        
        'Dense Urban', 'Industrial', 'None',
        'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
        'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
        'None.1', 'River', 'Snow', 'Sparse Forest']]

    if add_knn_mean:
        X_knn_aug = pd.read_csv(f'{name}_df_knn_mean.csv')
        X = pd.concat([X, X_knn_aug], axis=1)
        
    if add_knn_concat:
        X_knn_aug = pd.read_csv(f'{name}_df_knn_concat.csv', index_col = 0)
        X = pd.concat([X, X_knn_aug], axis=1)
    
    if add_dates:
        X_knn_aug = pd.read_csv(f'{name}_df_dates.csv', index_col = 0)
        X_knn_aug = X_knn_aug[['duration_to_reach1','duration_to_reach2','duration_to_reach3','duration_to_reach4','duration_to_reach5']]
        if len(X_knn_aug) != len(X):
            raise
        X = pd.concat([X, X_knn_aug], axis=1)
        
    if add_polynomial:
        print('Poly augment...')
        poly = PolynomialFeatures(degree =2, interaction_only = True, include_bias=False)
        X = poly.fit_transform(X)
        print('Done')

    print("X_train and Y_train loaded.")
    return X.iloc[idx][: n_data_max], Y.iloc[idx][: n_data_max]







def load_data_test(add_knn_mean = False, 
              add_knn_concat = False, 
              add_polynomial = False, 
              add_dates = False, 
              n_data_max = 99999999, shuffle = False):
    print("Loading data...")
    name = 'test'
    train_df = pd.read_csv(f'{name}_df.csv')
    idx = np.random.permutation(train_df.index) if shuffle else train_df.index  #Shuffle train dataset

    X = train_df
    
    X = X[[
       'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4', 'change_status_date5',
       'diff1', 'diff2', 'diff3', 'diff4', 
       'season_date1', 'season_date2', 'season_date3', 'season_date4', 'season_date5',
       'year_date1', 'year_date2', 'year_date3', 'year_date4', 'year_date5',
       
        'area', 'length', 'area/length**2',
        'elongation', 'centroid_x',
        'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
        'centroid_dist', 'length/width', 
        
        'Dense Urban', 'Industrial', 'None',
        'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
        'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
        'None.1', 'River', 'Snow', 'Sparse Forest']]

    if add_knn_mean:
        X_knn_aug = pd.read_csv(f'{name}_df_knn_mean.csv')
        X = pd.concat([X, X_knn_aug], axis=1)
        
    if add_knn_concat:
        X_knn_aug = pd.read_csv(f'{name}_df_knn_concat.csv', index_col = 0)
        X = pd.concat([X, X_knn_aug], axis=1)
    
    if add_dates:
        X_knn_aug = pd.read_csv(f'{name}_df_dates.csv', index_col = 0)
        X_knn_aug = X_knn_aug[['duration_to_reach1','duration_to_reach2','duration_to_reach3','duration_to_reach4','duration_to_reach5']]
        if len(X_knn_aug) != len(X):
            raise
        X = pd.concat([X, X_knn_aug], axis=1)
        
    if add_polynomial:
        print('Poly augment...')
        poly = PolynomialFeatures(degree =2, interaction_only = True, include_bias=False)
        X = poly.fit_transform(X)
        print('Done')

    print("X_val loaded.")
    return X[: n_data_max][idx]