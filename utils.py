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
              mean_weighted = True,
              add_knn_concat = False, 
              add_polynomial = False, 
              add_dates = False, 
              n_data_max = 99999999, 
              shuffle = False,
              test = False):
    print("Loading data...")
    name = 'train' if not test else 'test'
    X = pd.read_csv(f'{name}_df.csv')
    idx = np.random.permutation(X.index) if shuffle else X.index  #Shuffle train dataset

    if not test:
        Y = X["change_type"]
        X = X.drop("change_type", 1)
    
    
    X = X[[
       'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4', 'change_status_date5',
       'diff1', 'diff2', 'diff3', 'diff4', 
    #    'season_date1', 'season_date2', 'season_date3', 'season_date4', 'season_date5',
    #    'year_date1', 'year_date2', 'year_date3', 'year_date4', 'year_date5',
       
        'area', 'length', 'area/length**2',
        'elongation', 'centroid_x',
        'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
        'centroid_dist', 'length/width', 
        
        'Dense Urban', 'Industrial', 'None',
        'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
        'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
        'None.1', 'River', 'Snow', 'Sparse Forest']]
    print(f'\nBasic features lenght: {len(X)}')

    if add_knn_mean:
        if mean_weighted:
            X_aug = pd.read_csv(f'{name}_df_knn_weighted_mean.csv', index_col = 0)
            X = pd.concat([X, X_aug], axis=1)
            print(f'knn mean weighted features lenght: {len(X_aug)}')
        else:
            X_aug = pd.read_csv(f'{name}_df_knn_mean.csv', index_col = 0)
            X = pd.concat([X, X_aug], axis=1)
            print(f'knn mean features lenght: {len(X_aug)}')

        
    if add_knn_concat:
        X_aug = pd.read_csv(f'{name}_df_knn_concat.csv', index_col = 0)
        X = pd.concat([X, X_aug], axis=1)
        print(f'knn concat features lenght: {len(X_aug)}')

    
    if add_dates:
        X_aug = pd.read_csv(f'{name}_df_dates.csv', index_col = 0)
        X_aug = X_aug[['duration_to_reach1','duration_to_reach2','duration_to_reach3','duration_to_reach4','duration_to_reach5',
                               'old1','old2','old3','old4','old5',]]
        if len(X_aug) != len(X):
            raise
        print(f'Dates features lenght: {len(X_aug)}')
        X = pd.concat([X, X_aug], axis=1)
            
        
    if add_polynomial:
        poly = PolynomialFeatures(degree =2, interaction_only = True, include_bias=False)
        X_aug = poly.fit_transform(X[['knn_mean_length/width', 
                                  'area/length**2', 
                                  'centroid_dist',
                                  'knn_mean_area',
                                  'duration_to_reach5', 'nb_points',
                                  'height', 'length', 'elongation', 'length/width',
        ]])
        X_aug = pd.DataFrame(X_aug)# columns=[f'poly_aug_{i}' for i in range(X_aug.shape[0])])
        X = pd.concat([X, X_aug], axis=1)

    print("X_train and Y_train loaded.")
    if not test:
        return X.iloc[idx][: n_data_max], Y.iloc[idx][: n_data_max]
    else: 
        return X.iloc[idx][: n_data_max]



def load_data_test(**kwargs):
    return load_data(**kwargs, test = True)