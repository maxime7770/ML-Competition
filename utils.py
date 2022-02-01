import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def load_data(add_knn_mean = True, add_knn_concat = True, n_data_max = 99999999, shuffle = False):
    print("Loading data...")
    train_df = pd.read_csv('train_df.csv')
    idx = np.random.permutation(train_df.index) if shuffle else train_df.index  #Shuffle train dataset

    Y = train_df["change_type"].reindex(idx)[:n_data_max]
    X = train_df.drop("change_type", 1).reindex(idx)[:n_data_max]
    
    X = X[[
        'area', 'length', 'area/length**2',
        'elongation', 'centroid_x',
        'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
        'centroid_dist', 'length/width', 'Dense Urban', 'Industrial', 'None',
        'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
        'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
        'None.1', 'River', 'Snow', 'Sparse Forest']]

    if add_knn_mean:
        X_knn_aug = pd.read_csv('train_df_knn_mean.csv', index_col = 0).reindex(idx)[:n_data_max]
        X = pd.concat([X, X_knn_aug], axis=1)
        
    if add_knn_concat:
        X_knn_aug = pd.read_csv('train_df_knn_concat.csv', index_col = 0).reindex(idx)[:n_data_max]
        X = pd.concat([X, X_knn_aug], axis=1)
    print("X_train and Y_train loaded.")
    return X, Y


def load_data_test(add_knn_mean = True, add_knn_concat = True, n_data_max = 99999999, shuffle = False):
    print("Loading data...")
    test_df = pd.read_csv('test_df.csv')
    idx = np.random.permutation(test_df.index) if shuffle else test_df.index  #Shuffle train dataset

    X = test_df.reindex(idx)[:n_data_max]

    X = X[[
        'area', 'length', 'area/length**2',
        'elongation', 'centroid_x',
        'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
        'centroid_dist', 'length/width', 'Dense Urban', 'Industrial', 'None',
        'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
        'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
        'None.1', 'River', 'Snow', 'Sparse Forest']]

    if add_knn_mean:
        X_knn_aug = pd.read_csv('test_df_knn_mean.csv', index_col = 0).reindex(idx)[:n_data_max]
        X = pd.concat([X, X_knn_aug], axis=1)
                
    if add_knn_concat:
        X_knn_aug = pd.read_csv('test_df_knn_concat.csv', index_col = 0).reindex(idx)[:n_data_max]
        X = pd.concat([X, X_knn_aug], axis=1)
    print("X_test loaded.")
    return X
