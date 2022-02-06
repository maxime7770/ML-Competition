import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def plot_cluster(df_cluster, save=None):
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
    colors = ['r', 'b', 'g', 'y', 'c', 'w']

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
        plt.scatter(L_x, L_y, s=0.1, color=colors[i])
        if stopping:
            break

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def save_list(L, name):
    A = np.array(L)
    np.save(name, A)


def load_list(name):
    A = np.load(name)
    return A.tolist()


<<<<<<< HEAD
def load_data(add_knn_mean=False,
              add_knn_concat=False,
              add_polynomial=False,
              add_dates=False,
              add_fourier=False,
              add_capitals=False,
              add_countries=False,
              n_data_max=99999999, shuffle=False):
    print("Loading data...")
    name = 'train'
    train_df = pd.read_csv(f'{name}_df.csv')
    idx = np.random.permutation(
        train_df.index) if shuffle else train_df.index  # Shuffle train dataset

    Y = train_df["change_type"]
    X = train_df.drop("change_type", 1)

    X = X[[
        "1_Construction Done","1_Construction Midway","1_Construction Started","1_Excavation","1_Greenland","1_Land Cleared","1_Materials Dumped","1_NA","1_Operational","1_Prior Construction","2_Construction Done","2_Construction Midway","2_Construction Started","2_Excavation","2_Greenland","2_Land Cleared","2_Materials Dumped","2_NA","2_Operational","2_Prior Construction","3_Construction Done","3_Construction Midway","3_Construction Started","3_Excavation","3_Greenland","3_Land Cleared","3_Materials Dumped","3_NA","3_Operational","3_Prior Construction","4_Construction Done","4_Construction Midway","4_Construction Started","4_Excavation","4_Greenland","4_Land Cleared","4_Materials Dumped","4_NA","4_Operational","4_Prior Construction","5_Construction Done","5_Construction Midway","5_Construction Started","5_Excavation","5_Greenland","5_Land Cleared","5_Materials Dumped","5_NA","5_Operational","5_Prior Construction",
        'diff1', 'diff2', 'diff3', 'diff4',
        #    'season_date1', 'season_date2', 'season_date3', 'season_date4', 'season_date5',
        'year_date1', 'year_date2', 'year_date3', 'year_date4', 'year_date5',

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
        X_aug = pd.read_csv(f'{name}_df_knn_mean.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print(f'knn mean features lenght: {len(X_aug)}')

    if add_knn_concat:
        X_aug = pd.read_csv(f'{name}_df_knn_concat.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print(f'knn concat features lenght: {len(X_aug)}')

    if add_dates:
        X_aug = pd.read_csv(f'{name}_df_dates.csv', index_col=0)
        X_aug = X_aug[['duration_to_reach1', 'duration_to_reach2', 'duration_to_reach3', 'duration_to_reach4', 'duration_to_reach5',
                       'old1', 'old2', 'old3', 'old4', 'old5', ]]
        if len(X_aug) != len(X):
            raise
        print(f'Dates features lenght: {len(X_aug)}')
        X = pd.concat([X, X_aug], axis=1)

    if add_polynomial:
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)
        X_aug = poly.fit_transform(X[['knn_mean_length/width',
                                      'area/length**2',
                                      'centroid_dist',
                                      'knn_mean_area',
                                      'duration_to_reach5', 'nb_points',
                                      'height', 'length', 'elongation', 'length/width',
                                      ]])
        # columns=[f'poly_aug_{i}' for i in range(X_aug.shape[0])])
        X_aug = pd.DataFrame(X_aug)
        X = pd.concat([X, X_aug], axis=1)

    if add_fourier:
        print('Building Fourier coefficients...')
        X_aug = pd.read_csv(f'fourier_coefficients_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    if add_capitals:
        print('Distances to nearest capitals...')
        X_aug = pd.read_csv(f'dist_to_capitals_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    if add_countries:
        print('Adding countries...')
        X_aug = pd.read_csv(f'countries_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    print("X_train and Y_train loaded.")
    return X.iloc[idx][: n_data_max], Y.iloc[idx][: n_data_max]


def load_data_test(add_knn_mean=False,
                   add_knn_concat=False,
                   add_polynomial=False,
                   add_dates=False,
                   add_fourier=False,
                   add_capitals=False,
                   add_countries=False,
                   n_data_max=99999999, shuffle=False):
    print("Loading data...")
    name = 'test'
    train_df = pd.read_csv(f'{name}_df.csv')
    idx = np.random.permutation(
        train_df.index) if shuffle else train_df.index  # Shuffle train dataset

    X = train_df

    X = X[[
             "1_Construction Done","1_Construction Midway","1_Construction Started","1_Excavation","1_Greenland","1_Land Cleared","1_Materials Dumped","1_NA","1_Operational","1_Prior Construction","2_Construction Done","2_Construction Midway","2_Construction Started","2_Excavation","2_Greenland","2_Land Cleared","2_Materials Dumped","2_NA","2_Operational","2_Prior Construction","3_Construction Done","3_Construction Midway","3_Construction Started","3_Excavation","3_Greenland","3_Land Cleared","3_Materials Dumped","3_NA","3_Operational","3_Prior Construction","4_Construction Done","4_Construction Midway","4_Construction Started","4_Excavation","4_Greenland","4_Land Cleared","4_Materials Dumped","4_NA","4_Operational","4_Prior Construction","5_Construction Done","5_Construction Midway","5_Construction Started","5_Excavation","5_Greenland","5_Land Cleared","5_Materials Dumped","5_NA","5_Operational","5_Prior Construction",
        'diff1', 'diff2', 'diff3', 'diff4',
        #    'season_date1', 'season_date2', 'season_date3', 'season_date4', 'season_date5',
        'year_date1', 'year_date2', 'year_date3', 'year_date4', 'year_date5',

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
        X_aug = pd.read_csv(f'{name}_df_knn_mean.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print(f'knn mean features lenght: {len(X_aug)}')

    if add_knn_concat:
        X_aug = pd.read_csv(f'{name}_df_knn_concat.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print(f'knn concat features lenght: {len(X_aug)}')

    if add_dates:
        X_aug = pd.read_csv(f'{name}_df_dates.csv', index_col=0)
        X_aug = X_aug[['duration_to_reach1', 'duration_to_reach2', 'duration_to_reach3', 'duration_to_reach4', 'duration_to_reach5',
                       'old1', 'old2', 'old3', 'old4', 'old5', ]]
        if len(X_aug) != len(X):
            raise
        print(f'Dates features lenght: {len(X_aug)}')
        X = pd.concat([X, X_aug], axis=1)

    if add_polynomial:
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)
        X_aug = poly.fit_transform(X[['knn_mean_length/width',
                                      'area/length**2',
                                      'centroid_dist',
                                      'knn_mean_area',
                                      'duration_to_reach5', 'nb_points',
                                      'height', 'length', 'elongation', 'length/width',
                                      ]])
        # columns=[f'poly_aug_{i}' for i in range(X_aug.shape[0])])
        X_aug = pd.DataFrame(X_aug)
        X = pd.concat([X, X_aug], axis=1)

    if add_fourier:
        print('Building Fourier coefficients...')
        X_aug = pd.read_csv(f'fourier_coefficients_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    if add_capitals:
        print('Distances to nearest capitals...')
        X_aug = pd.read_csv(f'dist_to_capitals_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    if add_countries:
        print('Adding countries...')
        X_aug = pd.read_csv(f'countries_{name}.csv', index_col=0)
        X = pd.concat([X, X_aug], axis=1)
        print('Done')

    print("X_val loaded.")
    return X.iloc[idx][: n_data_max]
