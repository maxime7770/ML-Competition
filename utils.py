import matplotlib.pyplot as plt
import numpy as np

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
        X_feature = df_cluster[df_cluster['change_type'] == category]
        geometry = X_feature['geometry']
        L_x = geometry.apply(lambda x : x.boundary.centroid.x)
        L_y = geometry.apply(lambda x : x.boundary.centroid.y)
        plt.scatter(L_x, L_y, s = 0.1, color = colors[i])
    
    if save is not None: plt.savefig(save)
    else: plt.show()
    


def save_list(L, name):
    A = np.array(L)
    np.save(name, A)
def load_list(name):
    A = np.load(name)
    return A.tolist()