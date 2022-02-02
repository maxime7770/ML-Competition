from convexity import is_convex_polygon
from math import *
from xgboost import XGBClassifier, cv
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost
from joblib import dump, load
from datetime import datetime
from scipy.stats import boxcox


change_type_map = {
    "Demolition": 0,
    "Road": 1,
    "Residential": 2,
    "Commercial": 3,
    "Industrial": 4,
    "Mega Projects": 5,
}
# to delete last classes

# change_type_map = {
#     "Demolition": 0,
#     "Road": 1,
#     "Residential": 2,
#     "Commercial": 3,
#     "Industrial": 4
# }

train_df = gpd.read_file("train.geojson", index_col=0)
test_df = gpd.read_file("test.geojson", index_col=0)


# trying to delete last classes


# train_df = train_df.drop(
#     train_df[(train_df["change_type"] == "Mega Projects")].index)


# and thus drop if status_sate=excavation because it is essentially in the 5th class

# train_df = train_df.drop(
#     train_df[
#         (train_df["change_status_date1"] == "Excavation")
#         | (train_df["change_status_date2"] == "Excavation")
#         | (train_df["change_status_date3"] == "Excavation")
#         | (train_df["change_status_date4"] == "Excavation")
#         | (train_df["change_status_date5"] == "Excavation")
#     ].index
# )

# test_df = test_df.drop(
#     test_df[
#         (test_df["change_status_date1"] == "Excavation")
#         | (test_df["change_status_date2"] == "Excavation")
#         | (test_df["change_status_date3"] == "Excavation")
#         | (test_df["change_status_date4"] == "Excavation")
#         | (test_df["change_status_date5"] == "Excavation")
#     ].index
# )

# dropping rows if "Na" values in change_status_datei

# train_df = train_df.drop(
#     train_df[
#         (train_df["change_status_date1"] == "Na")
#         | (train_df["change_status_date2"] == "Na")
#         | (train_df["change_status_date3"] == "Na")
#         | (train_df["change_status_date4"] == "Na")
#         | (train_df["change_status_date5"] == "Na")
#     ].index
# )
# test_df = test_df.drop(
#     test_df[
#         (test_df["change_status_date1"] == "Na")
#         | (test_df["change_status_date2"] == "Na")
#         | (test_df["change_status_date3"] == "Na")
#         | (test_df["change_status_date4"] == "Na")
#         | (test_df["change_status_date5"] == "Na")
#     ].index
# )


# creating vectors of 0 or 1 if the construction finished before the 5 days or not:
# t_train = train_df[
#     [
#         "change_status_date1",
#         "change_status_date2",
#         "change_status_date3",
#         "change_status_date4",
#         "change_status_date5",
#     ]
# ].values

# finished_train = []
# for i in range(t_train.shape[0]):
#     if "Construction Done" not in t_train[i, :]:
#         finished_train.append(1)
#     else:
#         finished_train.append(0)
# train_df["finished"] = np.array(finished_train)

# t_test = test_df[
#     [
#         "change_status_date1",
#         "change_status_date2",
#         "change_status_date3",
#         "change_status_date4",
#         "change_status_date5",
#     ]
# ].values

# finished_test = []
# for i in range(t_test.shape[0]):
#     if "Construction Done" not in t_test[i, :]:
#         finished_test.append(1)
#     else:
#         finished_test.append(0)
# test_df["finished"] = np.array(finished_test)


# adding time differences

train_df["diff1"] = (
    train_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date1"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff2"] = (
    train_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff3"] = (
    train_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff4"] = (
    train_df["date5"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)


test_df["diff1"] = (
    test_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date1"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff2"] = (
    test_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff3"] = (
    test_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff4"] = (
    test_df["date5"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)

# Utils


# def moy(L):
#     if len(L) == 0:
#         return 0
#     return sum(L)/len(L)


# def most_freq(lst):
#     if len(lst) == 0:
#         return 13
#     return max(set(lst), key=lst.count)


# steps_tuple = [('Greenland', 'Land Cleared'),
#                ('Prior Construction',),
#                ('Materials Dumped',),
#                ('Construction Started', 'Excavation'),
#                ('Construction Midway',),
#                ('Construction Done', 'Operational'),
#                ]
# format_date = "%d-%m-%Y"
# N_a = len(steps_tuple)-1
# steps = ()
# for step_tuple in steps_tuple:
#     steps += step_tuple


# # Step (string) to avancement (int)
# def step2av(step):
#     '''
#     Input: step is a string
#     Output: av is an int representing avancement
#     '''
#     for i, steps in enumerate(steps_tuple):
#         if step in steps:
#             return i
#     if step != 'NA':
#         print(step)
#     return 'NA'


# def extract_features_date(df):
#     global n_na_found, n_na_only_found
#     n_na_found = 0
#     n_na_only_found = 0
#     verbose = False

#     int_adv_to_D = [list() for _ in range(N_a)]
#     int_adv_to_M = [list() for _ in range(N_a)]

#     # Extract duration and month where each advancement was made.

#     def augment_date(row):

#         # Feature for describing if construction is done at date5
#         is_constructed = int(row['change_status_date5'] == 'Construction Done')

#         # List [duration_for_reaching_avancement_A for A in Avancements]
#         duration_for_reaching = ['ToCompute' for _ in range(N_a)]
#         # List [int_representing_month_where_advancementA_was_made for A in Avancements]
#         month_where_was = ['ToCompute' for _ in range(N_a)]
#         # [0,0,1,3,5]
#         L_int_steps = [step2av(row[status]) for status in (
#             'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4', 'change_status_date5')]

#         # If some steps are NA return list of unknown
#         if 'NA' in L_int_steps:
#             # + ["Unknown" for _ in range(N_a)]   #MONTHS
#             return [is_constructed] + ["Unknown" for _ in range(N_a)]

#         # Each time we do an advancement (ie step changes), we fill the list duration_for_reaching with the diff time.
#         # If severals advancements are made we divise the duration by the number of advancements.
#         # To implement: instead of divising by the duration, for each advancement A, we do duration(A) = D(A) / Sum_A(D(A)) where D(A) is the mean duration of the advancement, computed on data where advancement was reached in one step
#         for k in range(len(L_int_steps)-1):
#             int_step = L_int_steps[k]
#             int_step_next = L_int_steps[k+1]
#             if verbose:
#                 print("STEP", k, int_step, int_step_next)

#             if int_step_next > int_step:
#                 if verbose:
#                     print(
#                         f"step {int_step} to step {int_step_next} happened at time {k}")
#                 for u in range(int_step, int_step_next):
#                     # If severals advancement are made between only two dates, the duration of each advancement is the duration divided by the number of dates-1.
#                     duration_for_reaching[u] = (
#                         row['diff' + str(k+1)] // (int_step_next-int_step)) / (3600 * 24 * 30.5)

#                     # The month where EVERY (Implement: not every advancements happend at the same time...) advancements are made is the month of the date between two dates
#                     t1 = datetime.timestamp(datetime.strptime(
#                         row["date" + str(k+1)], format_date))
#                     t2 = datetime.timestamp(datetime.strptime(
#                         row["date" + str(k+2)], format_date))
#                     month_where_was[u] = datetime.fromtimestamp(
#                         t1 + (t2-t1)/2).month
#                 if int_step_next - int_step == 1:
#                     # Save the duration and the mean in list
#                     int_adv_to_D[int_step].append(duration_for_reaching[u])
#                     int_adv_to_M[int_step].append(month_where_was[u])

#         if 'ToCompute' in duration_for_reaching:
#             # print(duration_for_reaching)
#             pass

#         L = [is_constructed, ] + duration_for_reaching
#         # L += month_where_was                               #MONTHS
#         # sys.exit()
#         return L

#     # Nom des features
#     columns_names = ['is_constructed'] + ['duration_to_reach' +
#                                           str(step2av(step[0])) for step in steps_tuple[1:]]
#     # columns_names += ['month_of_advancement' + str(step2av(step[0])) for step in steps_tuple[1:]]                      #MONTHS

#     # Features augmentées
#     df_augment = df.apply(lambda row: pd.Series(
#         augment_date(row), index=columns_names), axis=1)

#     # Fill
#     # Features avec des NA remplacés par durées moyennes
#     int_adv_to_D = [moy(L) for L in int_adv_to_D]
#     int_adv_to_M = [most_freq(L) for L in int_adv_to_M]

#     def fill_NA(row):
#         global n_na_found
#         R = row.copy()

#         # The last ToCompute values (meaning it was not computed ie advancement hasnt been reached) are given the values None
#         for col in ['duration_to_reach' + str(step2av(step[0])) for step in steps_tuple[1:]][::-1]:
#             if row[col] == "ToCompute":
#                 R[col] = None
#             else:
#                 break
#         # The first ToCompute values (meaning for advancement i->i+1, step i hasnt been seen because photos were made after this step) are given the mean values
#         for i, col in enumerate(['duration_to_reach' + str(step2av(step[0])) for step in steps_tuple[1:]]):
#             if row[col] in ("ToCompute", "Unknown"):
#                 R[col] = int_adv_to_D[i]

#         return R

#     df_augment = df_augment.apply(lambda row: pd.Series(
#         fill_NA(row), index=columns_names), axis=1)

#     print(f"NA :{n_na_found}")
#     return df_augment


# df_aug = extract_features_date(train_df)
# train_df = pd.merge(
#     left=train_df,
#     right=df_aug,
#     left_index=True,
#     right_index=True,
# )

# df_aug2 = extract_features_date(test_df)
# test_df = pd.merge(
#     left=test_df,
#     right=df_aug2,
#     left_index=True,
#     right_index=True,
# )

def season(day):
    # "day of year" ranges for the northern hemisphere
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else

    if day in spring:
        season = 0
    elif day in summer:
        season = 1
    elif day in fall:
        season = 2
    else:
        season = 3
    return season


def weekend(day):
    if day < 5:
        return 0
    else:
        return 1


for x in ["date1", "date2", "date3", "date4", "date5"]:
    # weekday
    train_df["weekday" + "_" + x] = train_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").weekday()
    )
    test_df["weekday" + "_" + x] = test_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").weekday()
    )

    # month
    train_df["month" + "_" + x] = train_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").month
    )
    test_df["month" + "_" + x] = test_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").month
    )

    # year
    train_df["year" + "_" + x] = train_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").year
    )
    test_df["year" + "_" + x] = test_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").year
    )

    # season

    train_df["season" + "_" + x] = train_df[x].apply(
        lambda x: season(datetime.strptime(
            x, "%d-%m-%Y").today().timetuple().tm_yday)
    )
    test_df["season" + "_" + x] = test_df[x].apply(
        lambda x: season(datetime.strptime(
            x, "%d-%m-%Y").today().timetuple().tm_yday)
    )

    # weekend flag

    # train_df["weekend" + "_" + x] = train_df[x].apply(
    #     lambda x: weekend(datetime.strptime(
    #         x, "%d-%m-%Y").weekday())
    # )
    # test_df["weekend" + "_" + x] = test_df[x].apply(
    #     lambda x: season(datetime.strptime(
    #         x, "%d-%m-%Y").weekday())
    # )


# adding perimters and areas of polygons

train_df["area"] = train_df["geometry"].area
train_df["length"] = train_df["geometry"].length


test_df["area"] = test_df["geometry"].area
test_df["length"] = test_df["geometry"].length


# # # square root transformation

# train_df["sqrt_area"] = np.sqrt(train_df["area"])
# test_df["sqrt_area"] = np.sqrt(test_df["area"])


# # # length squared

# train_df["squared_length"] = train_df["length"]**2
# test_df["squared_length"] = test_df["length"]**2


# # # area over length squared

train_df["area/length**2"] = train_df["area"]/(train_df["length"]**2)
test_df["area/length**2"] = test_df["area"]/(test_df["length"]**2)

# elongation

borders_train = train_df["geometry"]
borders_test = test_df["geometry"]


borders_train_el = train_df["geometry"].boundary
borders_test_el = test_df["geometry"].boundary

n = borders_train.shape[0]
p = borders_test.shape[0]


def elong(u):
    pt = list(u.coords)
    pt1 = max(pt, key=lambda x: x[1])   # point with maximal ordinate
    pt2 = min(pt, key=lambda x: x[1])   # point with minimal ordinate
    pt3 = max(pt, key=lambda x: x[0])   # point with maximal abscissa
    pt4 = min(pt, key=lambda x: x[0])   # point with minimal abscissa
    return max(pt1[1]-pt2[1], pt3[0]-pt4[0])


dic_train = {}
dic_test = {}

dic_train['elongation'] = [
    elong(borders_train_el.iloc[i]) for i in range(n)]
dic_test['elongation'] = [
    elong(borders_test_el.iloc[i]) for i in range(p)]

df_train_elong = pd.DataFrame.from_dict(dic_train)
df_test_elong = pd.DataFrame.from_dict(dic_test)


train_df = pd.merge(
    left=train_df,
    right=df_train_elong,
    left_index=True,
    right_index=True,
)

test_df = pd.merge(
    left=test_df,
    right=df_test_elong,
    left_index=True,
    right_index=True,
)


# minx etc.

train_df['minx'] = train_df['geometry'].bounds.minx
train_df['miny'] = train_df['geometry'].bounds.miny
train_df['maxx'] = train_df['geometry'].bounds.maxx
train_df['maxy'] = train_df['geometry'].bounds.maxy

test_df['minx'] = test_df['geometry'].bounds.minx
test_df['miny'] = test_df['geometry'].bounds.miny
test_df['maxx'] = test_df['geometry'].bounds.maxx
test_df['maxy'] = test_df['geometry'].bounds.maxy

train_df['centroid_x'] = train_df['geometry'].centroid.x
train_df['centroid_y'] = train_df['geometry'].centroid.y
test_df['centroid_x'] = test_df['geometry'].centroid.x
test_df['centroid_y'] = test_df['geometry'].centroid.y

train_df['height'] = train_df['maxy']-train_df['miny']
train_df['width'] = train_df['maxx']-train_df['minx']

test_df['height'] = test_df['maxy']-test_df['miny']
test_df['width'] = test_df['maxx']-test_df['minx']

# nb of points in convex_hull

train_df['nb_points'] = train_df['geometry'].convex_hull.boundary.apply(
    lambda x: len(x.coords))
test_df['nb_points'] = test_df['geometry'].convex_hull.boundary.apply(
    lambda x: len(x.coords))

# diff of area between original polygon and the rotated rectangle (ie check the likelihood with a rectangle)

train_df['interm'] = train_df['geometry']
train_df['diff_area'] = train_df['geometry'].apply(
    lambda x: x.minimum_rotated_rectangle.area)-train_df['area']
train_df = train_df.drop("interm", axis=1)

test_df['interm'] = test_df['geometry']
test_df['diff_area'] = test_df['geometry'].apply(
    lambda x: x.minimum_rotated_rectangle.area)-test_df['area']
test_df = test_df.drop("interm", axis=1)

# compare the area of the polygon and the area of the circumscribed circle


def d(x, y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)


# def area_circle(u):
#     centr = u.centroid  # center of the circle
#     c = [centr.x, centr.y]
#     pt = list(u.boundary.coords)
#     pt = sorted(pt, key=lambda x: d(x, c))
#     pmax = pt[-1]    # points the farest from centroid
#     return d(c, pmax)**2*pi


# dic_train = {}
# dic_test = {}

# dic_train['area_circle'] = [area_circle(borders_train.iloc[i])
#                             for i in range(n)]
# dic_test['area_circle'] = [area_circle(borders_test.iloc[i])
#                            for i in range(p)]

# df_train_circle = pd.DataFrame.from_dict(dic_train)
# df_test_circle = pd.DataFrame.from_dict(dic_test)

# train_df = pd.merge(
#     left=train_df,
#     right=df_train_circle,
#     left_index=True,
#     right_index=True,
# )

# test_df = pd.merge(
#     left=test_df,
#     right=df_test_circle,
#     left_index=True,
#     right_index=True,
# )

# train_df['area_circle'] = train_df['area_circle']-train_df['area']
# test_df['area_circle'] = test_df['area_circle']-test_df['area']


# test convexity (1 if convex, 0 if not)


train_df['is_convex'] = train_df['geometry'].apply(
    lambda x: list(x.boundary.coords)[:-1])
train_df['is_convex'] = train_df['is_convex'].apply(
    lambda x: is_convex_polygon(x))
train_df['is_convex'] = train_df['is_convex'].astype(int)

test_df['is_convex'] = test_df['geometry'].apply(
    lambda x: list(x.boundary.coords)[:-1])
test_df['is_convex'] = test_df['is_convex'].apply(
    lambda x: is_convex_polygon(x))
test_df['is_convex'] = test_df['is_convex'].astype(int)


# sum of the distances from the centroid to the summits of the rotated rectangle


def sum_dist(u):
    c = u.centroid
    centroid = [c.x, c.y]
    rect = list(set(u.minimum_rotated_rectangle.boundary.coords))

    sum = 0
    for i in range(4):
        sum += d(centroid, rect[i])
    return sum

def sum_dist2(u):
    c = u.centroid
    centroid = [c.x, c.y]
    polyg = u.boundary.coords

    sum = 0
    n_points = len(polyg)
    if n_points ==0: return None    # or 0 ?
    for i in range(n_points):
        sum += d(centroid, polyg[i])
    return sum/n_points


train_df['centroid_dist'] = train_df['geometry']
train_df['centroid_dist'] = train_df['centroid_dist'].apply(
    lambda x: sum_dist(x))

test_df['centroid_dist'] = test_df['geometry']
test_df['centroid_dist'] = test_df['centroid_dist'].apply(
    lambda x: sum_dist(x))


# distances from centroid to each summit of the rectangle

# def dist_centr(u, i):
#     c = u.centroid
#     centroid = [c.x, c.y]
#     rect = list(set(u.minimum_rotated_rectangle.boundary.coords))

#     if i == 0:
#         minx = sorted(rect, key=lambda x: x[0])[0]
#         return d(centroid, minx)
#     elif i == 1:
#         miny = sorted(rect, key=lambda x: x[1])[0]
#         return d(centroid, miny)
#     elif i == 2:
#         maxx = sorted(rect, key=lambda x: x[0])[-1]
#         return d(centroid, maxx)
#     else:
#         maxy = sorted(rect, key=lambda x: x[1])[-1]
#         return d(centroid, maxy)


# train_df['minx'] = train_df['geometry']
# train_df['minx'] = train_df['minx'].apply(lambda x: dist_centr(x, 0))

# train_df['maxx'] = train_df['geometry']
# train_df['maxx'] = train_df['maxx'].apply(lambda x: dist_centr(x, 1))

# train_df['miny'] = train_df['geometry']
# train_df['miny'] = train_df['miny'].apply(lambda x: dist_centr(x, 2))

# train_df['maxy'] = train_df['geometry']
# train_df['maxy'] = train_df['maxy'].apply(lambda x: dist_centr(x, 3))


# test_df['minx'] = test_df['geometry']
# test_df['minx'] = test_df['minx'].apply(lambda x: dist_centr(x, 0))

# test_df['maxx'] = test_df['geometry']
# test_df['maxx'] = test_df['maxx'].apply(lambda x: dist_centr(x, 1))

# test_df['miny'] = test_df['geometry']
# test_df['miny'] = test_df['miny'].apply(lambda x: dist_centr(x, 2))

# test_df['maxy'] = test_df['geometry']
# test_df['maxy'] = test_df['maxy'].apply(lambda x: dist_centr(x, 3))


# length/width


def ratio(u):
    c = u.centroid
    dmin = u.boundary.distance(c)
    pt = list(u.boundary.coords)
    c = [c.x, c.y]
    dist = [d(x, c) for x in pt]
    return max(dist)/dmin

    # pt = list(u.coords)
    # pt1 = max(pt, key=lambda x: x[1])   # point with maximal ordinate
    # pt2 = min(pt, key=lambda x: x[1])   # point with minimal ordinate
    # pt3 = max(pt, key=lambda x: x[0])   # point with maximal abscissa
    # pt4 = min(pt, key=lambda x: x[0])   # point with minimal abscissa
    # L = d(pt3, pt4)
    # l = d(pt1, pt2)
    # return max(L, l)/min(L, l)


def ratio2(u):  # using rotated rectangle
    rect = u.minimum_rotated_rectangle
    pt = list(set(rect.boundary.coords))
    pt = sorted(pt, key=lambda x: d(x, pt[0]))
    d1 = d(pt[0], pt[1])
    d2 = d(pt[0], pt[2])
    return d2/d1


dic_train = {}
dic_test = {}

dic_train['length/width'] = [ratio2(borders_train.iloc[i]) for i in range(n)]
dic_test['length/width'] = [ratio2(borders_test.iloc[i]) for i in range(p)]

df_train_ratio = pd.DataFrame.from_dict(dic_train)
df_test_ratio = pd.DataFrame.from_dict(dic_test)


train_df = pd.merge(
    left=train_df,
    right=df_train_ratio,
    left_index=True,
    right_index=True,
)

test_df = pd.merge(
    left=test_df,
    right=df_test_ratio,
    left_index=True,
    right_index=True,
)


# same with squared

# train_df['length/width**2'] = train_df['length/width']**2
# test_df['length/width**2'] = test_df['length/width']**2


train_df = train_df.drop("geometry", axis=1)
test_df = test_df.drop("geometry", axis=1)

dates = ["date1", "date2", "date3", "date4", "date5"]
for d in dates:
    train_df = train_df.drop(d, axis=1)
    test_df = test_df.drop(d, axis=1)

col_str = [
    "change_status_date1",
    "change_status_date2",
    "change_status_date3",
    "change_status_date4",
    "change_status_date5",
]
le = LabelEncoder()
train_df[col_str] = train_df[col_str].apply(le.fit_transform)
test_df[col_str] = test_df[col_str].apply(le.fit_transform)


train_df["urban_types"] = train_df["urban_types"].apply(
    lambda x: x.split(","))
test_df["urban_types"] = test_df["urban_types"].apply(
    lambda x: x.split(","))


train_df = pd.concat(
    [train_df, train_df["urban_types"].str.join("|").str.get_dummies()], axis=1
)
test_df = pd.concat(
    [test_df, test_df["urban_types"].str.join("|").str.get_dummies()], axis=1
)

train_df = train_df.drop("urban_types", axis=1)
test_df = test_df.drop("urban_types", axis=1)


train_df["geography_types"] = train_df["geography_types"].apply(
    lambda x: x.split(","))
test_df["geography_types"] = test_df["geography_types"].apply(
    lambda x: x.split(","))


train_df = pd.concat(
    [train_df, train_df["geography_types"].str.join("|").str.get_dummies()], axis=1
)
test_df = pd.concat(
    [test_df, test_df["geography_types"].str.join("|").str.get_dummies()], axis=1
)

train_df = train_df.drop("geography_types", axis=1)
test_df = test_df.drop("geography_types", axis=1)


train_df["change_type"] = train_df["change_type"].apply(
    lambda x: change_type_map[x])


############  FEATURETOOLS  ###############
# import featuretools as ft
# import woodwork

# # J'ai gardé les dates sous forme purement numérique, parce qu'il faut un time_index
# ES = ft.EntitySet(id='train df')

# logical_types = {}

# ES = ES.add_dataframe(dataframe_name='train date1', dataframe=train_df[['index', 'date1', 'change_status_date1', 'area', 'length']], index='index', logical_types={
#                       'change_status_date1': woodwork.logical_types.Categorical, 'date1': woodwork.logical_types.Datetime}, time_index='date1')
# ES = ES.add_dataframe(dataframe_name='train date2', dataframe=train_df[['index', 'date2', 'change_status_date2', 'area', 'length']], make_index=True, index='index2', logical_types={
#                       'change_status_date2': woodwork.logical_types.Categorical, 'date2': woodwork.logical_types.Datetime}, time_index='date2')

# ES = ES.add_relationship('train date1', 'index', 'train date2', 'index')
# print(ES)

# feature_matrix, feature_defs = ft.dfs(entityset=ES,
#                                       target_dataframe_name='train date1',
#                                       verbose=1
#                                       )
# print(feature_matrix)
######################################


train_df.to_csv("train_df.csv")
test_df.to_csv("test_df.csv")


print('Preprocessing done.')