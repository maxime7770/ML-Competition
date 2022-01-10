import geopandas as gpd
import pandas as pd
import numpy as np
from auxil import perimeter
from datetime import datetime


train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)
# print(train_df.columns)


# train_df['perimeter'] = train_df['geometry'].apply(lambda x: perimeter(x))

print((train_df['geometry'].area).max())
print((train_df['geometry'].length).max())


train_df['diff1'] = (train_df['date2'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                     - train_df['date1'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())

print(train_df['diff1'])
