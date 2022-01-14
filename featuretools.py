import featuretools as ft
import woodwork

# J'ai gardé les dates sous forme purement numérique, parce qu'il faut un time_index
ES = ft.EntitySet(id='train df')

logical_types = {}

ES = ES.add_dataframe(dataframe_name='train date1', dataframe=train_df[['index', 'date1', 'change_status_date1', 'area', 'length']], index='index', logical_types={
                      'change_status_date1': woodwork.logical_types.Categorical, 'date1': woodwork.logical_types.Datetime}, time_index='date1')
ES = ES.add_dataframe(dataframe_name='train date2', dataframe=train_df[['index', 'date2', 'change_status_date2', 'area', 'length']], make_index=True, index='index2', logical_types={
                      'change_status_date2': woodwork.logical_types.Categorical, 'date2': woodwork.logical_types.Datetime}, time_index='date2')

ES = ES.add_relationship('train date1', 'index', 'train date2', 'index')
print(ES)

feature_matrix, feature_defs = ft.dfs(entityset=ES,
                                      target_dataframe_name='train date1',
                                      verbose=1
                                      )
feature_matrix
