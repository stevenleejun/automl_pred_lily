# coding: utf-8
import pandas as pd
import numpy as np
import logging as logger


def get_df_columns_by_data_type(
        data_df,
        data_type,
):
    data_type_vs_key = {
        'object': ['object', 'str'],
        'int': ['int'],
        'float': ['float'],
        'datetime': ['time', 'ns'],
        'bool': ['bool']
    }
    dtypes_zip = zip(data_df.dtypes.index, data_df.dtypes)
    data_type_column_l = []
    for key in data_type_vs_key.get(data_type, []):
        tmp_column_l = [
            col_name for col_name, col_type in dtypes_zip
            if key in str(col_type).lower()
        ]
        data_type_column_l.extend(tmp_column_l)

    return data_type_column_l


def df_standard_depend_on_datadic(
        data_desc_df,
        df,
        is_reorder=True,
        source_namecol='source_namecol',
        to_namecol='to_namecol',
        default_valuecol='default_valuecol',
        primary_keycol='is_key',
        type_namecol='type',
        datetypename='date',
        floattypename='float',
        inttypename='int',
        varchartypename='varchar',
        booltypename='bool',
        default_col_valued={},
        is_merge_col='is_merge_col',
        is_uni_orderid='False'
):
    # data_desc_df删除to_namecol为空的行
    data_desc_df.dropna(subset=[to_namecol], axis=0, inplace=True)
    to_table_len = data_desc_df.shape[0]

    # 数据字典多的
    more_cols = set(data_desc_df[source_namecol]) - set(list(set(set(df.columns.values))))
    source_more_cols = set(list(set(set(df.columns.values)))) - set(data_desc_df[source_namecol])
    logger.info('数据字典多的:{}'.format(more_cols))
    # logger.debug('源数据多的：{}'.format(source_more_cols))
    # logger.debug('源数据有的：{}'.format(df.columns.values))

    # 多余字段 默认值或者nan填充,此时是原始数据表中的字段名称
    for col in more_cols:
        if np.isnan(data_desc_df[data_desc_df[source_namecol]==col][default_valuecol]).values[0]:
            df.loc[:, col] = np.NaN
        else:
            df.loc[:, col] = data_desc_df[data_desc_df[source_namecol]==col][default_valuecol].values[0]

    # rename为数据库中字段命名
    names_dic = dict(zip(data_desc_df[source_namecol], data_desc_df[to_namecol]))
    df.rename(columns=names_dic, inplace=True)

    # 按照数据字典的顺序
    if is_reorder:
        # uni_cols = [col for col in df.columns.tolist() if col in data_desc_df[to_namecol].tolist()]
        to_cols = data_desc_df[to_namecol].tolist()
        df = df[to_cols]

    # 是否有合并单元格的情况
    if is_merge_col in data_desc_df.columns.tolist():
        merge_cols = data_desc_df[data_desc_df[is_merge_col] == 1][to_namecol].tolist()
        df[merge_cols] = df[merge_cols].fillna(method='ffill')

    # 去重
    logger.info('before  shape:{}'.format(df.shape))
    df.drop_duplicates(inplace=True)
    logger.debug('drop_duplicates:{}'.format(df.shape))

    # 用指定值填充,此时是数据字典中的字段名称
    for col in default_col_valued.keys():
        df.loc[:, col] = default_col_valued[col]

    # 是否创建唯一order_id
    if is_uni_orderid:
        if ('dms_nick' in df.columns.values.tolist()) and ('dms_order_id' in df.columns.values.tolist()):
            df['order_id'] = df['dms_nick'] + '_' + df['dms_order_id']

    # 主键唯一性 主键不为空
    if primary_keycol in data_desc_df.columns.tolist():
        data_primary_keyl = data_desc_df[data_desc_df[primary_keycol] == 1][to_namecol].values.tolist()
        if (len(data_primary_keyl) > 0) and (len(data_primary_keyl) < to_table_len-3): #主键为空删除，防止伪主键情况
            logger.debug('data_primary_keyl:{}'.format(data_primary_keyl))
            df.dropna(subset=data_primary_keyl, axis=0, inplace=True)
            logger.debug('dropna:{}'.format(df.shape))
            df.drop_duplicates(subset=data_primary_keyl, keep='last', inplace=True)
            logger.debug('drop_duplicates subset:{}'.format(df.shape))
    logger.info('after all shape:{}'.format(df.shape))

    # 数据类型转化
    date_cols = data_desc_df[to_namecol][data_desc_df[type_namecol] == datetypename]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    float_cols = data_desc_df[to_namecol][data_desc_df[type_namecol]==floattypename]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype('float')

    int_cols = data_desc_df[to_namecol][data_desc_df[type_namecol]==inttypename]
    for col in int_cols:
        if col in df.columns:
            # df[col] = df[col]*1
            # print(col)
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
            df[col] = df[col].astype('Int64')

    bool_cols = data_desc_df[to_namecol][data_desc_df[type_namecol]==booltypename]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col]*1#True,False 转换为0，1，# int必须填充为0，所以取Int
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
            df[col] = df[col].astype('Int8')

    varchar_cols = data_desc_df[to_namecol][data_desc_df[type_namecol]==varchartypename]
    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    return df

