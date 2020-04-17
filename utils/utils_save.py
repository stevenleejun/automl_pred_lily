# coding: utf-8
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import dialects
import psycopg2
import pymysql
import json
import sys
from pandas.io import sql
# 每个database的insert语句不一样，所以不能直接同意为save_df_to_database
# insert gp很慢，需要直接用insert语句

def save_df_to_database(
        data_df,
        host,
        user,
        password,
        database,
        table_name,
        database_type,
        port=5432,
        is_truncate=False,
        is_init=False,
        df_type_desc_df=pd.DataFrame(),
        is_index=False,
        index_label="id",
        primary_keyl=None,
        batch_size=5000,
        special_dtype=None,
        **kwargs
):

    if database_type == 'greenplum':
        schema = kwargs.get('schema', 'public')
        schema_table_name = schema + '.' + table_name
        dbschema = schema
    else:
        schema_table_name = table_name

    if database_type == 'greenplum':
        conn = create_engine('postgresql+psycopg2://{}:{}@{}:{}/{}'.format(user, password, host, port, database),
                             connect_args={'options': '-csearch_path={}'.format(dbschema)})
        db_con = psycopg2.connect(host=host, user=user, password=password, database=database)
    if database_type == 'mysql':
        conn = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(user, password, host, port, database))
        db_con = pymysql.connect(host=host, user=user, password=password, database=database, port=port, charset="utf8")
    cur = db_con.cursor()

    if is_init:  # 需要重建表
        exe_sql = "drop table if exists {}".format(schema_table_name)
        cur.execute(exe_sql)
        db_con.commit()

    sql_db_pd = sql.SQLDatabase(conn)
    if not sql_db_pd.has_table(table_name):
        args = [table_name, sql_db_pd]
        kwargs = {
            "frame": data_df,
            "index": is_index,
            "index_label": index_label,
            "keys": primary_keyl,
            "dtype": sqlcoltypedict(df_type_desc_df, special_dtype, type=database_type, data_df=data_df)

        }
        sql_table = sql.SQLTable(*args, **kwargs)
        sql_table.create()

    if is_truncate:
        exe_sql = "truncate table {}".format(schema_table_name)
        cur.execute(exe_sql)
        db_con.commit()

    if database_type == 'mysql':
        data_df.to_sql(table_name, conn, if_exists='append', chunksize=10000, index=False)
    elif database_type == 'greenplum':
        data_df = data_df.replace({'"': ' ', "'": ' '}, regex=True)
        batch_count = int(np.ceil(data_df.shape[0] / batch_size))
        for i in range(batch_count):
            df_tmp = data_df.iloc[i * batch_size:(i + 1) * batch_size, :]
            data_list_sql = json.dumps(df_tmp.values.tolist(), ensure_ascii=False, default=str).replace(']]', "]").replace(
                '[[', "[").replace('"', "'").replace('[', '(').replace(']', ')').replace("'NaT'", "null").replace("NaN", "null").replace("None", "null")
            exe_sql = 'INSERT INTO {} VALUES {};'.format(schema_table_name, data_list_sql)
            cur.execute(exe_sql)
            db_con.commit()
    db_con.close()


def sqlcoltypedict(df_type_desc_df, special_dtype, type, data_df):
    dtypedict = {}
    if df_type_desc_df.shape[0] > 0:
        df_type_desc_df.fillna('0', inplace=True)
        df_type_desc_df['character_maximum_length'] = df_type_desc_df['character_maximum_length'].astype('int', casting='safe')
        df_type_desc_df['numeric_precision'] = df_type_desc_df['numeric_precision'].astype('int', casting='safe')
        df_type_desc_df['numeric_scale'] = df_type_desc_df['numeric_scale'].astype('int', casting='safe')

    else:
        df_type_desc_df = pd.DataFrame(
            {'column_name': data_df.columns.tolist(),
             'data_type': data_df.dtypes.tolist()
             })
        df_type_desc_df['character_maximum_length'] = 255
        df_type_desc_df['numeric_precision'] = 16
        df_type_desc_df['numeric_scale'] = 2

    for i, row in df_type_desc_df.iterrows():
        column_name = row['column_name']
        data_type = str(row['data_type'])

        character_maximum_length = (row['character_maximum_length'])
        if row['character_maximum_length']==0:
            character_maximum_length = 255

        numeric_precision = (row['numeric_precision'])
        if row['numeric_precision']==0:
            numeric_precision = 16

        numeric_scale = (row['numeric_scale'])
        if row['numeric_scale']==0:
            numeric_scale = 4

        if type == 'mysql':
            if "object" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.VARCHAR(length=character_maximum_length)})
            elif "datetime" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.DATETIME()})
            elif "float" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.DECIMAL(precision=numeric_precision, scale=numeric_scale, asdecimal=True)})
            elif ("Int64" in data_type) or ('int' in data_type):
                dtypedict.update({column_name: sqlalchemy.types.Integer()})
            elif "Int8" in data_type:
                dtypedict.update({column_name: dialects.mysql.SMALLINT(unsigned=True)})
            else:
                pass

        if type == 'greenplum':
            if "object" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.VARCHAR(length=character_maximum_length)})
            elif "datetime" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.TIMESTAMP()})
            elif "float" in data_type:
                dtypedict.update({column_name: sqlalchemy.types.DECIMAL(precision=numeric_precision, scale=numeric_scale, asdecimal=True)})
            elif ("Int64" in data_type) or ('int' in data_type):
                dtypedict.update({column_name: dialects.postgresql.INTEGER()})
            elif "Int8" in data_type:
                dtypedict.update({column_name: dialects.postgresql.SMALLINT()})
            else:
                pass

    dtypedict_special = {}
    if special_dtype:
        for col in special_dtype.keys():
            col_info = special_dtype[col]
            if col_info[0] == 'varchar':
                dtypedict_special[col] = sqlalchemy.types.VARCHAR(length=col_info[1])
            elif col_info[0] == 'decimal':
                dtypedict_special[col] = sqlalchemy.types.DECIMAL(precision=col_info[1], scale=col_info[2], asdecimal=True)
            else:
                pass

    for col, coltype in dtypedict_special.items():
            dtypedict.update({col: coltype})

    return dtypedict