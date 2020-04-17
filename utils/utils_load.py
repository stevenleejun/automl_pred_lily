# coding = utf-8
import pandas as pd
from xlutils.copy import copy
import xlrd
from .utils_database import get_run_sql
from .utils_database import get_database_sqlalchemy_conn
from .utils_database import run_database_sql
import os
import logging
logging.basicConfig(level=logging.DEBUG)

def rewrite_excel(filepath):
    # filepath_excel = xlrd.open_workbook(filepath, encoding_override='ISO-8859-1')
    # filepath_excel = xlrd.open_workbook(filepath, encoding_override='gbk')
    # 包含的字符个数：GB2312 < GBK < GB18030
    filepath_excel = xlrd.open_workbook(filepath, encoding_override='GB18030')
    filepath_excel_copy = copy(filepath_excel)
    # print(filepath)
    # shutil.rmtree(filepath)
    filepath_excel_copy.save(filepath)


def read_file_to_df(
        filepath,
        nrows=None,
        source_type='csv',
        sep=',',
        data_desc_df=pd.DataFrame()
):
    data_df = pd.DataFrame()
    if source_type == 'csv':
        if not os.path.exists(filepath):
            return pd.DataFrame()

        if data_desc_df.shape[0] > 0:
            try:
                data_df = pd.read_csv(filepath, type='str', sep=sep, nrows=nrows)
            except:
                logging.error("data_df = pd.read_csv(filepath, type='str', sep=sep, nrows=nrows) error, no data")
                return pd.DataFrame()
        else:
            try:
                data_df = pd.read_csv(filepath, sep=sep, nrows=nrows)
            except:
                logging.error("data_df = pd.read_csv(filepath, type='str', sep=sep, nrows=nrows) error, no data")
                return pd.DataFrame()

    if source_type == 'excel':
        try:
            if data_desc_df.shape[0] > 0:
                data_df = pd.read_excel(filepath, type='str', nrows=nrows)
            else:
                data_df = pd.read_excel(filepath, nrows=nrows)
                logging.info('################{}'.format(data_df.columns))
        except Exception as e:
            rewrite_excel(filepath)
            if data_desc_df.shape[0] > 0:
                data_df = pd.read_excel(filepath, dtype='str', nrows=nrows)
            else:
                data_df = pd.read_excel(filepath, nrows=nrows)
    return data_df


def read_database_to_df(
        database_type,
        host,
        user,
        password,
        database,
        table_name,
        sql=None,
        port=5432,
        nrows=None,
        modified_date_colname='modified_date',
        modified_date=None,
        **kwargs
):
    # 获取conn
    conn = get_database_sqlalchemy_conn(
        database_type=database_type,
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        **kwargs
    )

    # 获取数据
    schema = kwargs.get('schema', None)
    run_sql = get_run_sql(
        table_name=table_name,
        sql=sql,
        schema=schema,
        modified_date=modified_date,
        modified_date_colname=modified_date_colname,
        nrows=nrows
    )

    data_df = run_database_sql(
        database_type=database_type,
        run_sql=run_sql,
        conn=conn
    )

    return data_df
