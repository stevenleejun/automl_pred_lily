# coding = utf-8
import pandas as pd
import numpy as np
from .utils_load import read_file_to_df
from .utils_load import read_database_to_df
from .utils_df import get_df_columns_by_data_type
import logging

class UtilsDataLoad:
    _support_database_type = ['mysql', 'greenplum', 'hive', 'csv', 'excel']
    _support_field_type = ['varchar', 'float', 'datetime', 'int', 'bool']
    _support_para_value_type = ['string', 'int', 'float', 'bool']

    def __init__(
            self,
            host,
            database,
            port=None,
            table_name=None,
            user=None,
            password=None,
            sql=None,
            url=None,
            sep=',',
            database_type='csv',
            nrows=None,
            extra_para={},
    ):
        self.database_type = database_type
        self.nrows = nrows

        self.sql = sql
        self.url = url
        self.sep = sep
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.port = port
        extra_para = extra_para

        kwargs = {}
        for para_info in extra_para:
            para_name = para_info['para_name']
            para_value = para_info['para_value']
            para_value_type = para_info['para_value_type']
            para_value = self.change_para_value_type(para_value, para_value_type)
            kwargs[para_name] = para_value

        self.kwargs = kwargs
        assert self.database_type in self._support_database_type

    def process(self, sql=None, table_name=None):
        if sql:
            self.sql = sql
        if table_name:
            self.table_name = table_name

        if self.database_type == 'csv':
            data_df = read_file_to_df(filepath=self.url, source_type='csv', sep=self.sep, nrows=self.nrows)
        elif self.database_type == 'excel':
            data_df = read_file_to_df(filepath=self.url, source_type='excel', sep=self.sep, nrows=self.nrows)
        elif self.database_type in ['hive', 'greenplum', 'mysql']:
            data_df = read_database_to_df(
                database_type=self.database_type,
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                table_name=self.table_name,
                port=self.port,
                sql=self.sql,
                nrows=self.nrows,
                **self.kwargs
            )
        else:
            data_df = None

        return data_df

    def standardize_data_type(
            self,
            data_df,
            field_df_type_list=[],
    ):
        # 指定 数据类型转化
        for field_type in field_df_type_list:
            name = field_type['name']
            variable_type = field_type['variable_type']
            if name in data_df.columns:
                data_df[name] = self._change_pd_type(
                    single_col=data_df[name],
                    variable_type=variable_type
                )

        # 常规
        datetime_features_columns = get_df_columns_by_data_type(
            data_df=data_df,
            data_type='datetime'
        )
        # 数据类型转化
        # pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 1190-07-31 00:00:00
        date_cols = datetime_features_columns
        for col in date_cols:
            if col in data_df.columns:
                data_df[col] = pd.to_datetime(data_df[col], errors='coerce')
                data_df[col][data_df[col].astype('str') <= '1900-01-01 00:00:00'] = np.nan

        return data_df

    def _change_pd_type(
            self,
            single_col,
            variable_type
    ):
        assert variable_type in self._support_field_type, \
            'variable_type:{} not in {}'.format(variable_type, self._support_field_type)

        if variable_type == 'datetime':
            single_col = pd.to_datetime(single_col, errors='coerce')
        elif variable_type == 'float':
            single_col = pd.to_numeric(single_col, errors='coerce')
            single_col = single_col.astype('float')
        elif variable_type == 'int':
            single_col = pd.to_numeric(single_col, downcast='integer', errors='coerce')
            single_col = single_col.astype('Int64')
        elif variable_type == 'varchar':
            single_col = single_col.astype('object')
        elif variable_type == 'bool':
            single_col = single_col.astype('bool')

        return single_col

    def change_para_value_type(
            self,
            value,
            value_type
    ):
        assert value_type in self._support_para_value_type, self._support_para_value_type
        if value_type == 'string':
            value = str(value)
        elif value_type == 'int':
            value = int(value)
        elif value_type == 'float':
            value = float(value)
        elif value_type == 'bool':
            value = bool(value)

        return value
