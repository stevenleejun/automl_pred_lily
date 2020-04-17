# coding = utf-8
from .utils_save import save_df_to_database


class UtilsDataSave:
    _support_database_type = ['mysql', 'greenplum', 'csv']
    _support_field_type = ['varchar', 'float', 'datetime', 'int', 'bool']
    _support_para_value_type = ['string', 'int', 'float', 'bool']

    def __init__(
            self,
            host=None,
            user=None,
            password=None,
            database=None,
            table_name=None,
            port=None,
            is_truncate=False,
            url=None,
            nrows=None,
            sep=',',
            database_type='csv',
            extra_para=[],
    ):
        self.database_type = database_type
        self.url = url
        self.sep = sep
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.port = port
        self.nrows = nrows
        self.is_truncate = is_truncate
        extra_para = extra_para

        kwargs = {}
        for para_info in extra_para:
            para_name = para_info['para_name']
            para_value = para_info['para_value']
            para_value_type = para_info['para_value_type']
            para_value = self._change_database_para_value_type(para_value, para_value_type)
            kwargs[para_name] = para_value

        self.kwargs = kwargs
        assert self.database_type in self._support_database_type

    def process(self, data_df, index=False):
        if self.database_type == 'csv':
            data_df.to_csv(file=self.url, sep=self.sep, nrows=self.nrows, index=index)

        elif self.database_type in ['greenplum', 'mysql']:
            save_df_to_database(
                data_df=data_df,
                database_type=self.database_type,
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                table_name=self.table_name,
                port=self.port,
                is_truncate=self.is_truncate,
                **self.kwargs
            )

    def _change_database_para_value_type(
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
