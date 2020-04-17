import logging
logging.basicConfig(level=logging.DEBUG)
from utils.utils_load_class import UtilsDataLoad


class EntitySetDataLoad:
    def __init__(
            self,
            entity_set_info,
            ana_type_vs_data_type=None,
            logger=None,
            is_save=1,
            nrows=None,
    ):
        self.entity_set_info = entity_set_info
        self.ana_type_vs_data_type = ana_type_vs_data_type
        self.logger = logger
        self.is_save = is_save
        self.nrows = nrows

    def process(self):
        # 处理真实实体
        data_dfs_dic = {}
        origin_entity = {entity_name: entity_name_info
                         for entity_name, entity_name_info in self.entity_set_info.items()
                         if entity_name_info.get('normalize_base_entity', None) is None}
        for entity_name, entity_name_info in origin_entity.items(): 
            self.logger.info('entity_name:{} begin'.format(entity_name))
            table_name = entity_name_info.get('table_name', entity_name)
            sql = entity_name_info.get('sql', None)
            url = entity_name_info.get('url', None)
            sep = entity_name_info.get('url_sep', ',')
            url_type = entity_name_info.get('url_type', 'csv')
            db_info = entity_name_info.get('db_info', {})
            filter_data_list = entity_name_info.get('filter_data_list', [])
            field_list = entity_name_info.get('field_list', [])
            field_df_type_list = self._transform_field_df_type_list(field_list)
            data_df = self.load_single_data(
                sql=sql,
                url=url,
                sep=sep,
                url_type=url_type,
                db_info=db_info,
                field_df_type_list=field_df_type_list,
                table_name=table_name,
                filter_data_list=filter_data_list
            )
            self.logger.info('entity_name:{} 的shape为:{}'.format(entity_name, data_df.shape))
            data_dfs_dic[entity_name] = data_df
        return data_dfs_dic

    def load_single_data(
            self,
            sql=None,
            url=None,
            sep=',',
            db_info={},
            url_type=None,
            field_df_type_list=[],
            table_name=None,
            filter_data_list=[]
    ):
        if url is not None:
            database_type = url_type
        else:
            database_type = db_info.get('type', None)

        host = db_info.get('host', None)
        database = db_info.get('database', None)
        port = db_info.get('port', None)
        user = db_info.get('user', None)
        password = db_info.get('password', None)
        extra_para = db_info.get('extra_para', {})

        utils_data_load = UtilsDataLoad(
            host=host,
            database=database,
            port=port,
            user=user,
            password=password,
            sql=sql,
            url=url,
            sep=sep,
            extra_para=extra_para,
            database_type=database_type,
            nrows=self.nrows,
            table_name=table_name
        )
        data_df = utils_data_load.process()
        data_df = utils_data_load.standardize_data_type(
            data_df=data_df,
            field_df_type_list=field_df_type_list,
        )

        # 数据筛选功能
        for filter_data_info in filter_data_list:
            column = filter_data_info.get('column', None)
            value_list = filter_data_info.get('value', [])
            if column and (column in data_df.columns):
                before_shape = data_df.shape
                data_df = data_df[~data_df[column].isin(value_list)]
                self.logger.info('column: {column} is filter by {value_list}, before_shape: {before_shape}, after_shape: {after_shape}'.format(
                    column=column,
                    value_list=value_list,
                    before_shape=before_shape,
                    after_shape=data_df.shape
                ))
            else:
                self.logger.error('column: {column} is None or not in data_df.columns:{data_df_columns}'.format(
                    column=column,
                    data_df_columns=data_df.columns
                ))

        return data_df

    def _transform_field_df_type_list(self, field_list=None):
        if field_list is None:
            return []

        field_df_type_list = []
        for field_ana_dic in field_list:
            variable_type = field_ana_dic['variable_type']
            assert variable_type in self.ana_type_vs_data_type.keys(), \
                '{} not in {}'.format(variable_type, self.ana_type_vs_data_type.keys())

            field_df_type_list.append({
                'name':  field_ana_dic['name'],
                'variable_type': self.ana_type_vs_data_type.get(variable_type, None)
            })
        return field_df_type_list

