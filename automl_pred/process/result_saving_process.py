import os
import pandas as pd
import sys

from .base_process import BaseProcess
from utils.utils_load import read_file_to_df
from utils.utils_save_class import UtilsDataSave


class ResultSavingProcess(BaseProcess):
    def __init__(self,
                 process_config,
                 root_config_path,
                 logger=None):
        super().__init__(
            process_config=process_config,
            root_config_path=root_config_path,
            logger=logger
        )
        self.process_config = process_config
        self.logger = logger

        self._resolve_process_config()

        self.model = None

    # 配置解析
    def _resolve_process_config(self):
        # project_id
        self.result_saving_desc = self.process_config.get('result_saving_desc', {})

    def process(self):
        # 根据标签结果列表
        table_name = self.result_saving_desc['table_name']
        db_info = self.result_saving_desc['db_info']
        result_url = self.result_saving_desc.get('result_url', self.default_prediction_result_url)
        insert_type = self.result_saving_desc['insert_type']
        if insert_type == 'into':
            is_truncate = False
        elif insert_type == 'overwrite':
            is_truncate = True
        else:
            is_truncate = False

        host = db_info['host']
        user = db_info['user']
        password = db_info['password']
        database = db_info['database']
        port = db_info['port']
        database_type = db_info['type']
        extra_para = db_info['extra_para']

        if not os.path.exists(result_url):
            self.logger.error('result_url:{result_url} does not exists'.format(result_url=result_url))
            return False

        data_df = read_file_to_df(
            filepath=result_url,
            source_type='csv',
        )

        utils_data_save = UtilsDataSave(
            table_name=table_name,
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            database_type=database_type,
            is_truncate=is_truncate,
            extra_para=extra_para

        )

        utils_data_save.process(data_df)
        return True