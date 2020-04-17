import base64
import os
from .base_process import BaseProcess


class EntitySetPreviewProcess(BaseProcess):
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

    def _resolve_process_config(self):
        self.nrows = 0

    def process(self):
        entity_set_data_dfs_dic = self.load_entity_set_data()

        entity_set_cutoff_time_dic = self.get_entity_set_cutoff_time_dic(entity_set_data_dfs_dic)
        entity_set = entity_set_cutoff_time_dic['entity_set']

        tmp_dir = '../../tmp_data'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_path = os.path.join(tmp_dir, 'entity_set_build.png')
        entity_set.plot(to_file=tmp_path)
        with open(tmp_path, 'rb') as file:
            entity_set_jpg_base64 = base64.b64encode(file.read())
            entity_set_jpg_base64 = entity_set_jpg_base64.decode('utf-8')

        return entity_set_jpg_base64
