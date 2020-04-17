import shutil
import os
import sys

from .base_process import BaseProcess


class ModelDeployingProcess(BaseProcess):
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
        self.model_url = self.process_config.get('model_url', self.default_model_url)

    def process(self):
        if not os.path.exists(self.model_url):
            self.logger.error('model_url:{} not exists'.format(self.model_url))
            return None

        new_project_dir = self.base_project_dir_deploying + '/' + self.model_url.replace(self.base_project_dir, '')
        if os.path.exists(new_project_dir):
            shutil.rmtree(new_project_dir)
        try:
            shutil.copytree(self.model_url, new_project_dir)
        except Exception as e:
            self.logger.error('error:{}'.format(e))
            return None

        return new_project_dir

