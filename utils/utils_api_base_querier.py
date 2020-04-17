# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import requests
import json
import traceback
import sys

from .utils_tools_init import initlog


class BaseQuerier:

    def __init__(self, querier_fun, **kwargs):
        self.querier_fun = querier_fun

    def build_query(self, args):
        logger = initlog(self.querier_fun.__name__)
        try:
            error = None
            query_re = self.querier_fun(args, logger=logger)
        except Exception as ex:
            query_re = None
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error = str(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))  # 将异常信息转为字符串
            logger.error('{} error: {}'.format(self.querier_fun.__name__, ex))
        return query_re, error

    def build_result(self, query):
        query_re, error = query
        if query_re is None:
            result = {}
            result['code'] = "99999"
            result['message'] = "操作失败:"
            result['data'] = {'error': error}
        else:
            result = {}
            result['code'] = "00000"
            result['message'] = "操作成功"
            result['data'] = {'result': query_re}
        return result


class BaseQuerierWithCallback:
    executor = ThreadPoolExecutor(100)

    def __init__(self, querier_fun, **kwargs):
        self.querier_fun = querier_fun
        self.callback_url = kwargs['callback_url']
        self.callback_id_name = kwargs['callback_id_name']
        self.callback_status_name = kwargs['callback_status_name']

    def build_query(self, args):
        logger = initlog(self.querier_fun.__name__)
        # self.run_long_task(args, logger)
        self.executor.submit(self.run_long_task, args, logger)
        return {"log_url": logger.handlers[0].baseFilename}

    def build_result(self, query):
        result = {}
        result["code"] = "00000"
        result["message"] = "操作成功"
        result["data"] = query
        return result

    def run_long_task(self, args, logger):
        # 运行逻辑
        try:
            result = self.querier_fun(args, logger=logger)
        except Exception as ex:
            result = None
            logger.error('{} error: {}'.format(self.querier_fun.__name__, ex))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error = str(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))  # 将异常信息转为字符串
            logger.error('{} error: {}'.format(self.querier_fun.__name__, error))

        if result is None:
            result = {}
            result[self.callback_id_name] = args[self.callback_id_name]
            result[self.callback_status_name] = 3

        # 回调接口
        logger.info("result：{}".format(result))
        try:
            callback_text = requests.post(
                self.callback_url,
                json=result,
                headers={'Content-type': 'application/json; charset=utf-8'}
            ).text
        except Exception as ex:
            logger.error('callback error: {}'.format(ex))
            logger.error('callback result: {}'.format(result))

            result = {}
            result[self.callback_id_name] = args[self.callback_id_name]
            result[self.callback_status_name] = 3
            # 错误重传
            callback_text = requests.post(
                self.callback_url,
                json=result,
                headers={'Content-type': 'application/json; charset=utf-8'}
            ).text
        code = json.loads(callback_text)["code"]
        if code != "00000":
            logger.error("callback_text：{}".format(callback_text))
        else:
            logger.info("callback_text：{} ok".format(callback_text))
        return code



