# coding=utf-8

from .utils_api_base_querier import BaseQuerier
from .utils_api_base_querier import BaseQuerierWithCallback

def api_get_resource(
        api_config=None,
        api_name=None,
        querier_fun=None,
        parser_args=None,
        resource_class=None,
):
    callback_id_name = api_config[api_name].get('callback_id_name', api_name + '_id')
    callback_status_name = api_config[api_name].get('callback_status_name', api_name + '_status')

    callback_url = api_config[api_name].get('callback_url', None)
    if callback_url is None:
        querier_class = BaseQuerier
    else:
        querier_class = BaseQuerierWithCallback

    kwargs = {}
    if querier_class.__name__ == 'BaseQuerierWithCallback':
        kwargs['callback_url'] = callback_url
        kwargs['callback_id_name'] = callback_id_name
        kwargs['callback_status_name'] = callback_status_name

    post_querier = querier_class(querier_fun=querier_fun, **kwargs)
    resource_class.set_resource(
        parser_args=parser_args,
        post_querier=post_querier
    )
    return resource_class
