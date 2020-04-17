# -*- coding: utf-8 -*-
# base_resource.py
# Created by Hardy on 11th, Apr
# Copyright 2018 杭州网川教育科技有限公司. All rights reserved.

from flask import jsonify
from flask_restful import Resource
import logging
import hashlib
import copy
from flask import request
from webargs.flaskparser import parser
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

logger = logging.getLogger()
import json

# class BaseRedisResource(Resource):
#
#     parser = None
#     get_querier = None
#     post_querier = None
#     redis_store = None
#
#     @classmethod
#     def set_resource(cls, parser=None, args=None, get_querier=None, post_querier=None, redis_store=None):
#         cls.args = args,
#         cls.parser = parser
#         cls.get_querier = get_querier
#         cls.post_querier = post_querier
#         if redis_store:
#             cls.redis_store = redis_store
#
#     def get(self):
#         if self.__class__.get_querier:
#             args = self.__class__.parser.parse_args()
#             class_name = self.__class__.__name__
#
#             md5 = gen_md5(args, class_name)
#             logger.debug('get args: %s md5: %s' % (str(args), md5))
#
#             if md5 and self.__class__.redis_store:
#                 res = self.__class__.redis_store.get(md5)
#                 if res:
#                     logger.debug('class: %s args: %s md5: %s' % (class_name, str(args), str(md5)))
#                     return jsonify(json.loads(res.decode('utf-8')))
#
#             try:
#                 res = self.__class__.get_querier.search(args)
#                 logger.debug('class: %s args: %s md5: %s' % (class_name, str(args), str(md5)))
#                 if self.__class__.redis_store:
#                     self.__class__.redis_store.set(md5, json.dumps(res))
#                     self.__class__.redis_store.expire(md5, 3600 * 4)
#             except Exception as ex:
#                 logger.error('error: %s' % ex, exc_info=True)
#                 res = {'message': str(ex)}
#             return jsonify(res)
#         else:
#             return jsonify({'message': 'no GET handler binded'})
#
#     def post(self):
#         if self.__class__.post_querier:
#             args = self.__class__.parser.parse_args()
#             # if not request.get_json():
#             #     return ('No input data provided')
#             # args = request.get_json().get("posts")
#
#             class_name = self.__class__.__name__
#             md5 = gen_md5(args, class_name)
#
#             if md5 and self.__class__.redis_store:
#                 res = self.__class__.redis_store.get(md5)
#
#                 if res:
#                     logger.debug('hit: class: %s args: %s md5: %s' % (class_name, str(args), str(md5)))
#                     return jsonify(json.loads(res.decode('utf-8')))
#
#             try:
#                 res = self.__class__.post_querier.evaluate(args)
#                 logger.debug('class: %s args: %s md5: %s' % (class_name, str(args), str(md5)))
#                 if res and self.__class__.redis_store:
#                     logger.debug('save: class: %s args: %s md5: %s' % (class_name, str(args), str(md5)))
#                     self.__class__.redis_store.set(md5, json.dumps(res))
#                     self.__class__.redis_store.expire(md5, 3600 * 4)
#             except Exception as ex:
#                 logger.error('error: %s' % ex, exc_info=True)
#                 res = {'message': str(ex)}
#             return jsonify(res)
#         else:
#             return jsonify({'message': 'no POST handler binded'})
#
#
# def gen_md5(args, class_name):
#     args_copy = copy.deepcopy(args)
#     filters = args_copy.get('filters', {})
#     if filters is None:
#         filters = {}
#     from_ = args_copy.get('from')
#     to_ = args_copy.get('to')
#
#     if from_ and len(str(from_)) > 10:
#         args_copy['from'] = from_[0:10] + ' 00:00:00'
#
#     if to_ and len(str(to_)) > 10:
#         args_copy['to'] = to_[0:10] + ' 23:59:59'
#
#     publish_timestamp = filters.get('publish_timestamp', [])
#     for i in range(len(publish_timestamp)):
#         hms = ' 00:00:00' if i == 0 else ' 23:59:59'
#         publish_timestamp[i] = publish_timestamp[i][0:10] + hms
#
#     key = class_name + str(args_copy)
#     md5 = None
#     m = hashlib.md5()
#     try:
#         m.update(key.encode('utf-8'))
#         md5 = m.hexdigest()
#     except Exception as ex:
#         logger.error('error: %s' % ex, exc_info=True)
#
#     return md5


class BaseResource(Resource):
    # parser = None
    get_querier = None
    post_querier = None

    @classmethod
    def set_resource(cls, parser_args=None, get_querier=None, post_querier=None):
        cls.parser_args = parser_args
        cls.get_querier = get_querier
        cls.post_querier = post_querier

    def get(self):
        if self.__class__.get_querier:
            args = self.__class__.parser.parse_args()
            logger.debug('get args: %s' % str(args))

            try:
                query = self.__class__.get_querier.build_query(args)
                res = self.__class__.get_querier.build_result(query)
                logger.debug('args: %s' % str(args))
            except Exception as ex:
                logger.error('error: %s' % ex, exc_info=True)
                res = {'message': str(ex)}
            return jsonify(res)
        else:
            return jsonify({'message': 'no GET handler binded'})

    def post(self):
        if self.__class__.post_querier:
            args = parser.parse(self.__class__.parser_args, request)
            # request.headers.get('User-Agent')
            # test_args = json.dumps(args).decode('utf-8')
            header = str(request.headers)
            logger.debug('post args: %s' % str(args))
            logger.debug('header args: %s' % header)
            try:
                query = self.__class__.post_querier.build_query(args)
                res = self.__class__.post_querier.build_result(query)
                logger.debug('args: %s' % str(args))
            except Exception as ex:
                logger.error('error: %s' % ex, exc_info=True)
                res = {'message': str(ex)}
            return jsonify(res)
        else:
            return jsonify({'message': 'no POST handler binded'})
