#-*- coding: utf-8 -*-

import jieba
import datetime
from datetime import date
from dateutil import parser
from utils.utils_load_class import UtilsDataLoad
from utils.utils_nlp import get_words_list_str
from utils.utils_nlp import get_stop_words_list
from utils.utils_save_class import UtilsDataSave

def comment_anlysis(logger, root_config, begin_date, end_date):
    # 配置解析
    stop_words_path = root_config['resource']['stop_words_file']
    dict_path = root_config['resource']['dict_file']

    host = root_config['greenplum']['host']
    port = root_config['greenplum']['port']
    user = root_config['greenplum']['user']
    password = root_config['greenplum']['password']
    database = root_config['greenplum']['database']
    extra_para = root_config['greenplum']['extra_para']


    # 资源读入
    stop_words_list = get_stop_words_list(stop_words_path)  # 1236个常用停用词
    jieba.load_userdict(dict_path)  # 加载自定义词库


    # 读入数据
    utils_data_load = UtilsDataLoad(
        host=host,
        database=database,
        port=port,
        user=user,
        password=password,
        database_type='greenplum',
    )

    all_sql = '''
            select a1.oid, a1.content,a1.created,order_item.dalei_name
            from ods.crm_tmall_dsr a1
            left join (select * from dm.t_cdp_f_order_item_0
                       where channel_I='线上' and paytime > '2019-12-31') order_item
            on a1.oid=order_item.oid
            left join (select oid
                 from dm.t_cdp_f_dsr_word
                 group by oid)re
            on a1.oid = re.oid
            where a1.content not in ('评价方未及时做出评价,系统默认好评!', '好评！', ' ')
                  and re.oid is null'''
        #         and a1.create::date >= {last_year_anaday}""".format(
        # last_year_anaday=last_year_anaday)

    source_data_df = utils_data_load.process(sql=all_sql)
    source_data_df = source_data_df.drop_duplicates(keep='first')
    source_data_df = source_data_df.sort_values(by='created', ascending=False)

    logger.info('apply(get_words_list)')
    source_data_df['corpus'] = source_data_df['content'].apply(lambda row: get_words_list_str(row, stop_words_list))
    source_data_df=source_data_df.drop('corpus', axis=1).join(source_data_df['corpus'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('corpus'))
    source_data_df=source_data_df[source_data_df.corpus.str.len()>1]

    utils_data_save = UtilsDataSave(
        table_name='t_cdp_f_dsr_word',
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        database_type='greenplum',
        extra_para=extra_para
    )

    utils_data_save.process(source_data_df)
    return True

