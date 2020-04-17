##-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils.utils_time import get_date_relativedelta
import logging
# ==== tools函数
def churn(
        df_label_object,
        churn_window_unit,
        churn_window,
        actions_list=[]
):
    """
    流失标签逻辑:
    前窗口: df_sample_filtering_window + df_lead_time_window
    后窗口: df_prediction_window
        1、获取观察窗口最后一次时间 last_date_in_before_window
        2、获取表现窗口首次时间 first_date_in_after_window
        3、由 last_date_in_before_window + churn_window 确定 occurs_date
        4.  if occurs_date > prediction_window_end:
                label_extraction = 0
            elif first_date_in_after_window < occurs_date:
                label_extraction = 0
            elif first_date_in_after_window is None: # occurs_date <= prediction_window_end
                label_extraction = 1
            elif first_date_in_after_window >= occurs_date:
                label_extraction = 1

    :param df_slice:
    :return: label_extraction
    """

    df_sample_filtering_window = df_label_object.df_sample_filtering_window
    df_lead_time_window = df_label_object.df_lead_time_window
    df_prediction_window = df_label_object.df_prediction_window
    prediction_window_end = df_label_object.context.prediction_window_end

    if df_sample_filtering_window.shape[0] == 0: # 没有在抽样窗口出现
        return None

    df_prediction_window = filter_df_by_actions_list(
        data_df=df_prediction_window,
        actions_list=actions_list
    )

    before_window = df_sample_filtering_window.append(df_lead_time_window)
    last_date_in_before_window = before_window.index.max()
    first_date_in_after_window = df_prediction_window.index.min()



    occurs_date = get_date_relativedelta(
            now_date=last_date_in_before_window,
            window_unit=churn_window_unit,
            window=churn_window,
            operate='+'
    )

    if occurs_date > prediction_window_end:
        label = 0
    elif first_date_in_after_window < occurs_date:
        label = 0
    elif pd.isnull(first_date_in_after_window):  # occurs_date <= prediction_window_end
        label = 1
    elif first_date_in_after_window >= occurs_date:
        label = 1
    else:
        label = None
        print('error: churn遇到未处理情况')

    return label


def purchase(
        df_label_object,
        actions_list=[]
):
    """
    """
    df_sample_filtering_window = df_label_object.df_sample_filtering_window
    df_prediction_window = df_label_object.df_prediction_window

    if df_sample_filtering_window.shape[0] == 0:  # 没有在抽样窗口出现
        return None

    df_prediction_window = filter_df_by_actions_list(
        data_df=df_prediction_window,
        actions_list=actions_list
    )

    if df_prediction_window.shape[0] > 0:
        label = 1
    else:
        label = 0

    return label


def product_purchase(
        df_label_object,
        product_column,
        product_list,
        actions_list=[]
):
    '''
    产品预测有多个标签，每个产品一个标签，
    客户如果购买的产品在观察窗口和预测窗口均出现则标签值为1，否则为0

    问题：标签值字段为label_产品名，是否对后面的程序产生影响。

    :param df_label_object:
    :param product_column:
    :param product_list:
    :return:
    '''
    df_sample_filtering_window = df_label_object.df_sample_filtering_window
    df_prediction_window = df_label_object.df_prediction_window

    if df_sample_filtering_window.shape[0] == 0: # 没有在抽样窗口出现
        return None

    df_prediction_window = filter_df_by_actions_list(
        data_df=df_prediction_window,
        actions_list=actions_list
    )

    # 全集
    labelre = dict.fromkeys(product_list, 0)
    # 子集
    prediction_window_product_list = df_prediction_window[product_column].unique()
    inner_product_list = list(set(prediction_window_product_list) & set(product_list))
    label_inner = zip(inner_product_list, [1]*len(inner_product_list))
    # update
    labelre.update(label_inner)

    return labelre


def value(
        df_label_object,
        value_column,
        value_agg,
        actions_list=[]
):
    df_sample_filtering_window = df_label_object.df_sample_filtering_window
    df_prediction_window = df_label_object.df_prediction_window

    if df_sample_filtering_window.shape[0] == 0: # 没有在抽样窗口出现
        return None

    df_prediction_window = filter_df_by_actions_list(
        data_df=df_prediction_window,
        actions_list=actions_list
    )

    if df_prediction_window[value_column].count() > 0:
        if value_agg == 'mean':
            label = df_prediction_window[value_column].sum(skipna=True)/df_prediction_window[value_column].count()
        elif value_agg == 'nunique':
            label = df_prediction_window[value_column].count()
        else:
            label = df_prediction_window[value_column].agg(value_agg)
    else:
        label = 0.0

    return label


def highvalue():
    pass


def filter_df_by_actions_list(
        data_df,
        actions_list=[],
):
    for action_dic in actions_list:
        column = action_dic['column']
        operator = action_dic['operator']
        value = action_dic['value']
        if column in data_df.columns:
            command = "data_df['{column}'] {operator} {value}".format(
                column=column,
                operator=operator,
                value=value
            )
            filter = eval(command)
            data_df = data_df[filter]

    return data_df


#
# import pandas as pd
# import numpy as np
# df = pd.DataFrame(data=[['data1', 2, np.nan], ['data2', 3, 4], ['data3', 4, 4]], index=[1, 2, 3], columns=['a', 'b', 'c'])
#


# data_df = pd.DataFrame({'t': [1,3,6,8,0]})
# actions_list = [{
#     'column': 't',
#     'operator': '==',
#     'value':6
# }]
# filter_df_by_actions_list(
#         data_df,
#         actions_list)