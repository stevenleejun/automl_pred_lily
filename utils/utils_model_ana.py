# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:18:28 2018
@author: shuyun
"""
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import re
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import kde
import seaborn as sns
from matplotlib.ticker import NullFormatter
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'


def model_profit_report_classifier(y_true, y_probas):
    result_df = pd.DataFrame({'y_probas': y_probas, 'y_true': y_true})
    report_list = []
    threshold_list = ((np.linspace(9, 0, 10)) * 10).tolist()
    for i in threshold_list:
        thres_value = np.percentile(result_df['y_probas'], i)

        result_df['y_pred'] = 0
        result_df.loc[result_df['y_probas'] > thres_value, 'y_pred'] = 1
        tn, fp, fn, tp = confusion_matrix(result_df['y_true'], result_df['y_pred']).ravel()

        # 统计各个指标
        true_precision = (tp + fn) / (tn + fp + fn + tp)
        Decile = (int(100 - i))
        cul_positive = tp
        cul_negative = fp
        cul_total = cul_positive + cul_negative
        Recall = tp / (tp + fn)
        Precision = tp / (tp + fp)
        Lift = Precision / true_precision

        Decile = str(Decile) + '%'
        Recall = format(Recall, '.2%')
        Precision = format(Precision, '.2%')
        Lift = format(Lift, '.2%')

        report_list.append([Decile, cul_total, cul_positive, cul_negative, Precision, Recall, Lift])
    model_profit_report_classifier_df = pd.DataFrame(report_list,
                                           columns=["Decile", "cul_total", "cul_positive", "cul_negative", "Precision",
                                                    "Recall", "Lift"])
    return model_profit_report_classifier_df


def plot_feature_importance(
        feature_importance,
        feature_top_num=10,
        type='PredictionValuesChange'
):
    feature_importance = feature_importance.sort_values(by='feature_importance', ascending=True).head(feature_top_num)
    feature_importance.reset_index(inplace=True, drop=True)
    plt.show()

    rcParams.update({'figure.autolayout': True})
    ax = feature_importance.plot('feature_name', 'feature_importance', kind='barh', legend=False, color='c')
    ax.set_title("Feature Importance using {}".format(type), fontsize=14)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    # plt.show()
    # plt.tight_layout()
    # plt.savefig('test.png', bbox_inches="tight")
    # plt.show()
    # plt.savefig(feature_importance_path)
    return feature_importance




def plot_abs_error_recall (y_true, y_pred, errors_show_list=[1000], plot=True):
    precisions = []
    total_count = y_true.shape[0]
    for err in errors_show_list:
        precisions.append(np.sum(np.abs(y_true - y_pred) <= err) / total_count)
    res = pd.DataFrame({'error': errors_show_list, 'precision': precisions}, columns=['error', 'precision'])
    if plot:
        plt.plot(res.error, res.precision, '-')
        plt.title('Error vs Precision')
        plt.xlabel('Error')
        plt.ylabel('Precision')
        # plt.show()
    return res


def plot_abs_percent_error_recall (y_true, y_pred, errors_show_list=np.arange(0, 2.1, 0.1), plot=True):
    precisions = []
    error_rate = (abs(y_true - y_pred)) / y_true

    total_count = error_rate.shape[0]
    for err in errors_show_list:
        precisions.append(np.sum(error_rate <= err) / total_count)
    res = pd.DataFrame({'error_rate': errors_show_list, 'precision': precisions}, columns=['error_rate', 'precision'])
    if plot:
        plt.show()
        plt.plot(res.error_rate, res.precision, '-')
        plt.title('Error_rate vs Precision')
        plt.xlabel('Error')
        plt.ylabel('Precision')

    return res


def plot_errorsCDF(error):
    qs = np.linspace(0, 100, 101)
    es = np.percentile(error, q=qs)
    es_sp = np.percentile(error, q=[60, 70])
    plt.show()
    plt.plot(es, qs, '-')
    plt.plot(es_sp, [60, 70], 'o', color='r')
    plt.text(es_sp[0], 60, '60% -> {:.2f}'.format(es_sp[0]))
    plt.text(es_sp[1], 70, '70% -> {:.2f}'.format(es_sp[1]))
    plt.title('CDF of milein error')
    return pd.DataFrame({'percentile': qs, 'error': es}, columns=['percentile', 'error'])


def plot_pred_vs_true(y_true, y_pred):
    re_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    plt.subplots(1, 1)
    sns_plot = sns.regplot(y_true, y_pred)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title("true vs pred")  # You can comment this line out if you don't need title
    # plt.show(sns_plot)
    return re_df, sns_plot

def plot_cumulative_gains_regression(y_true, y_pred, title='Cumulative Gains Curve',
                                      ax=None, figsize=None, title_fontsize="large",
                                      text_fontsize="medium"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    percentages, gains = cumulative_gain_curve_regression(y_true, y_pred)

    plt.show()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains, lw=3, label='Class {}'.format(''))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    # 收益表
    qs = np.linspace(0, 100, 11)
    percentages_resize = np.percentile(percentages, q=qs)
    gains_resize = np.percentile(gains, q=qs)
    gains_reports = pd.DataFrame({'percentages': percentages_resize,
                                  'gains': gains_resize})
    return gains_reports


def model_profit_report_regression(y_true, y_pred):
    result_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
    result_df.sort_values(by='y_pred', ascending=False, inplace=True)
    result_df['y_true_cumsum'] = result_df['y_true'].cumsum()
    decile = np.linspace(1, 10, 10)*0.1
    total = result_df['y_true'].sum()

    cul_sum = result_df['y_true_cumsum'].quantile(q=decile, interpolation='higher')
    cul_num = cul_sum.apply(lambda r: (result_df['y_true_cumsum']<=r).sum())
    cul_avg = cul_sum/cul_num
    recall = cul_sum/total
    lift = recall/decile

    decile = pd.Series(decile,index=decile).apply(lambda x: format(x, '.0%'))

    cul_sum = cul_sum.apply(lambda x: format(x, ',.2f'))
    # cul_num = cul_num.apply(lambda x: format(x, ','))
    cul_avg = cul_avg.apply(lambda x: format(x, ',.2f'))
    recall = recall.apply(lambda x: format(x, '.2%'))
    lift = lift.apply(lambda x: format(x, '.2f'))

    model_profit_report_df = pd.DataFrame({
        "Decile": decile,
        "cul_num": cul_num,
        "cul_sum": cul_sum,
        "cul_avg": cul_avg,
        "Recall": recall,
        "Lift": lift})

    return model_profit_report_df


def cumulative_gain_curve_regression(y_true, y_pred, pos_label=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    sorted_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


