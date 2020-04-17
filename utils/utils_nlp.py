# -*- coding: utf-8 -*-
import re
import jieba


def load_stop_words(file_name):
    with open(file_name, 'r', encoding='utf8') as file:
        return list(set([line.strip() for line in file]))


def get_words_list_str(record, stop_words=[]):
    # sklearn要求语料库中文章的分词之间以空格分隔
    # 匹配中文分词
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    numberPattern = re.compile(u'[0-9]+')
    words_list = jieba.lcut(record, cut_all=False, HMM=False)
    words_list_str = ",".join([
        term for term in words_list
        if zhPattern.search(term)
           and not numberPattern.search(term)
           and str(term) not in stop_words
                           ])
    return words_list_str


def get_words_list(record, stop_words=[]):
    # sklearn要求语料库中文章的分词之间以空格分隔
    # 匹配中文分词
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    numberPattern = re.compile(u'[0-9]+')
    words_list = jieba.lcut(record, cut_all=False, HMM=False)
    words_list = [
        term for term in words_list
        if zhPattern.search(term)
           and not numberPattern.search(term)
           and str(term) not in stop_words
                ]
    return words_list

def get_stop_words_list(file_name):
    with open(file_name,'r', encoding='utf8') as file:
        return list(set([line.strip() for line in file]))


