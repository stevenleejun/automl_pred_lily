from sys import stdout
import pandas as pd
import sys
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from utils.utils_time import get_date_relativedelta


class LabelExtraction():
    """Automatically makes labels for prediction problems."""
    def __init__(
            self,
            customer_entity,
            behavior_entity,
            product_entity=None,

            labeling_function=None,
            problem_type=None,
            label_type=None,
            actions_list=[],
            target_para={},

            cutoff_day=1,
            cutoff_num=1,
            cutoff_last_date=None,
            gap_window_unit='months',
            gap_window=1,
            lead_time_window_unit='months',
            lead_time_window=0,
            prediction_window_unit='months',
            prediction_window=1,
            sample_filtering_window_unit='months',
            sample_filtering_window=12,
            product_label_column_name_sep='@',
            customer_entity_index=None,
            behavior_entity_time_index=None,
            default_time_col_name='time'
    ):
        """Creates an instance of label_extraction maker.
        """
        self.customer_entity = customer_entity
        self.behavior_entity = behavior_entity
        self.product_entity = product_entity

        self.labeling_function = labeling_function
        self.problem_type = problem_type
        self.actions_list = actions_list
        self.target_para = target_para
        self.label_type = label_type

        self.cutoff_day = cutoff_day
        self.cutoff_num = cutoff_num
        self.cutoff_last_date = cutoff_last_date
        self.gap_window_unit = gap_window_unit
        self.gap_window = gap_window
        self.lead_time_window_unit = lead_time_window_unit
        self.lead_time_window = lead_time_window
        self.prediction_window_unit = prediction_window_unit
        self.prediction_window = prediction_window
        self.sample_filtering_window_unit = sample_filtering_window_unit
        self.sample_filtering_window = sample_filtering_window
        self.product_label_column_name_sep = product_label_column_name_sep

        self.customer_entity_index = customer_entity_index
        self.behavior_entity_time_index = behavior_entity_time_index
        self.default_time_col_name = default_time_col_name

        self.label_name = self.labeling_function.__name__

        if self.cutoff_num is None:
            self.cutoff_num = 1
        if self.cutoff_last_date:
            self.cutoff_last_date = pd.Timestamp(self.cutoff_last_date)


    @property
    def product_column(self):
        return self.target_para.get('product_column', None)

    @property
    def get_customer_entity_index(self):
        return self.customer_entity_index

    def get_product_label_column_name_dict(self):
        product_label_column_name_dict = {}
        product_list = self.target_para.get('product_list', None)
        if product_list:
            for product in product_list:
                product_label_column_name_dict[product] = self._get_comb_labeling_colname(product)
        else:
            product_label_column_name_dict[None] = self.labeling_function.__name__

        return product_label_column_name_dict

    def transform(
            self,
            entity_set,
            is_retrain=False,
            is_predict=False,
            verbose=True,
            logger=None,
            cutoff_last_date=None
    ):
        """
        """
        logger.info('transform begin')
        if cutoff_last_date: # cutoff_last_date 允许predict重置
            orig_cutoff_last_date = cutoff_last_date
        elif is_predict:
            orig_cutoff_last_date = None
        else:
            orig_cutoff_last_date = self.cutoff_last_date

        if orig_cutoff_last_date is not None:
            orig_cutoff_last_date = pd.Timestamp(orig_cutoff_last_date)

        # 获取 behavior_df
        data_df = entity_set[self.behavior_entity].df
        if data_df.shape[0] == 0:
            logger.error('self.behavior_entity:{} shape[0] == 0'.format(self.behavior_entity))
            return pd.DataFrame()

        # 如果是商品购买预测，则加入商品实体维表
        if self.problem_type == 'product_purchase':
            product_entity_df = entity_set.entity_dict[self.product_entity].df
            product_entity_index = entity_set.entity_dict[self.product_entity].index
            data_df = data_df.merge(product_entity_df, on=[product_entity_index], suffixes=('', '_product_entity'))
            logger.debug('{} unique为: {}'.format(product_entity_index, data_df[self.target_para['product_column']].value_counts()))
            if data_df.shape[0] == 0:
                logger.error('self.behavior_entity:{} merge self.product_entity:{} shape[0] == 0'.format(self.behavior_entity, self.product_entity))
                return pd.DataFrame()

        # 如果是有action_entity, 则处理特殊逻辑 TODO special
        if self.behavior_entity == 'sample_filtering_entity_repair_order':
            target_entity_df = entity_set.entity_dict['target_entity_increase_order'].df

            data_df['is__action__entity'] = 0
            target_entity_df['is__action__entity'] = 1
            data_df = data_df[[self.customer_entity_index, self.behavior_entity_time_index, 'is__action__entity']]
            target_entity_df = target_entity_df[[self.customer_entity_index, self.behavior_entity_time_index, 'is__action__entity']]
            data_df = data_df.append(target_entity_df)

            self.actions_list.append(
                {
                    'column': 'is__action__entity',
                    'operator': '==',
                    'value': 1
                }
            )

        if is_predict:
            cutoff_num = 1
        else:
            cutoff_num = self.cutoff_num

        # cutoff_last_date
        data_max_date = data_df[self.behavior_entity_time_index].max()
        cutoff_last_date = self.get_cutoff_last_date(
            data_max_date,
            is_predict=is_predict,
            is_retrain=is_retrain,
            orig_cutoff_last_date=orig_cutoff_last_date,
            cutoff_day=self.cutoff_day,
            lead_time_window_unit=self.lead_time_window_unit,
            lead_time_window=self.lead_time_window,
            prediction_window_unit=self.prediction_window_unit,
            prediction_window=self.prediction_window,
        )

        # 指定 progress_bar
        bar_format = "Elapsed: {elapsed} | Remaining: {remaining} | "
        bar_format += "Progress: {l_bar}{bar}| "
        bar_format += self.customer_entity_index + ": {n}/{total} "
        total = len(data_df.groupby(self.customer_entity_index))
        total *= cutoff_num
        progress_bar_file = open(logger.handlers[0].baseFilename, mode='a')
        progress_bar = tqdm(file=progress_bar_file, total=total, bar_format=bar_format, disable=not verbose)

        # 获取 分片
        slices = self.get_df_slices(
            data_df=data_df,
            cutoff_num=cutoff_num,
            cutoff_last_date=cutoff_last_date,
            customer_entity_index=self.customer_entity_index,
            behavior_entity_time_index=self.behavior_entity_time_index,
            gap_window_unit=self.gap_window_unit,
            gap_window=self.gap_window,
            sample_filtering_window_unit=self.sample_filtering_window_unit,
            sample_filtering_window=self.sample_filtering_window,
            lead_time_window_unit=self.lead_time_window_unit,
            lead_time_window=self.lead_time_window,
            prediction_window_unit=self.prediction_window_unit,
            prediction_window=self.prediction_window,
        )

        # 一个个打标签
        labeling_name = self.labeling_function.__name__
        labels, instance = [], 0

        for slice_df in slices:
            # 更新 labels
            labelre = self.labeling_function(
                df_label_object=slice_df,
                actions_list=self.actions_list,
                **self.target_para)
            if labelre is not None:
                label = {
                        self.customer_entity_index: slice_df.context.customer_id,
                        self.default_time_col_name: slice_df.context.cutoff_time
                }
                if isinstance(labelre, dict):
                    for product, lable_value in labelre.items():
                        comb_labeling_name = self._get_comb_labeling_colname(product)
                        label[comb_labeling_name] = lable_value
                else:
                    label[labeling_name] = labelre
                labels.append(label)

            # 更新 progress_bar
            if slice_df.context.slice_number == 1:
                progress_bar.update(n=cutoff_num)

        # 更新 progress_bar
        total -= progress_bar.n
        progress_bar.update(n=total)
        progress_bar.close()
        progress_bar_file.close()

        # 美化log文件
        sys.stdout.write('')
        sys.stdout.flush()

        # 更新 labels
        if len(labels)>0:
            labels = pd.DataFrame(labels)
            labels[self.default_time_col_name] = labels[self.default_time_col_name].dt.strftime('%Y-%m-%d')
        else:
            return pd.DataFrame()

        return labels

    def set_index(self, df):
        """Sets the time index in a data frame (if not already set).

        Args:
            df (DataFrame) : Data frame to set time index in.

        Returns:
            DataFrame : Data frame with time index set.
        """
        if df.index.name != self.behavior_entity_time_index:
            df = df.set_index(self.behavior_entity_time_index)

        # if self.default_time_col_name not in str(df.index.dtype):
        df.index = df.index.astype('datetime64[ns]')

        return df

    def filter_df_by_actions_list(
            self,
            data_df,
            actions_list,
    ):
        return data_df

    def get_cutoff_last_date(
            self,
            data_max_date,
            is_predict,
            is_retrain,
            orig_cutoff_last_date,
            cutoff_day,
            lead_time_window_unit,
            lead_time_window,
            prediction_window_unit,
            prediction_window,
    ):
        """
        如果是 预测问题 则取数据中最大的观察点
        或者 重训练 则取数据中 可训练的最大观察点
        如果设置的最大观察点 + lead_time_window + prediction_window > data_max_date，则取数据中可训练的最大观察点
        如果cutoff_last_date为空, 则取数据中可训练的最大观察点
        """
        prediction_window_begin = get_date_relativedelta(
            now_date=data_max_date,
            window_unit=prediction_window_unit,
            window=prediction_window,
            operate='-'
        )
        lead_time_window_begin = get_date_relativedelta(
            now_date=prediction_window_begin,
            window_unit=lead_time_window_unit,
            window=lead_time_window,
            operate='-'
        )

        if is_predict:
            if orig_cutoff_last_date is None:
                tmp_cutoff_last_date = data_max_date
            elif orig_cutoff_last_date > data_max_date:
                tmp_cutoff_last_date = data_max_date
            else:
                tmp_cutoff_last_date = orig_cutoff_last_date
        elif is_retrain:
            tmp_cutoff_last_date = lead_time_window_begin
        elif (orig_cutoff_last_date is None) or (orig_cutoff_last_date > lead_time_window_begin):
            tmp_cutoff_last_date = lead_time_window_begin
        else:
            tmp_cutoff_last_date = orig_cutoff_last_date

        if tmp_cutoff_last_date.day < cutoff_day: # 最后一个观察日day比cutoff_day小，则观察日往前一个月
            cutoff_last_date = tmp_cutoff_last_date + relativedelta(months=-1, day=cutoff_day)
        else:
            cutoff_last_date = tmp_cutoff_last_date + relativedelta(day=cutoff_day) #day是指替换

        return cutoff_last_date

    def get_df_slices(
            self,
            data_df,
            customer_entity_index,
            behavior_entity_time_index,
            cutoff_num,
            cutoff_last_date,
            gap_window_unit,
            gap_window,
            sample_filtering_window_unit,
            sample_filtering_window,
            lead_time_window_unit,
            lead_time_window,
            prediction_window_unit,
            prediction_window,
            verbose=False
    ):
        """
        对一个df根据预测主体做分片:
            对每一个id:
                从最后一个cutoff_last_date往前每一个cutoff_num:
                    获取cutoff_time
                    1、获取sample_filtering_window的开始日期与结束日期(cutoff_time)
                        截取 df_sample_filtering_window
                    2、获取lead_time_window的开始日期(cutoff_time)与结束日期 lead_time_window_end
                        截取 df_lead_time_window
                    3、获取prediction_window的开始日期(lead_time_window_end)与结束日期 prediction_window_end
                        截取 df_prediction_window
        分片输出为Context对象:
            id
            cutoff_time
            df_sample_filtering_window
            df_lead_time_window
            df_prediction_window
        """

        # 排序, index, 时间
        data_df = data_df.sort_values(by=[customer_entity_index, behavior_entity_time_index])
        data_df = data_df.dropna(subset=[behavior_entity_time_index])
        data_df = self.set_index(data_df)

        for group in data_df.groupby(customer_entity_index):
            slices = self._single_instance_slices(
                group=group,
                customer_entity_index=customer_entity_index,
                behavior_entity_time_index=behavior_entity_time_index,
                cutoff_num=cutoff_num,
                cutoff_last_date=cutoff_last_date,
                gap_window_unit=gap_window_unit,
                gap_window=gap_window,
                sample_filtering_window_unit=sample_filtering_window_unit,
                sample_filtering_window=sample_filtering_window,
                lead_time_window_unit=lead_time_window_unit,
                lead_time_window=lead_time_window,
                prediction_window_unit=prediction_window_unit,
                prediction_window=prediction_window,
            )
            for data_df in slices:
                if verbose:
                    print(data_df)
                yield data_df
    
    def _single_instance_slices(
            self,
            group,
            customer_entity_index,
            behavior_entity_time_index,
            cutoff_num=1,
            cutoff_last_date=None,
            gap_window_unit='months',
            gap_window=1,
            sample_filtering_window_unit='months',
            sample_filtering_window=12,
            lead_time_window_unit='day',
            lead_time_window=None,
            prediction_window_unit='months',
            prediction_window=1,
    ):
        """
            分片输出为Context对象:
            id
            cutoff_time
            df_sample_filtering_window
            df_lead_time_window
            df_prediction_window
        """
        id, data_df = group

        assert data_df.index.is_monotonic_increasing, "Please sort your dataframe chronologically before calling search"
    
        if data_df.empty:
            return
    
        cutoff_time = cutoff_last_date
        cutoff_num_cnt = 0
        while cutoff_time > data_df.index[0] and cutoff_num_cnt < cutoff_num:
            df_sample_filtering_window, df_lead_time_window, df_prediction_window, prediction_window_end = self._single_cutoff_instance_df(
                data_df=data_df,
                cutoff_time=cutoff_time,
                sample_filtering_window_unit=sample_filtering_window_unit,
                sample_filtering_window=sample_filtering_window,
                lead_time_window_unit=lead_time_window_unit,
                lead_time_window=lead_time_window,
                prediction_window_unit=prediction_window_unit,
                prediction_window=prediction_window
            )

            df_slice = DataSlice()
            df_slice.context = Context(
                cutoff_time=cutoff_time,
                slice_number=cutoff_num_cnt,
                customer_entity_index=customer_entity_index,
                customer_id=id,
                prediction_window_end=prediction_window_end,
            )
            df_slice.df_sample_filtering_window = df_sample_filtering_window
            df_slice.df_lead_time_window = df_lead_time_window
            df_slice.df_prediction_window = df_prediction_window

            # 更新 循环条件
            cutoff_time = get_date_relativedelta(
                now_date=cutoff_time,
                window_unit=gap_window_unit,
                window=gap_window,
                operate='-'
            )
            cutoff_num_cnt += 1
    
            if df_sample_filtering_window.empty:
                continue

            yield df_slice
    
    def _single_cutoff_instance_df(
            self,
            data_df,
            cutoff_time,
            sample_filtering_window_unit='months',
            sample_filtering_window=0,
            lead_time_window_unit='months',
            lead_time_window=0,
            prediction_window_unit='months',
            prediction_window=0
    ):
        sample_filtering_window_begin = get_date_relativedelta(
                now_date=cutoff_time,
                window_unit=sample_filtering_window_unit,
                window=sample_filtering_window,
                operate='-'
        )
        lead_time_window_end = get_date_relativedelta(
                now_date=cutoff_time,
                window_unit=lead_time_window_unit,
                window=lead_time_window,
                operate='+'
        )
        prediction_window_end = get_date_relativedelta(
                now_date=lead_time_window_end,
                window_unit=prediction_window_unit,
                window=prediction_window,
                operate='+'
        )
        df_sample_filtering_window = data_df[(data_df.index >= sample_filtering_window_begin) & (data_df.index < cutoff_time)]
        df_lead_time_window = data_df[(data_df.index >= cutoff_time) & (data_df.index < lead_time_window_end)]
        df_prediction_window = data_df[(data_df.index >= lead_time_window_end) & (data_df.index < prediction_window_end)]
        return df_sample_filtering_window, df_lead_time_window, df_prediction_window, prediction_window_end

    def _get_comb_labeling_colname(self, product):
        return self.label_name + self.product_label_column_name_sep + str(product)

    def count_by_time(self, label_result_df):
        """Returns label_extraction count across cutoff times.
        cutoff_time
        """
        # self.df = self.df.reset_index(drop=False)
        if label_result_df.shape[0] == 0:
            return pd.DataFrame()
        if self.label_type == 'classifier':
            label_result_df.sort_values(by=self.default_time_col_name, inplace=True)
            value = label_result_df.pivot_table(index=self.default_time_col_name, columns=self.label_name, aggfunc='count', margins=True).fillna(0)
            value.columns = value.columns.get_level_values(1).values
            if 0 not in value.columns:
                value['1'] = value[1] / value['All']
                value['0'] = value['1'].apply(lambda r: 1-r)
            else:
                value['0'] = value[0] / value['All']
                value['1'] = value['0'].apply(lambda r: 1-r)
            value[['0', '1']] = value[['0', '1']].applymap(lambda x: format(x, '.4%'))
            # value['All'] = value['All'].apply(lambda x: format(x, ','))

            value.reset_index(inplace=True)
            value = value[[self.default_time_col_name,'All', '0', '1']].rename(columns={
                self.default_time_col_name: '观察点(Cutoff)',
                'All': '样本数',
                '1': 'label=1占比',
                '0': 'label=0占比',
            })

        else:
            value = label_result_df.pivot_table(index=self.default_time_col_name, values=self.label_name, aggfunc=['count', 'mean', 'median'], margins=True).fillna(0)
            value.columns = value.columns.get_level_values(0).values

            # value['count'] = value['count'].apply(lambda x: format(x, ','))
            value['mean'] = value['mean'].apply(lambda x: format(x, '.2f'))
            value['median'] = value['median'].apply(lambda x: format(x, '.2f'))

            value = value.rename(columns={
                'count': '样本数',
                'mean': '均值',
                'median': '中位数'
            })
            value.reset_index(inplace=True)

        return value

# ===== 本文件函数

class Context:
    """Metadata for data slice."""

    def __init__(self, cutoff_time=None, slice_number=None, customer_entity_index=None, customer_id=None, prediction_window_end=None):
        """Metadata for data slice.

        Args:
            cutoff_gap (tuple) : Start and stop time for cutoff_gap.
            window (tuple) : Start and stop time for window.
            slice (int) : Slice number.
            customer_entity_index (int) : Target entity.
            customer_id (int) : Target instance.
        """
        self.cutoff_time = cutoff_time or (None, None)
        self.slice_number = slice_number
        self.customer_entity_index = customer_entity_index
        self.customer_id = customer_id
        self.prediction_window_end = prediction_window_end




class DataSlice:
    """Data slice for labeling function."""
    # 一个类可以直接赋一个变量
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSlice

    def __str__(self):
        """Metadata of data slice."""
        info = {
            'slice_number': self.context.slice_number,
            self.context.customer_entity_index: self.context.customer_id,
            'window': '[{}, {})'.format(*self.context.sample_filtering_window),
            'cutoff_gap': '[{}, {})'.format(*self.context.cutoff_gap)
            # 'DataFrame': '[{})'.format(self.__name__)
            # 'DataFrame': '[{})'.format(self.T)
        }

        info = pd.Series(info).to_string()
        return info


