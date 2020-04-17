import os
import featuretools as ft
import logging

logging.basicConfig(level=logging.DEBUG)


class EntitySetBuild:
    _sup_ft_variable_types_vs_name = {
        'datetime': ft.variable_types.Datetime,
        'numeric': ft.variable_types.Numeric,
        'categorical': ft.variable_types.Categorical,
        'boolean': ft.variable_types.Boolean,
        'text': ft.variable_types.Text,
        'latlong': ft.variable_types.LatLong,
    }

    def __init__(
            self,
            entity_set_info=None,
            data_dfs_dic=None,
            default_index_name='tmp_index',
            logger=logging,
            is_save=0,
            replace_comma='#',
            auto_max_values=None,
            manual_interesting_values_info=None,
            is_drop_duplicates_by_index=True

    ):
        self.entity_set_info = entity_set_info
        self.data_dfs_dic = data_dfs_dic
        self.default_index_name = default_index_name
        self.logger = logger
        self.is_save = is_save
        self.replace_comma = replace_comma
        self.manual_interesting_values_info = manual_interesting_values_info
        self.auto_max_values = auto_max_values
        self.is_drop_duplicates_by_index=is_drop_duplicates_by_index

    def transform(self):
        entity_set = ft.EntitySet(id='entity_set_build')

        # 处理实体表 origin_entity
        origin_entitys_info = {entity_name: entity_name_info
                               for entity_name, entity_name_info in self.entity_set_info.items()
                               if entity_name_info.get('normalize_base_entity', None) is None}
        entity_set = self._build_origin_entity(
            entity_set=entity_set,
            origin_entitys_info=origin_entitys_info
        )

        # 处理虚拟表 normalize_entity
        normalize_entitys_info = {entity_name: entity_name_info
                                  for entity_name, entity_name_info in self.entity_set_info.items()
                                  if entity_name_info.get('normalize_base_entity', None) is not None}
        entity_set = self._build_normalize_entity(
            entity_set=entity_set,
            normalize_entitys_info=normalize_entitys_info
        )

        # 处理 Relationship
        child_entitys_info = {entity_name: entity_name_info
                              for entity_name, entity_name_info in self.entity_set_info.items()
                              if entity_name_info.get('parent_entity', None) is not None}
        entity_set = self._build_child_entity(
            entity_set=entity_set,
            child_entitys_info=child_entitys_info
        )

        # 处理 intresting value
        entity_set = self.build_interesting_value(entity_set)

        return entity_set

    def build_interesting_value(self, entity_set):
        if self.auto_max_values is not None:
            entity_set.add_interesting_values(max_values=self.auto_max_values, verbose=True)

        if self.manual_interesting_values_info is not None:
            for entity_name, entity_name_info in self.manual_interesting_values_info.items():
                columns_desc = entity_name_info['columns_desc']
                for tmp_desc in columns_desc:
                    column_name = tmp_desc['column_name']
                    column_values = tmp_desc['column_values']
                    assert column_name is not None, 'column_name:{} is not None'.format(column_name)
                    assert column_values is not None, 'column_values:{} is not None'.format(column_values)
                    entity_set[entity_name][column_name].interesting_values = column_values
        return entity_set

    def _build_origin_entity(self, entity_set, origin_entitys_info):
        for entity_name, entity_name_info in origin_entitys_info.items():
            self.logger.info('entity_name:{entity_name} begin'.format(entity_name=entity_name))

            index = entity_name_info.get('index', [])
            time_index = entity_name_info.get('time_index', None)
            parent_entitys = entity_name_info.get('parent_entity', [])

            index_list = []
            index_list.extend(index)
            for parent_entity in parent_entitys:
                join_column = parent_entity.get('join_column', None)
                if join_column is not None:
                    join_column = [join_column]
                    index_list.extend(join_column)
                else:
                    parent_entity_index = self.entity_set_info[parent_entity['entity_name']]['index']
                    index_list.extend(parent_entity_index)

            data_df = self.data_dfs_dic[entity_name_info['entity_name']]

            if self.is_save:
                if self.replace_comma:
                    data_df.replace(',', self.replace_comma, regex=True, inplace=True)
                tmp_dir = '../../tmp_data'
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                data_df.to_csv(os.path.join(tmp_dir, entity_name + '.csv'), index=False)

            for column in index_list:
                data_df[column] = data_df[column].astype('str')

            if len(index) == 0:
                make_index = True
                index = self.default_index_name
            else:
                make_index = False
                index = index[0]
                if self.is_drop_duplicates_by_index:
                    data_df.drop_duplicates(subset=index, keep='last', inplace=True)

            fields_ana_type = entity_name_info.get('field_list', [])
            ft_variable_types = {}
            for field_ana_dic in fields_ana_type:
                name = field_ana_dic['name']
                variable_type = field_ana_dic['variable_type']
                assert variable_type in self._sup_ft_variable_types_vs_name, \
                    'variable_type:{} not in {}'.format(variable_type, self._sup_ft_variable_types_vs_name)

                if (name in data_df.columns) and variable_type in self._sup_ft_variable_types_vs_name:
                    ft_variable_types[name] = self._sup_ft_variable_types_vs_name[variable_type]

            entity_set = entity_set.entity_from_dataframe(entity_id=entity_name,
                                                          dataframe=data_df,
                                                          make_index=make_index,
                                                          index=index,
                                                          variable_types=ft_variable_types,
                                                          time_index=time_index
                                                          )

        return entity_set

    def _build_normalize_entity(self, entity_set, normalize_entitys_info):
        for entity_name, entity_name_info in normalize_entitys_info.items():
            index = entity_name_info.get('index')
            normalize_base_entity = entity_name_info['normalize_base_entity']
            entity_set = entity_set.normalize_entity(base_entity_name=normalize_base_entity,
                                                     new_entity_name=entity_name,
                                                     index=index
                                                     )
        return entity_set

    def _build_child_entity(self, entity_set, child_entitys_info):
        for entity_name, entity_name_info in child_entitys_info.items():
            parent_entitys = entity_name_info['parent_entity']
            for parent_entity in parent_entitys:
                parent_entity_name = parent_entity['entity_name']
                join_column = parent_entity.get('join_column', None)
                parent_entity_index = self.entity_set_info[parent_entity_name]['index'][0]
                if join_column is None:
                    join_column = parent_entity_index
                entity_set.add_relationship(ft.Relationship(
                    entity_set[parent_entity_name][parent_entity_index],
                    entity_set[entity_name][join_column]))
        return entity_set
