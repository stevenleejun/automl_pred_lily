# coding = utf-8
import os
import shutil


def from_base_create_run_sql(brand_list, base_sql_dir='base_sql', run_dir='run_dir'):
    assert os.path.exists(base_sql_dir)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    os.mkdir(run_dir)

    for file in sorted(os.listdir(base_sql_dir)):
        table_name = os.path.splitext(file)[0]
        file_path = os.path.join(base_sql_dir, file)
        with open(file_path, encoding='utf-8') as exe_sql_file:
            base_exe_sql = exe_sql_file.read()

            is_first_run = 1
            for brand_nick in brand_list:
                exe_sql = base_exe_sql.format(brand_nick=brand_nick)

                if is_first_run:
                    insert_or_create = 'drop table if exists {table_name}; \ncreate table {table_name} as \n'.format(
                        table_name=table_name
                    )
                else:
                    insert_or_create = 'insert into {table_name} \n'.format(
                        table_name=table_name
                    )

                all_sql = insert_or_create + exe_sql

                re_save_file = table_name + '_' + str(1 - is_first_run) + '_' + brand_nick + '.sql'
                re_save_path = os.path.join(run_dir, re_save_file)
                with open(re_save_path, 'w', encoding='utf-8') as file:
                    file.write(all_sql)

                is_first_run = 0
