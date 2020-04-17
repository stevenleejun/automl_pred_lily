# coding = utf-8
from .utils_tools_database import run_sql
import os


## 设计原则 =====
# 只在数据录入时做数据清洗
# 其他都是分层次做整合
## ==========
def run_standard_base_dir(
        host,
        port,
        user,
        password,
        database,
        sub_dir,
        base_dir,
        brand_nick='',
        create_sql_file=0
):
    sql_files_dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(sql_files_dir):
        sql_files_dir = os.path.join(base_dir, 'standard')

    for file in sorted(os.listdir(sql_files_dir)):
        wholedir_file = os.path.join(sql_files_dir, file)
        with open(wholedir_file, encoding='utf-8') as exe_sql_file:
            exe_sql = exe_sql_file.read().format(
                brand_nick=brand_nick
            )
        run_sql(host, user, password, database, port,
                exe_sql,
                database_type='greenplum'
                )

        # 生成实际运行文件
        if create_sql_file:
            brand_wholedir_file = os.path.join(sql_files_dir, file)
            if not os.path.exists(sql_files_dir):
                os.mkdir(sql_files_dir)
            with open(brand_wholedir_file, 'w', encoding='utf-8') as write_file:
                write_file.write(exe_sql)
