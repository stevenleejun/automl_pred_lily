from .utils_load_class import UtilsDataLoad
from .utils_save_class import UtilsDataSave
import psycopg2
import pymysql

def run_sql(host, user, password, database, port, exe_sql,
                            database_type
                            ):
    if database_type == 'greenplum':
        # conn = create_engine('postgresql+psycopg2://{}:{}@{}:{}/{}'.format(user, password, host, port, database))
        db_con = psycopg2.connect(host=host, user=user, password=password, database=database)
    if database_type == 'mysql':
        # conn = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(user, password, host, port, database))
        db_con = pymysql.connect(host=host, user=user, password=password, database=database, port=port, charset="utf8")
    cur = db_con.cursor()

    cur.execute(exe_sql)
    db_con.commit()
    db_con.close()


def run_sql_file(
        sql_path,
        host=None,
        user=None,
        password=None,
        database=None,
        port=None,
        database_type=None
):
    with open(sql_path, encoding='utf-8') as exe_sql_file:
        exe_sql = exe_sql_file.read()
    run_sql(
        host,
        user,
        password,
        database,
        port,
        exe_sql,
        database_type=database_type
    )


def transfer_database_table(from_database_info, to_database_info):
    table_name = from_database_info['table_name']
    database_type = from_database_info['database_type']
    host = from_database_info['host']
    port = from_database_info['port']
    user = from_database_info['user']
    password = from_database_info['password']
    database = from_database_info['database']

    utils_data_load = UtilsDataLoad(
        host=host,
        user=user,
        password=password,
        database=database,
        table_name=table_name,
        port=port,
        database_type=database_type,
    )
    data_df = utils_data_load.process()

    table_name = to_database_info['table_name']
    database_type = to_database_info['database_type']
    host = to_database_info['host']
    port = to_database_info['port']
    user = to_database_info['user']
    password = to_database_info['password']
    database = to_database_info['database']

    utils_data_save = UtilsDataSave(
        host=host,
        user=user,
        password=password,
        database=database,
        table_name=table_name,
        port=port,
        database_type=database_type,
    )
    utils_data_save.process(data_df=data_df)
