import psycopg2
import pymysql
import sqlalchemy
import ibis
import pandas as pd
from ibis.impala.api import connect


def get_run_sql(table_name,
                sql=None, schema=None, modified_date=None, modified_date_colname='modified_date', nrows=None):
    if sql is None:
        if schema is not None:
            table_name = schema + '.' + table_name

        sql = 'select * from {table_name}'.format(table_name=table_name)

        if modified_date is not None:
            sql = sql + ' where {modified_date_colname} >= {modified_date}'.format(
                modified_date_colname=modified_date_colname,
                modified_date=modified_date
            )
    if nrows is not None:
        sql = sql + ' limit ' + str(nrows)

    return sql


def get_database_sqlalchemy_conn(
        database_type,
        host,
        port,
        database,
        user=None,
        password=None,
        **kwargs
):
    if database_type == 'greenplum':
        conn = sqlalchemy.create_engine('postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
            user, password, host, port, database))
    elif database_type == 'mysql':
        conn = sqlalchemy.create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(user, password, host, port, database))
    elif database_type == 'hive':
        hdfs_host = kwargs.get('hdfs_host', host)
        hdfs_port = kwargs.get('hdfs_port', 50070)
        auth_mechanism = kwargs.get('hive_auth_mechanism', 'PLAIN')

        hdfs_client = ibis.hdfs_connect(host=hdfs_host, port=hdfs_port)
        conn = connect(
            host,
            port,
            auth_mechanism=auth_mechanism,
            database=database,
            hdfs_client=hdfs_client,
            user=user,
            password=password
            )
    else:
        return None
    return conn


def run_database_sql(database_type, run_sql, conn):
    if database_type == 'hive':
        requ = conn.sql(run_sql)
        data_df = requ.execute(limit=None)
    elif database_type in ['mysql', 'greenplum']:
        data_df = pd.read_sql_query(run_sql, conn)
    else:
        data_df = None
    return data_df


def get_database_cur(
        database_type,
        user,
        password,
        host,
        port,
        database
):
    if database_type == 'greenplum':
        db_con = psycopg2.connect(host=host, user=user, password=password, database=database)
    elif database_type == 'mysql':
        db_con = pymysql.connect(host=host, user=user, password=password, database=database, port=port,
                                 charset="utf8")
    else:
        return None
    cur = db_con.cursor()
    return cur
