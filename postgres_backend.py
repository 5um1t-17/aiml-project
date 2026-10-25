"""PostgreSQL transport for the application's parameterized persistence queries."""
import re
import psycopg
from psycopg import sql


class Record(dict):
    """Preserve sqlite3.Row's named and positional access for shared queries."""
    def __getitem__(self,key):
        return tuple(self.values())[key] if isinstance(key,int) else super().__getitem__(key)


def record_factory(cursor):
    names=[column.name for column in cursor.description] if cursor.description else []
    return lambda values:Record(zip(names,values))


class PostgresConnection:
    def __init__(self,url,schema='public'):
        if not re.fullmatch(r'[a-z_][a-z0-9_]{0,62}',schema):
            raise ValueError('DATABASE_SCHEMA must be a simple lowercase PostgreSQL schema name.')
        self.raw=psycopg.connect(url,row_factory=record_factory,connect_timeout=10,prepare_threshold=None)
        try:
            self.raw.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(schema)))
            if self.raw.execute('SELECT current_schema() AS name').fetchone()['name']!=schema:
                raise ValueError('The configured PostgreSQL schema does not exist.')
            self.raw.execute("SET LOCAL statement_timeout = '20s'")
        except Exception:
            self.raw.close();raise

    def execute(self,query,params=None):
        # Only repository-owned SQL uses qmark placeholders. Values are always
        # bound separately; never interpolate user input or transform values.
        if isinstance(query,str):query=query.replace('?', '%s')
        return self.raw.execute(query,params)

    def executemany(self,query,rows):
        if isinstance(query,str):query=query.replace('?', '%s')
        with self.raw.cursor() as cursor:cursor.executemany(query,rows)

    def commit(self):self.raw.commit()
    def rollback(self):self.raw.rollback()
    def close(self):self.raw.close()
    def __enter__(self):return self
    def __exit__(self,kind,value,traceback):
        if kind is None:self.raw.commit()
        else:self.raw.rollback()


def is_postgres(target):
    return isinstance(target,str) and target.startswith(('postgresql://','postgres://'))
