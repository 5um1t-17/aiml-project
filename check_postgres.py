"""Check the configured PostgreSQL connection without printing credentials."""
from contextlib import closing
from configuration import database_target
from database import connect
from postgres_backend import is_postgres


def check():
    target=database_target()
    if not is_postgres(target):raise ValueError('DATABASE_URL is not configured.')
    with closing(connect(target)) as connection,connection:
        connection.execute('SELECT 1').fetchone()
        print('PostgreSQL connection successful.')
        for table in ['applications','predictions','staff_reviews']:
            exists=connection.execute('SELECT to_regclass(?) AS name',(table,)).fetchone()['name']
            if exists:
                count=connection.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
                print(f'{table}: {count} records')
            else:print(f'{table}: not created yet (run migrate_postgres.py)')


if __name__=='__main__':
    try:check()
    except Exception as error:
        # Driver exceptions can include connection details: never echo them.
        print(f'PostgreSQL check failed ({type(error).__name__}). Check DATABASE_URL, its SSL settings and network access.')
        raise SystemExit(1)
