"""Copy SQLite history to an empty PostgreSQL destination without rescoring."""
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,sqlite3
from psycopg import sql
from configuration import database_target
from database import connect,init_database,begin_write
from postgres_backend import is_postgres

ROOT=Path(__file__).parent
TABLES=['applications','predictions','staff_reviews']


def snapshot_source(source,backup_directory):
    source=Path(source).resolve()
    if not source.is_file():raise ValueError('SQLite source file does not exist.')
    directory=Path(backup_directory);directory.mkdir(parents=True,exist_ok=True)
    backup=directory/('before-postgres-migration-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.db')
    with closing(sqlite3.connect(source)) as current,closing(sqlite3.connect(backup)) as destination:
        current.backup(destination)
        if destination.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed.')
        if destination.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('SQLite has broken record references.')
    with closing(sqlite3.connect(backup)) as connection:
        connection.row_factory=sqlite3.Row
        rows={table:[dict(row) for row in connection.execute('SELECT * FROM '+table+' ORDER BY id')] for table in TABLES}
    return backup,rows


def migrate(source,target,backup_directory):
    if not is_postgres(target):raise ValueError('A PostgreSQL DATABASE_URL is required.')
    backup,rows=snapshot_source(source,backup_directory)
    init_database(target)
    with closing(connect(target)) as connection,connection:
        begin_write(connection)
        connection.execute('LOCK TABLE applications,predictions,staff_reviews IN ACCESS EXCLUSIVE MODE')
        existing={table:[dict(row) for row in connection.execute('SELECT * FROM '+table+' ORDER BY id')] for table in TABLES}
        if any(existing.values()):
            if existing==rows:
                return dict(status='already_migrated',backup=str(backup),counts={table:len(rows[table]) for table in TABLES})
            raise ValueError('Destination contains different records. No destination data was changed. Use an empty dedicated database or schema.')
        for table in TABLES:
            if not rows[table]:continue
            columns=list(rows[table][0])
            query=sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,columns)),sql.SQL(',').join(sql.Placeholder() for _ in columns))
            connection.executemany(query,[[row[key] for key in columns] for row in rows[table]])
        for table in TABLES:
            actual=[dict(row) for row in connection.execute('SELECT * FROM '+table+' ORDER BY id')]
            if actual!=rows[table]:raise ValueError('Migration verification failed; copied rows were rolled back.')
            query=sql.SQL("SELECT setval(pg_get_serial_sequence(%s,'id'),COALESCE(MAX(id),1),COUNT(*)>0) FROM {}").format(sql.Identifier(table))
            connection.execute(query,(table,))
    return dict(status='migrated',backup=str(backup),counts={table:len(rows[table]) for table in TABLES})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/loanlens.db')
    args=parser.parse_args()
    try:
        result=migrate(args.source,database_target(),ROOT/'data/backups')
        print(json.dumps(result,indent=2))
        (ROOT/'tmp').mkdir(exist_ok=True)
        (ROOT/'tmp/postgres-migration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    except ValueError as error:
        # This module's own validation messages never include connection values.
        print(str(error));raise SystemExit(1)
    except Exception as error:
        print(f'Migration failed ({type(error).__name__}); destination inserts were rolled back. Credentials were not printed.')
        raise SystemExit(1)
