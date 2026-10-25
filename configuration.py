"""Environment-based configuration, with an explicit local SQLite fallback."""
import os,secrets
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).parent
load_dotenv(ROOT/'.env',override=False)


def database_target():
    url=os.environ.get('DATABASE_URL','').strip()
    if url:
        if not url.startswith(('postgresql://','postgres://')):
            raise ValueError('DATABASE_URL must contain the PostgreSQL connection URI only.')
        return url
    if os.environ.get('VERCEL'):
        raise RuntimeError('Set DATABASE_URL before deploying on Vercel.')
    return ROOT/'data/loanlens.db'


def signing_keys():
    secret=os.environ.get('SECRET_KEY','').strip()
    if not secret and os.environ.get('VERCEL'):
        raise RuntimeError('Set a stable SECRET_KEY before deploying on Vercel.')
    secret=secret or secrets.token_hex(32)
    return secret,os.environ.get('BATCH_RECEIPT_SECRET','').strip() or secret
