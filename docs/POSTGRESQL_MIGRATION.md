# PostgreSQL migration verification

LoanLens is configured through the private local `.env` file. The running website's `/health` endpoint confirms `database: postgresql`.

Migrated without rescoring or changing IDs:

- 4 applications
- 12 saved model predictions
- 2 explicitly simulated demo staff reviews

The migration verifies every copied field before committing. The SQLite source was retained and backed up at:

`C:\Users\sumit\Downloads\aiml project\data\backups\before-postgres-migration-20261025T161309494589Z.db`

Verification: 45 local tests passed and all 6 live PostgreSQL integration tests passed in an isolated disposable schema. Both prediction sources, history/search/filter/export, CSV uploads, atomic rollback, foreign keys, concurrent stale-review protection, migration idempotency, conflict refusal and identity sequences were exercised. Temporary test data was removed. The live history was checked again after restarting the server against PostgreSQL.

Connection credentials were not printed or committed. Set DATABASE_URL, SECRET_KEY and BATCH_RECEIPT_SECRET in Vercel's environment settings before deployment. This database migration does not itself publish or deploy the website; the remaining Vercel runtime/static-asset/bundle/access preparation is separate.
