"""Encrypt existing streaming_credentials.password rows with Fernet

Revision ID: 025
Revises: 024
Create Date: 2026-04-24

Vault CP2: column type is unchanged (Text); what changes is the semantics —
we now store Fernet ciphertext (base64, starts with 'gAAAAAB') instead of
raw plaintext. Any rows that already look encrypted are skipped (idempotent
in case the migration is re-run).

Downgrade decrypts rows back to plaintext.

Note: encryption runs in Python, the key NEVER enters SQL parameters.
"""
from alembic import op
import sqlalchemy as sa


revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Imported here so the migration doesn't fail at module-load time in
    # environments that don't have the settings populated yet (e.g. CI dry-run).
    from app.services.credential_crypto import encrypt_str, looks_encrypted

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, password FROM streaming_credentials")
    ).fetchall()

    encrypted = 0
    already = 0
    for row in rows:
        pw = row.password
        if pw is None:
            continue
        if looks_encrypted(pw):
            already += 1
            continue
        ct = encrypt_str(pw)
        conn.execute(
            sa.text("UPDATE streaming_credentials SET password=:p WHERE id=:i"),
            {"p": ct, "i": row.id},
        )
        encrypted += 1

    print(
        f"[025] streaming_credentials: encrypted={encrypted} already_encrypted={already}"
    )


def downgrade() -> None:
    from app.services.credential_crypto import decrypt_str, looks_encrypted

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, password FROM streaming_credentials")
    ).fetchall()

    decrypted = 0
    for row in rows:
        pw = row.password
        if pw is None or not looks_encrypted(pw):
            continue
        pt = decrypt_str(pw)
        conn.execute(
            sa.text("UPDATE streaming_credentials SET password=:p WHERE id=:i"),
            {"p": pt, "i": row.id},
        )
        decrypted += 1

    print(f"[025] downgrade: decrypted={decrypted}")
