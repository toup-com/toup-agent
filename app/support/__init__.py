"""Maintenance / support agent package.

Drives the issue lifecycle: intake → classify → triage (route via the
docs/skills source of truth) → diagnose → ADMIN APPROVAL → implement →
verify → PR. See ``docs/support-agent/README.md``.

This package intentionally does NO eager submodule imports: model code
(``app.db.models.support``) imports ``app.support.enums``, so importing
heavy submodules here would risk an import cycle. Import submodules
explicitly where needed (e.g. ``from app.support.pipeline import ...``).
"""
