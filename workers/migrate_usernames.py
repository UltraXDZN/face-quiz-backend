"""One-shot migration so every user doc satisfies:
  • Firestore doc ID  == `email`
  • `username` field  == `email.split("@")[0]`

For each user:
  1. expected_doc_id  = email
  2. expected_username = email.split("@")[0]
  3. If doc ID is already the email: only update `username` if it's wrong.
  4. Otherwise: write a new doc at `users/{email}` with the fixed
     `username` field, then delete the old doc.
  5. Skip if the target doc ID is already taken by a different user
     (duplicate-registration case — listed for manual review).

Users with no email field, no "@" in their email, or an empty email
local part are skipped with a warning.

NOTE: If you run this with --execute, the backend code that currently
writes new users with `db.collection("users").document(user.username)`
must also be updated to use `user.email` as the doc ID. Otherwise the
next signup will create a doc keyed by username again, re-introducing
the inconsistency this script fixes.

Run:
    python workers/migrate_usernames.py             # dry run
    python workers/migrate_usernames.py --execute   # actually migrate
    python workers/migrate_usernames.py --prod      # force prod Firestore
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from google.auth.credentials import AnonymousCredentials  # noqa: E402
from google.cloud import firestore  # noqa: E402


def _get_db(force_prod: bool = False):
    env = os.getenv("ENVIRONMENT", "")
    emu_host = os.getenv("FIRESTORE_EMULATOR_HOST", "")

    # When forcing prod, the Firestore SDKs auto-use FIRESTORE_EMULATOR_HOST if set.
    # Clear it so we actually hit real Firestore.
    if force_prod and emu_host:
        print(f"[init] --prod set: clearing FIRESTORE_EMULATOR_HOST (was {emu_host!r})")
        os.environ.pop("FIRESTORE_EMULATOR_HOST", None)
        emu_host = ""

    use_emulator = (env != "production") or bool(emu_host and not force_prod)

    if use_emulator:
        host = emu_host or "127.0.0.1:8080"
        project = os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test")
        print(f"[init] Connecting to EMULATOR at {host} (project={project})")
        return firestore.Client(project=project, credentials=AnonymousCredentials())

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    print(f"[init] Connecting to PRODUCTION Firestore via {cred_path}")
    import firebase_admin
    from firebase_admin import credentials, firestore as admin_firestore
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    return admin_firestore.client()


def _classify(docs):
    """Pass 1: bucket every doc into a category. No writes, no side effects.

    Target state: doc ID == email, username == email.split("@")[0].

    Collisions are detected two ways:
      1. Another doc currently lives at the target ID.
      2. Multiple source docs want the same target ID (duplicate registrations).

    Returns four collections:
      - already_clean (list[str]):    doc IDs needing no work
      - bad_email     (list[dict]):   docs with missing/invalid email
      - to_migrate    (list[dict]):   target ID is uniquely owned, will succeed
      - collisions    (dict[str, list[dict]]):
            grouped by target_id → every doc that wants that target_id
    """
    from collections import defaultdict

    snapshot = {d.id: (d, d.to_dict() or {}) for d in docs}
    bad_email = []
    plans_by_target: dict[str, list] = defaultdict(list)

    for doc_id, (doc, data) in snapshot.items():
        username = data.get("username", "")
        email = data.get("email", "")

        if not email or "@" not in email or not email.split("@")[0]:
            bad_email.append({"doc_id": doc_id, "data": data, "reason": f"email={email!r}"})
            continue

        expected_doc_id = email
        expected_username = email.split("@")[0]

        plans_by_target[expected_doc_id].append({
            "doc_id": doc_id,
            "data": data,
            "expected_doc_id": expected_doc_id,
            "expected_username": expected_username,
            "old_username": username,
            "email": email,
            "username_wrong": username != expected_username,
            "doc_id_wrong": doc_id != expected_doc_id,
            "ref": doc.reference,
        })

    already_clean, to_migrate, collisions = [], [], {}

    for target_id, group in plans_by_target.items():
        if len(group) == 1:
            p = group[0]
            if not p["username_wrong"] and not p["doc_id_wrong"]:
                already_clean.append(p["doc_id"])
            else:
                to_migrate.append(p)
        else:
            # Multiple docs want this target — duplicate registration, skip all
            collisions[target_id] = group

    return already_clean, bad_email, to_migrate, collisions


def _hr(title: str, width: int = 80):
    print()
    print("─" * width)
    print(title)
    print("─" * width)


def _print_report(already_clean, bad_email, to_migrate, collisions, total: int):
    colliding_docs = sum(len(g) for g in collisions.values())

    print()
    print("=" * 80)
    print("USERNAME MIGRATION REPORT")
    print("=" * 80)
    print(f"  Scanned:           {total}")
    print(f"  Already clean:     {len(already_clean)}")
    print(f"  Will migrate:      {len(to_migrate)}")
    print(f"  Collisions:        {len(collisions)} group(s) / {colliding_docs} doc(s)")
    print(f"  Missing email:     {len(bad_email)}")

    if to_migrate:
        _hr(f"[1] WILL MIGRATE ({len(to_migrate)})")
        for p in to_migrate:
            changes = []
            if p["doc_id_wrong"]:
                changes.append(f"docId {p['doc_id']!r} → {p['expected_doc_id']!r}")
            if p["username_wrong"]:
                changes.append(f"username {p['old_username']!r} → {p['expected_username']!r}")
            print(f"  • {p['email']}")
            for c in changes:
                print(f"      {c}")

    if collisions:
        _hr(f"[2] COLLISIONS — MANUAL REVIEW ({len(collisions)} group(s))")
        print("  Multiple docs share the same email (= target doc ID).")
        print("  Use --merge to auto-merge each group into the email-keyed doc.\n")
        for target_id, group in collisions.items():
            print(f"  ⚠ target id {target_id!r} — {len(group)} docs claim it:")
            for p in group:
                d = p["data"]
                marker = "  ★ at target" if not p["doc_id_wrong"] else ""
                exams = len(d.get("acessedExams") or [])
                tags = len(d.get("tags") or [])
                bits = [
                    f"exams={exams}",
                    f"tags={tags}",
                    f"admin={bool(d.get('admin'))}",
                    f"gid={'Y' if d.get('google_uid') else 'N'}",
                    f"jmbag={d.get('jmbag', '')!r}",
                ]
                print(f"      doc={p['doc_id']!r:40}  username={p['old_username']!r:20}  [{', '.join(bits)}]{marker}")
            print()

    if bad_email:
        _hr(f"[3] MISSING / INVALID EMAIL ({len(bad_email)})")
        for p in bad_email:
            print(f"  ⚠ doc={p['doc_id']!r}: {p['reason']}")


def _best_exam(a: dict, b: dict) -> dict:
    """Pick the more 'progressed' of two AccessedExam entries for the same exam id."""
    ap, bp = a.get("pointsEarned", 0) or 0, b.get("pointsEarned", 0) or 0
    if ap != bp:
        return a if ap > bp else b
    af, bf = a.get("lastFinished", 0) or 0, b.get("lastFinished", 0) or 0
    if af != bf:
        return a if af > bf else b
    aa, ba = a.get("lastAccessed", 0) or 0, b.get("lastAccessed", 0) or 0
    return a if aa >= ba else b


def _merge_group(group: list, target_id: str) -> dict:
    """Merge N duplicate user docs into a single dict ready to write at target_id."""
    # Process in order: doc already at target first, then by doc_id for stability
    ordered = sorted(group, key=lambda p: (p["doc_id_wrong"], p["doc_id"]))
    primary = ordered[0]["data"]

    merged = dict(primary)  # start with primary's fields

    for p in ordered[1:]:
        other = p["data"]
        for key, val in other.items():
            if key in ("acessedExams", "tags", "admin"):
                continue  # handled below
            if merged.get(key) in (None, "", []) and val not in (None, "", []):
                merged[key] = val

    # acessedExams: union by exam id, keep best progress
    exam_by_id: dict[str, dict] = {}
    for p in ordered:
        for exam in (p["data"].get("acessedExams") or []):
            eid = exam.get("id")
            if not eid:
                continue
            if eid not in exam_by_id:
                exam_by_id[eid] = exam
            else:
                exam_by_id[eid] = _best_exam(exam_by_id[eid], exam)
    merged["acessedExams"] = list(exam_by_id.values())

    # tags: union
    merged["tags"] = sorted({t for p in ordered for t in (p["data"].get("tags") or [])})

    # admin: any True wins
    merged["admin"] = any(bool(p["data"].get("admin")) for p in ordered)

    # Enforce rule: username == email local part
    merged["username"] = target_id.split("@")[0]
    merged["email"] = target_id

    return merged


def _apply_merges(db, collisions: dict, execute: bool) -> int:
    """Merge each collision group into a single email-keyed doc; delete duplicates."""
    if not collisions:
        return 0
    _hr(f"[MERGE] Resolving {len(collisions)} collision group(s)")
    count = 0
    for target_id, group in collisions.items():
        merged = _merge_group(group, target_id)
        print(f"  ⚡ {target_id}  (merging {len(group)} docs)")
        print(f"      exams={len(merged['acessedExams'])}  tags={len(merged['tags'])}  admin={merged['admin']}  gid={'Y' if merged.get('google_uid') else 'N'}  username={merged['username']!r}")

        if not execute:
            continue

        target_ref = db.collection("users").document(target_id)
        target_ref.set(merged)
        for p in group:
            if p["doc_id"] != target_id:
                p["ref"].delete()
        count += 1
        print(f"      ✓ written to {target_id!r}, {len(group) - 1} duplicate(s) deleted")
    return count


def migrate(db, execute: bool, merge: bool = False) -> dict:
    print("[scan] Streaming users collection…")
    docs = list(db.collection("users").stream())
    total = len(docs)
    print(f"[scan] Found {total} user doc(s)")

    already_clean, bad_email, to_migrate, collisions = _classify(docs)
    _print_report(already_clean, bad_email, to_migrate, collisions, total)

    colliding_docs = sum(len(g) for g in collisions.values())
    stats = {
        "scanned": total,
        "already_clean": len(already_clean),
        "needs_migration": len(to_migrate) + colliding_docs,
        "migrated": 0,
        "skipped_collision": colliding_docs,
        "skipped_no_email": len(bad_email),
    }

    if to_migrate and execute:
        _hr(f"[EXECUTE] Applying {len(to_migrate)} migration(s)")
        for p in to_migrate:
            new_data = {**p["data"], "username": p["expected_username"]}
            if p["doc_id_wrong"]:
                db.collection("users").document(p["expected_doc_id"]).set(new_data)
                p["ref"].delete()
            else:
                p["ref"].update({"username": p["expected_username"]})
            stats["migrated"] += 1
            print(f"  ✓ {p['doc_id']!r} → {p['expected_doc_id']!r}")

    if merge:
        merged_count = _apply_merges(db, collisions, execute)
        stats["merged_groups"] = merged_count
        if execute:
            # Each merge resolves a group, so those docs no longer count as skipped
            stats["skipped_collision"] -= sum(len(g) for g in collisions.values())

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--prod", action="store_true", help="Force production Firestore (ignore FIRESTORE_EMULATOR_HOST)")
    parser.add_argument("--merge", action="store_true",
                        help="Also merge collision groups into the email-keyed doc (union acessedExams/tags, etc.)")
    args = parser.parse_args()

    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}{' + MERGE' if args.merge else ''}")
    db = _get_db(force_prod=args.prod)
    stats = migrate(db, args.execute, merge=args.merge)

    if not args.execute and stats["needs_migration"] > 0:
        print()
        print("=" * 80)
        print(f"  DRY RUN — no changes made. Re-run with --execute to apply.")
        if stats["skipped_collision"] and not args.merge:
            print(f"  Note: {stats['skipped_collision']} collision(s) will still be skipped — pass --merge to resolve.")
        print("=" * 80)


if __name__ == "__main__":
    main()
