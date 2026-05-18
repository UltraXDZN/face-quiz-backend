"""One-shot cleanup of stale `pending_analysis` data from the old worker.

The previous in-process worker (api/proctoring/worker.py, since removed)
wrote results to `exams/{exam_id}/pending_analysis/*` with a broken vision
model that classified every frame as "FACE NOT DETECTED". That data is
worthless; this script deletes it across all exams so the new worker can
re-analyze from S3 into the new `photoAnalysis` collection.

Run once before starting the new worker:
    python workers/cleanup_old_data.py             # dry run
    python workers/cleanup_old_data.py --execute   # actually delete
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

OLD_COLLECTION = "pending_analysis"
DELETE_BATCH = 200


def _get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    import firebase_admin
    from firebase_admin import credentials, firestore as admin_firestore
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    return admin_firestore.client()


def _delete_subcollection(db, exam_id: str, execute: bool) -> int:
    coll = db.collection("exams").document(exam_id).collection(OLD_COLLECTION)
    total = 0
    while True:
        docs = list(coll.limit(DELETE_BATCH).stream())
        if not docs:
            break
        total += len(docs)
        if execute:
            batch = db.batch()
            for d in docs:
                batch.delete(d.reference)
            batch.commit()
        else:
            # Dry-run: stop after the first page so we don't iterate forever.
            break
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete. Without this flag, only counts.")
    args = parser.parse_args()

    db = _get_db()
    exams = list(db.collection("exams").stream())
    print(f"Scanning {len(exams)} exam(s) for stale '{OLD_COLLECTION}' docs "
          f"({'EXECUTE' if args.execute else 'DRY RUN'})")

    grand_total = 0
    for exam in exams:
        n = _delete_subcollection(db, exam.id, args.execute)
        if n:
            print(f"  exams/{exam.id}/{OLD_COLLECTION}: {n}{' (first page only)' if not args.execute else ''}")
            grand_total += n

    print(f"Done. {'Deleted' if args.execute else 'Would delete at least'} {grand_total} document(s).")
    if not args.execute:
        print("Re-run with --execute to actually delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
