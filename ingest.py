import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from src.ingestion.pipeline import IngestionPipeline

def ingest_file(file_path: str, org_id: str = "default-org-123", user_id: str = "user-456", acl_tags: list = None):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return False

    print(f"\n📄 Starting ingestion for: {path.name} ({path.resolve()})")
    print(f"🏢 Org ID: {org_id} | 👤 User ID: {user_id} | 🏷️  Tags: {acl_tags or ['INTERNAL']}")

    pipeline = IngestionPipeline()
    chunks = pipeline.process_document(
        file_path=str(path.resolve()),
        org_id=org_id,
        user_id=user_id,
        acl_tags=acl_tags or ["INTERNAL"]
    )

    if chunks:
        print(f"✅ Successfully parsed, embedded, and indexed {len(chunks)} chunks into Qdrant!")
        print(f"Preview of chunk 1: {chunks[0].content[:120]}...\n")
        return True
    else:
        print(f"⚠️  Ingestion completed, but no chunks were produced.")
        return False

def ingest_directory(dir_path: str, org_id: str = "default-org-123", user_id: str = "user-456"):
    folder = Path(dir_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Error: Directory not found: {dir_path}")
        return

    extensions = [".txt", ".md", ".pdf", ".docx", ".json", ".csv"]
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not files:
        print(f"⚠️ No matching documents found in {dir_path} with extensions {extensions}")
        return

    print(f"📂 Found {len(files)} document(s) in {dir_path} to ingest.")
    success_count = 0
    for file in files:
        if ingest_file(str(file), org_id, user_id):
            success_count += 1

    print(f"\n🎉 Finished batch ingestion: {success_count}/{len(files)} files indexed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into SentinelRAG vector store (Qdrant).")
    parser.add_argument("path", nargs="?", default="data", help="File or folder path to ingest (default: 'data' folder)")
    parser.add_argument("--org", default="default-org-123", help="Organization / tenant ID")
    parser.add_argument("--user", default="user-456", help="User ID")
    parser.add_argument("--tags", default="INTERNAL", help="Comma-separated ACL tags")

    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    target = Path(args.path)
    if target.is_dir():
        ingest_directory(str(target), org_id=args.org, user_id=args.user)
    elif target.is_file():
        ingest_file(str(target), org_id=args.org, user_id=args.user, acl_tags=tags)
    else:
        # If neither exists and default 'data' was used, create data folder with a sample doc
        if args.path == "data":
            target.mkdir(exist_ok=True)
            sample = target / "sample_security_policy.txt"
            if not sample.exists():
                sample.write_text(
                    "SentinelRAG Security Policy\n\n"
                    "1. Access Control: All users must authenticate via SSO with MFA enabled.\n"
                    "2. Data Protection: Customer data at rest must be encrypted using AES-256.\n"
                    "3. Incident Response: Security incidents must be reported to security@example.com within 1 hour.\n",
                    encoding="utf-8"
                )
                print(f"Created sample document: {sample}")
            ingest_directory(str(target), org_id=args.org, user_id=args.user)
        else:
            print(f"❌ Target path does not exist: {args.path}")
            sys.exit(1)
