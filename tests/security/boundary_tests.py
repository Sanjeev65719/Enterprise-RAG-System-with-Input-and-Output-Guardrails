import unittest
from uuid import uuid4
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.indexing.store import SentinelVectorStore, VectorPayload
from src.ingestion.indexing.schema import DocumentMetadata
import os

class TestDataBoundaries(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-...")
        self.store = SentinelVectorStore()
        self.pipeline = IngestionPipeline(openai_api_key=self.api_key)

        self.org_a = str(uuid4())
        self.org_b = str(uuid4())
        self.user_a = str(uuid4())
        self.user_b = str(uuid4())

    def test_cross_tenant_isolation(self):
        """Verify that Org A cannot see Org B's data."""
        # 1. Ingest a document for Org B
        with open("org_b_secret.txt", "w") as f:
            f.write("This is a top secret document for Organization B.")

        self.pipeline.process_file(
            file_path="org_b_secret.txt",
            org_id=self.org_b,
            user_id=self.user_b,
            acl_tags=["TOP_SECRET"]
        )

        # 2. Attempt to search using Org A's credentials
        # We use a dummy vector for the search
        dummy_vector = [0.0] * 3072
        results = self.store.filtered_search(
            query_vector=dummy_vector,
            org_id=self.org_a,
            user_id=self.user_a,
            tags=["TOP_SECRET"]
        )

        self.assertEqual(len(results), 0, "Security Breach: Org A retrieved Org B's data!")

    def test_acl_tag_access(self):
        """Verify that users can only access data they have the tags for."""
        with open("acl_doc.txt", "w") as f:
            f.write("This document is for HR only.")

        self.pipeline.process_file(
            file_path="acl_doc.txt",
            org_id=self.org_a,
            user_id=self.user_a,
            acl_tags=["HR"]
        )

        # Search without HR tag
        dummy_vector = [0.0] * 3072
        results_no_tag = self.store.filtered_search(
            query_vector=dummy_vector,
            org_id=self.org_a,
            user_id=self.user_b,
            tags=["MARKETING"]
        )
        self.assertEqual(len(results_no_tag), 0, "Security Breach: User without HR tag accessed HR document!")

        # Search with HR tag
        results_with_tag = self.store.filtered_search(
            query_vector=dummy_vector,
            org_id=self.org_a,
            user_id=self.user_b,
            tags=["HR"]
        )
        self.assertGreater(len(results_with_tag), 0, "Functional Failure: User with HR tag could not access HR document!")

if __name__ == "__main__":
    unittest.main()
