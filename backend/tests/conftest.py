"""
Test configuration: set required environment variables *before* any app
modules are imported so that pydantic-settings (Settings) and module-level
code never need real cloud credentials during pytest collection or execution.
"""

import os

# Provide safe default values for every required Settings field.
# Using setdefault means real env vars (set by CI secrets) take precedence.
os.environ.setdefault("GEMINI_API_KEY", "ci-fake-gemini-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://admin:admin123@localhost:5432/docusense"
)
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "fake-service-key")
os.environ.setdefault("SUPABASE_BUCKET", "docusense-files")
os.environ.setdefault("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
os.environ.setdefault("UPSTASH_REDIS_REST_TOKEN", "fake-token")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
