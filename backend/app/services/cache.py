import hashlib
import json

from upstash_redis import Redis

from app.core.config import settings

redis = Redis(
    url=settings.upstash_redis_rest_url,
    token=settings.upstash_redis_rest_token,
)


def _make_key(doc_id: str, question: str) -> str:
    h = hashlib.md5(f"{doc_id}:{question.lower().strip()}".encode()).hexdigest()
    return f"docusense:query:{h}"


def get_cached(doc_id: str, question: str):
    """Return cached result or None."""
    try:
        val = redis.get(_make_key(doc_id, question))
        return json.loads(val) if val else None
    except Exception:
        return None


def set_cached(doc_id: str, question: str, result: dict, ttl: int = 3600):
    """Cache result for ttl seconds (default 1 hour)."""
    try:
        redis.setex(_make_key(doc_id, question), ttl, json.dumps(result))
    except Exception:
        pass  # cache failure must never break the main flow
