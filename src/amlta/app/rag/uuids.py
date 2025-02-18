import hashlib
from uuid import UUID, uuid5

UUID_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")


def get_uuid(content: str) -> UUID:
    return uuid5(UUID_NAMESPACE, hashlib.md5(content.encode()).hexdigest())
