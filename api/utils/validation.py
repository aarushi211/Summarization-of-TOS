import ipaddress
import socket
import logging
from urllib.parse import urlparse
from api.core.config import settings

logger = logging.getLogger(__name__)

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fd00::/8"),
]

try:
    import magic as _magic
    def detect_mime(data: bytes) -> str:
        return _magic.from_buffer(data[:2048], mime=True)
except ImportError:
    logger.warning("python-magic not available — basic MIME detection fallback.")
    def detect_mime(data: bytes) -> str:
        return "application/pdf" if data[:4] == b"%PDF" else "application/octet-stream"

def assert_safe_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
        
        # DNS resolution to check for private IPs (SSRF protection)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("No hostname")
            
        ips = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in ips:
            ip_obj = ipaddress.ip_address(sockaddr[0])
            for blocked in _BLOCKED_NETWORKS:
                if ip_obj in blocked:
                    raise ValueError(f"Blocked internal IP: {ip_obj}")
        
        return url
    except Exception as e:
        raise ValueError(f"URL validation failed: {e}")
