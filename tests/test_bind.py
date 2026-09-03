import pytest

from whisper.bind import LOOPBACK_HOST, BindError, require_loopback


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST"])
def test_require_loopback_accepts_localhost(host):
    assert require_loopback(host) == LOOPBACK_HOST == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "::0",
        "*",
        "",
        " ",
        "[::]",
        "192.168.1.10",
        "10.0.0.1",
        "example.com",
        "8.8.8.8",
    ],
)
def test_require_loopback_rejects_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback(host)


def test_require_loopback_rejects_none():
    with pytest.raises(BindError):
        require_loopback(None)
