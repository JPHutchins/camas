from tauri_app_sdk import Client


def test_invoke_no_args() -> None:
    c = Client(base_url="http://localhost:1420")
    assert c.invoke("ping") == {"command": "ping", "args": {}}


def test_invoke_with_args() -> None:
    c = Client(base_url="http://localhost:1420")
    assert c.invoke("greet", {"name": "world"}) == {
        "command": "greet",
        "args": {"name": "world"},
    }
