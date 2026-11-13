import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from allograph_clustering import desk


class Page:
    """A desk whose page is whatever the test says, or the error it raises."""
    code, categories = "U+5019", "categories/U+5019.yaml"

    def __init__(self):
        self.result = "<p>候</p>"

    def page(self) -> str:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def served(monkeypatch):
    page, made = Page(), []

    class Server(ThreadingHTTPServer):
        def __init__(self, address, handler):
            super().__init__(("127.0.0.1", 0), handler)
            made.append(self)

    monkeypatch.setattr(desk, "ThreadingHTTPServer", Server)
    thread = threading.Thread(target=desk.serve, args=(page, 0), kwargs={"log": lambda m: None}, daemon=True)
    thread.start()
    while not made:
        pass
    yield page, f"http://127.0.0.1:{made[0].server_address[1]}"
    made[0].shutdown()
    thread.join()


def test_the_desk_serves_the_page_as_utf8(served):
    _, url = served
    with urllib.request.urlopen(url + "/") as reply:
        assert reply.headers["Content-Type"] == "text/html; charset=utf-8"
        assert reply.read().decode() == "<p>候</p>"


def test_a_broken_categories_file_is_reported_and_the_desk_keeps_serving(served):
    page, url = served
    page.result = ValueError("shape clusters [3] belong to no category")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(url + "/")
    assert caught.value.code == 500 and "belong to no category" in caught.value.read().decode()
    page.result = "<p>fixed</p>"
    assert urllib.request.urlopen(url + "/").read().decode() == "<p>fixed</p>"


def test_other_paths_are_not_found(served):
    _, url = served
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(url + "/report.html")
    assert caught.value.code == 404
