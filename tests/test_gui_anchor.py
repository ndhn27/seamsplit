"""The GUI side of the anchored merge and the listening kit."""

import json
import threading
import zipfile
import io

import numpy as np
import pytest

from fake_service import hand_back
from seamsplit.gui import create_server, summarize_anchor
from seamsplit.merger import ChunkInfo, MergeReport
from test_gui import Client, _song, _upload_a_manifest_and_one_chunk


@pytest.fixture
def gui(tmp_path):
    server = create_server(tmp_path / "work", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(server)
    server.shutdown()
    server.server_close()


@pytest.fixture
def served(gui, tmp_path):
    """The GUI has split a song, and a pretend service has given back both stems."""
    gui.json("POST", "/api/upload", _song(), area="input", name="song.wav")
    status, started = gui.json(
        "POST",
        "/api/split",
        {"limit": "20", "format": "wav", "overlap": "2", "before_slack": "8", "after_slack": "3"},
    )
    assert status == 200 and gui.wait(started["job"])["state"] == "done"
    parts = gui.json("GET", "/api/state")[1]["split"]["parts"]
    chunks, back = tmp_path / "chunks", tmp_path / "back"
    chunks.mkdir()
    for name in parts:
        (chunks / name).write_bytes(gui.call("GET", f"/download/chunks/{name}")[1])
    lags = [120, -250, 60, -90, 300, 0][: len(parts)]
    gains = [1.5, -2.0, 0.5, 3.0, -1.0, 0.0][: len(parts)]
    hand_back(str(chunks), str(back), parts, lags, gains, np.random.default_rng(4))
    for path in sorted(back.iterdir()):
        gui.json("POST", "/api/upload", path.read_bytes(), area="processed", name=path.name)
    return parts


def _merge(gui, **params):
    status, started = gui.json(
        "POST", "/api/merge", {"samplerate": "22050", "format": "wav", **params}
    )
    assert status == 200, started
    return gui.wait(started["job"])


# --- the merge ----------------------------------------------------------------------------


def test_an_anchored_merge_of_both_stems_through_the_api(gui, served):
    job = _merge(gui, match="instrumental", other_match="vocals", anchor=True)
    assert job["state"] == "done", job["error"]
    result = job["result"]
    assert result["file"] == "song_merged.wav" and result["other_file"] == "song_merged_vocals.wav"
    anchor = result["anchor"]
    assert anchor["chunks"] == len(served) == anchor["in_time"] == anchor["in_level"]
    assert anchor["worst_recon_db"] > 20 and anchor["flagged"] == []
    assert abs(anchor["largest_offset_ms"]) > 1  # the service did offset them

    state = gui.json("GET", "/api/state")[1]
    assert [f["name"] for f in state["merged"]] == ["song_merged.wav", "song_merged_vocals.wav"]
    assert gui.call("GET", "/download/merged/song_merged_vocals.wav")[0] == 200
    report = json.loads(gui.call("GET", "/download/merged/song_merged_report.json")[1])
    assert len(report["chunks"]) == len(served)


def test_without_anchoring_the_result_has_no_anchor_summary(gui, served):
    job = _merge(gui, match="instrumental")
    assert job["state"] == "done", job["error"]
    assert job["result"]["anchor"] is None and job["result"]["other_file"] is None
    assert job["result"]["audition"] == []


def test_anchoring_without_the_original_chunks_is_refused_up_front(gui):
    _upload_a_manifest_and_one_chunk(gui)  # chunks that came back, but no split done here
    status, answer = gui.json("POST", "/api/merge", {"anchor": True, "match": "instrumental"})
    assert status == 400 and "original chunks" in answer["error"]
    assert gui.json("GET", "/api/state")[1]["busy"] is False  # no job was started


@pytest.mark.parametrize(
    "params, message",
    [({"audition": "sometimes"}, "audition must be one of")],
)
def test_bad_audition_choice_is_refused(gui, served, params, message):
    status, answer = gui.json("POST", "/api/merge", {"samplerate": "22050", **params})
    assert status == 400 and message in answer["error"]


# --- the listening kit ----------------------------------------------------------------------


def test_the_kit_is_made_and_served_to_the_player(gui, served):
    assert gui.json("GET", "/api/audition")[1] == {"kit": None, "choices": [], "summary": None}
    job = _merge(gui, match="instrumental", other_match="vocals", anchor=True, audition="all")
    assert job["state"] == "done", job["error"]
    seams = job["result"]["audition"]
    assert seams == list(range(1, len(served)))
    assert gui.json("GET", "/api/state")[1]["audition"] is True

    data = gui.json("GET", "/api/audition")[1]
    assert [s["seam"] for s in data["kit"]["seams"]] == seams and data["choices"] == []
    first = data["kit"]["seams"][0]["clips"]["adaptive"]["file"]
    status, body, headers = gui.call("GET", f"/audition/{first}")
    assert status == 200 and headers["Content-Type"] == "audio/flac"
    assert headers["Accept-Ranges"] == "bytes" and len(body) == int(headers["Content-Length"])


def test_clips_are_served_in_ranges_so_that_the_player_can_seek(gui, served):
    _merge(gui, match="instrumental", audition="all")
    name = gui.json("GET", "/api/audition")[1]["kit"]["seams"][0]["clips"]["straight"]["file"]
    whole = gui.call("GET", f"/audition/{name}")[1]
    size = len(whole)

    status, body, headers = gui.call("GET", f"/audition/{name}", headers={"Range": "bytes=10-19"})
    assert (status, body) == (206, whole[10:20])
    assert headers["Content-Range"] == f"bytes 10-19/{size}"
    status, body, headers = gui.call("GET", f"/audition/{name}", headers={"Range": "bytes=100-"})
    assert (status, body) == (206, whole[100:])
    status, body, _ = gui.call("GET", f"/audition/{name}", headers={"Range": "bytes=-16"})
    assert (status, body) == (206, whole[-16:])
    status, body, _ = gui.call("GET", f"/audition/{name}", headers={"Range": f"bytes=0-{size * 9}"})
    assert (status, body) == (206, whole)
    for bad in (f"bytes={size}-", "bytes=9-3", "bytes=-0"):
        status, _, headers = gui.call("GET", f"/audition/{name}", headers={"Range": bad})
        assert status == 416 and headers["Content-Range"] == f"bytes */{size}"
    status, body, _ = gui.call("GET", f"/audition/{name}", headers={"Range": "lines=1-2"})
    assert (status, body) == (200, whole)  # not a range we know: the whole clip


@pytest.mark.parametrize(
    "path",
    [
        "/audition/nothing.flac",
        "/audition/..%2Fchunks%2Fmanifest.json",
        "/audition/audition.json",  # not a media file: the player reads it from the API
        "/audition/choices.json",
        "/audition/",
        "/download/audition/nothing.flac",
    ],
)
def test_only_clips_of_the_kit_are_served_from_it(gui, served, path):
    _merge(gui, match="instrumental", audition="all")
    assert gui.call("GET", path)[0] == 404


def test_the_kit_needs_the_token_like_everything_else(gui, served):
    _merge(gui, match="instrumental", audition="all")
    for path in ("/api/audition", "/download/audition.zip", "/audition/x.flac"):
        assert gui.call("GET", path, token=False)[0] == 403
    assert gui.call("POST", "/api/audition-choice", {}, token=False)[0] == 403


def test_the_whole_kit_downloads_as_a_zip_with_its_page(gui, served):
    assert gui.call("GET", "/download/audition.zip")[0] == 404  # nothing yet
    _merge(gui, match="instrumental", audition="all")
    status, body, headers = gui.call("GET", "/download/audition.zip")
    assert status == 200 and "listening-kit.zip" in headers["Content-Disposition"]
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert "index.html" in names and "audition.json" in names
    assert sum(n.endswith(".flac") for n in names) == 3 * (len(served) - 1)
    page = zipfile.ZipFile(io.BytesIO(body)).read("index.html").decode()
    assert "SeamAudition" in page


def test_answers_are_kept_summed_up_and_survive_a_reload(gui, served):
    _merge(gui, match="instrumental", audition="all")
    for seam, choice in ((1, "seam_search"), (2, "tie")):
        status, answer = gui.json(
            "POST", "/api/audition-choice", {"seam": seam, "choice": choice, "blind": True}
        )
        assert status == 200
    assert answer["summary"]["choices"] == 2 and answer["summary"]["counts"]["tie"] == 1
    data = gui.json("GET", "/api/audition")[1]
    assert [(c["seam"], c["kind"]) for c in data["choices"]] == [(1, "seam_search"), (2, "tie")]
    saved = gui.call("GET", "/download/audition/choices.json")[1]
    assert json.loads(saved)["choices"][0]["blind"] is True

    gui.json("POST", "/api/audition-choice", {"seam": 1, "choice": "straight", "blind": False})
    kinds = {c["seam"]: c["kind"] for c in gui.json("GET", "/api/audition")[1]["choices"]}
    assert kinds == {1: "straight", 2: "tie"}


@pytest.mark.parametrize(
    "body",
    [
        {"seam": "1", "choice": "straight"},
        {"seam": True, "choice": "straight"},
        {"seam": 1, "choice": 3},
        {"seam": 1},
        {"seam": 99, "choice": "straight"},
        {"seam": 1, "choice": "loudest"},
    ],
)
def test_a_bad_answer_is_refused_and_nothing_is_recorded(gui, served, body):
    _merge(gui, match="instrumental", audition="all")
    assert gui.json("POST", "/api/audition-choice", body)[0] == 400
    assert gui.json("GET", "/api/audition")[1]["choices"] == []


def test_there_is_nothing_to_answer_before_a_kit_exists(gui, served):
    status, answer = gui.json("POST", "/api/audition-choice", {"seam": 1, "choice": "straight"})
    assert status == 400 and "no listening kit" in answer["error"]


def test_a_new_merge_replaces_the_kit_and_its_answers(gui, served):
    _merge(gui, match="instrumental", audition="all")
    gui.json("POST", "/api/audition-choice", {"seam": 1, "choice": "tie", "blind": True})
    _merge(gui, match="instrumental", audition="off")
    assert gui.json("GET", "/api/audition")[1]["kit"] is None
    assert gui.json("GET", "/api/state")[1]["audition"] is False
    _merge(gui, match="instrumental", audition="all")
    assert gui.json("GET", "/api/audition")[1]["choices"] == []


def test_a_new_split_clears_the_kit_too(gui, served):
    _merge(gui, match="instrumental", audition="all")
    status, started = gui.json(
        "POST", "/api/split", {"limit": "20", "format": "wav", "overlap": "2"}
    )
    assert gui.wait(started["job"])["state"] == "done"
    assert gui.json("GET", "/api/state")[1]["audition"] is False


def test_cancelling_during_the_kit_leaves_no_half_made_one(gui, served, monkeypatch):
    import seamsplit.audition as audition

    original = audition._make_seam
    seen = []

    def slow(*args, **kwargs):
        seen.append(1)
        entry = original(*args, **kwargs)
        for job in gui.server.jobs.values():  # asked for while the first seam is made
            if job.state == "running":
                job.cancel()
        return entry

    monkeypatch.setattr(audition, "_make_seam", slow)
    status, started = gui.json(
        "POST",
        "/api/merge",
        {"samplerate": "22050", "format": "wav", "audition": "all", "match": "instrumental"},
    )
    job = gui.wait(started["job"])
    assert job["state"] == "cancelled" and seen
    state = gui.json("GET", "/api/state")[1]
    assert state["audition"] is False and state["merged"] == []


# --- the page and the summary --------------------------------------------------------------


def test_the_page_carries_the_player_and_the_new_controls(gui):
    status, body, _ = gui.call("GET", "/")
    page = body.decode()
    assert status == 200 and "/*AUDITION_JS*/" not in page and "SeamAudition" in page
    for element in ('id="other-match"', 'id="anchor"', 'id="audition"', 'id="listen-root"'):
        assert element in page


def test_summarize_anchor():
    assert summarize_anchor(MergeReport()) is None
    report = MergeReport(
        chunks=[
            ChunkInfo(1, lag_ms=-3.0, correlation=0.99, gain_db=1.0, recon_db=40.0),
            ChunkInfo(2, lag_ms=12.34, correlation=0.99, gain_db=None, recon_db=11.0, flagged=True),
            ChunkInfo(3, lag_ms=None, correlation=0.2),
        ]
    )
    assert summarize_anchor(report) == {
        "chunks": 3,
        "in_time": 2,
        "in_level": 1,
        "largest_offset_ms": 12.3,
        "worst_recon_db": 11.0,
        "worst_recon_chunk": 2,
        "flagged": [2],
    }
    bare = summarize_anchor(MergeReport(chunks=[ChunkInfo(1)]))
    assert bare["largest_offset_ms"] is None and bare["worst_recon_db"] is None
