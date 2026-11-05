"""The local GUI: security checks, uploads, and split -> merge through the HTTP API."""

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pytest
from seamsplit.audio import AudioSegment

import seamsplit.gui as gui_module
from seamsplit.cli import _build_parser
from seamsplit.gui import (
    build_split_config,
    create_server,
    safe_name,
    summarize_merge,
)
from seamsplit.merger import BoundaryInfo, MergeReport

SR = 22050


def _song(seconds: int = 40) -> bytes:
    rng = np.random.default_rng(1)
    t = np.arange(SR * seconds) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    for gap in (12, 26):
        x[int(gap * SR) : int((gap + 0.4) * SR)] *= 0.01
    pcm = (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)
    seg = AudioSegment(pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


class Client:
    def __init__(self, server):
        self.server = server
        self.base = f"http://127.0.0.1:{server.port}"

    def call(self, method, path, body=None, token=True, headers=None, **params):
        if token:
            params["t"] = self.server.token
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        data = None
        hdrs = dict(headers or {})
        if isinstance(body, (dict, list)):
            data, hdrs["Content-Type"] = json.dumps(body).encode(), "application/json"
        elif body is not None:
            data = body
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def json(self, method, path, body=None, **params):
        status, raw, _ = self.call(method, path, body, **params)
        return status, json.loads(raw)

    def wait(self, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, job = self.json("GET", "/api/job", id=job_id)
            if job["state"] != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")


@pytest.fixture
def gui(tmp_path):
    server = create_server(tmp_path / "work", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(server)
    server.shutdown()
    server.server_close()


# --- security -------------------------------------------------------------------


def test_requests_without_the_token_are_refused(gui):
    for path in ("/", "/api/state", "/api/info", "/download/chunks.zip"):
        assert gui.call("GET", path, token=False)[0] == 403
    assert gui.call("POST", "/api/split", {}, token=False)[0] == 403
    assert gui.call("POST", "/api/quit", token=False)[0] == 403
    assert gui.call("GET", "/api/state", t="wrong", token=False)[0] == 403


def test_requests_for_another_host_or_origin_are_refused(gui):
    # DNS rebinding: the browser sends the attacker's name as Host
    assert gui.call("GET", "/api/state", headers={"Host": "evil.example"})[0] == 403
    # a page on another site posting to us
    status, _, _ = gui.call(
        "POST", "/api/clear", headers={"Origin": "http://evil.example"}, area="input"
    )
    assert status == 403
    same_origin = {"Origin": f"http://127.0.0.1:{gui.server.port}"}
    assert gui.call("POST", "/api/clear", headers=same_origin, area="input")[0] == 200


def test_the_server_listens_on_loopback_only(gui):
    assert gui.server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("song.mp3", "song.mp3"),
        ("../../evil.wav", "evil.wav"),
        ("a\\b\\c.flac", "c.flac"),
        ("we:ird|na?me.wav", "we_ird_na_me.wav"),
        ("Bài hát.mp3", "Bài hát.mp3"),
    ],
)
def test_safe_name_cannot_leave_its_folder(raw, expected):
    assert safe_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "..", ".", "///", " . "])
def test_safe_name_rejects_names_with_nothing_usable(raw):
    with pytest.raises(ValueError):
        safe_name(raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CON", "_CON"),
        ("nul.wav", "_nul.wav"),
        ("Aux.tar.gz", "_Aux.tar.gz"),
        ("PRN.mp3", "_PRN.mp3"),
        ("COM1.flac", "_COM1.flac"),
        ("lpt9.wav", "_lpt9.wav"),
        ("COM\u00b9.mp3", "_COM\u00b9.mp3"),
        ("CONIN$", "_CONIN$"),
        ("con .mp3", "_con .mp3"),
        ("C:\\music\\NUL.wav", "_NUL.wav"),
    ],
)
def test_safe_name_defuses_windows_device_names(raw, expected):
    assert safe_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["console.mp3", "nullable.wav", "COM10.wav", "a.con", "my CON.mp3", "Bài hát.mp3", "_CON.mp3"],
)
def test_safe_name_leaves_names_that_only_look_like_devices(raw):
    assert safe_name(raw) == raw


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("song.", "song"),
        ("song .", "song"),
        ("a. .", "a"),
        ("a" * 199 + ". b", "a" * 199),  # cut to 200 characters, and not left ending in a dot
    ],
)
def test_safe_name_does_not_end_in_a_dot_or_a_space(raw, expected):
    assert safe_name(raw) == expected


def test_upload_name_with_a_path_is_stored_inside_the_work_folder(gui, tmp_path):
    status, body = gui.json(
        "POST", "/api/upload", b"RIFFxxxx", area="processed", name="../../evil.wav"
    )
    assert status == 200 and body["name"] == "evil.wav"
    assert (tmp_path / "work" / "processed" / "evil.wav").is_file()
    assert not (tmp_path / "evil.wav").exists()


def test_upload_named_like_a_windows_device_is_stored_under_another_name(gui, tmp_path):
    status, body = gui.json("POST", "/api/upload", b"RIFFxxxx", area="processed", name="NUL.wav")
    assert status == 200 and body["name"] == "_NUL.wav"
    assert (tmp_path / "work" / "processed" / "_NUL.wav").is_file()


def test_downloads_cannot_reach_outside_the_chunks_folder(gui, tmp_path):
    (tmp_path / "work" / "secret.txt").write_text("nope")
    for name in ("..%2Fsecret.txt", "..", "%2e%2e%2fsecret.txt"):
        assert gui.call("GET", f"/download/chunks/{name}")[0] == 404
    assert gui.call("GET", "/download/input/anything")[0] == 404


# --- uploads ----------------------------------------------------------------------


def test_upload_validates_the_manifest_and_replaces_single_file_areas(gui):
    bad = gui.json("POST", "/api/upload", b"not json", area="manifest", name="manifest.json")
    assert bad[0] == 400
    assert gui.json("GET", "/api/state")[1]["manifest"] is None

    good = json.dumps({"overlap_ms": 0, "parts": ["a.wav"]}).encode()
    assert gui.json("POST", "/api/upload", good, area="manifest", name="whatever.json")[0] == 200
    assert gui.json("GET", "/api/state")[1]["manifest"] == "uploaded"

    gui.json("POST", "/api/upload", b"one", area="input", name="a.mp3")
    gui.json("POST", "/api/upload", b"two", area="input", name="b.mp3")
    assert [f["name"] for f in gui.json("GET", "/api/state")[1]["input"]] == ["b.mp3"]


def test_upload_rejects_a_manifest_with_the_wrong_types(gui):
    for bad in (
        {"parts": [1], "overlap_ms": 0},
        {"parts": ["a.wav"], "overlap_ms": "0"},
        {"parts": ["a.wav"], "overlap_ms": 0, "segments": "x"},
    ):
        status, body = gui.json(
            "POST", "/api/upload", json.dumps(bad).encode(), area="manifest", name="manifest.json"
        )
        assert status == 400
        assert "not a valid seamsplit manifest" in body["error"]
    assert gui.json("GET", "/api/state")[1]["manifest"] is None


def test_a_bad_manifest_upload_keeps_the_manifest_already_there(gui):
    keep = json.dumps({"overlap_ms": 0, "parts": ["a.wav"]}).encode()
    assert gui.json("POST", "/api/upload", keep, area="manifest", name="manifest.json")[0] == 200
    stored = gui.server.workspace.path("manifest", "manifest.json")

    for bad in (b"not json", json.dumps({"parts": [1], "overlap_ms": 0}).encode()):
        status, _ = gui.json("POST", "/api/upload", bad, area="manifest", name="manifest.json")
        assert status == 400
        assert gui.json("GET", "/api/state")[1]["manifest"] == "uploaded"
        assert stored.read_bytes() == keep
        assert not list(stored.parent.glob("*.part"))

    newer = json.dumps({"overlap_ms": 0, "parts": ["b.wav"]}).encode()
    assert gui.json("POST", "/api/upload", newer, area="manifest", name="manifest.json")[0] == 200
    assert stored.read_bytes() == newer  # a valid manifest still replaces the old one


def test_a_cut_short_upload_keeps_the_file_already_there(gui):
    import socket

    assert gui.json("POST", "/api/upload", b"first", area="input", name="a.mp3")[0] == 200

    query = urllib.parse.urlencode({"area": "input", "name": "b.mp3", "t": gui.server.token})
    with socket.create_connection(("127.0.0.1", gui.server.port)) as sock:
        sock.sendall(
            (
                f"POST /api/upload?{query} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{gui.server.port}\r\n"
                "Content-Length: 100\r\n\r\n"
            ).encode()
            + b"only ten b"  # promises 100 bytes, sends 10, then hangs up its side
        )
        sock.shutdown(socket.SHUT_WR)
        assert b" 400 " in sock.recv(4096).split(b"\r\n", 1)[0]

    state = gui.json("GET", "/api/state")[1]
    assert [f["name"] for f in state["input"]] == ["a.mp3"]
    assert gui.server.workspace.path("input", "a.mp3").read_bytes() == b"first"


def test_upload_needs_a_known_area(gui):
    assert gui.json("POST", "/api/upload", b"x", area="merged", name="a.wav")[0] == 400
    assert gui.json("POST", "/api/clear", area="chunks")[0] == 400


# --- split and merge ------------------------------------------------------------------


def test_split_then_merge_through_the_api(gui):
    original = _song()
    assert gui.json("POST", "/api/upload", original, area="input", name="song.wav")[0] == 200

    status, started = gui.json(
        "POST",
        "/api/split",
        {"limit": "20", "format": "flac", "overlap": "2", "before_slack": "8", "after_slack": "3"},
    )
    assert status == 200
    job = gui.wait(started["job"])
    assert job["state"] == "done", job["error"]

    state = gui.json("GET", "/api/state")[1]
    parts = state["split"]["parts"]
    assert len(parts) >= 2 and all(p.endswith(".flac") for p in parts)
    assert state["manifest"] == "split"
    assert (
        state["split"]["max_export_ms"] <= 20_000
    )  # limit 20 = target + after + overlap, flac: no margin

    # download the zip, and one chunk
    status, raw, headers = gui.call("GET", "/download/chunks.zip")
    assert status == 200 and headers["Content-Type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        assert sorted(zf.namelist()) == sorted(parts + ["manifest.json"])
    assert gui.call("GET", f"/download/chunks/{urllib.parse.quote(parts[0])}")[0] == 200

    # "the AI service" hands every chunk back with a suffix
    for name in parts:
        _, chunk, _ = gui.call("GET", f"/download/chunks/{name}")
        gui.json(
            "POST",
            "/api/upload",
            chunk,
            area="processed",
            name=name.replace(".flac", "_instrumental.flac"),
        )
    gui.json(
        "POST", "/api/upload", chunk, area="processed", name="song_part01_vocals.flac"
    )  # a decoy

    status, started = gui.json(
        "POST", "/api/merge", {"match": "instrumental", "format": "wav", "samplerate": "22050"}
    )
    assert status == 200
    job = gui.wait(started["job"])
    assert job["state"] == "done", job["error"]
    assert job["result"]["file"] == "song_merged.wav"
    assert job["result"]["seams"] == len(parts) - 1

    status, merged_raw, _ = gui.call("GET", "/download/merged/song_merged.wav")
    merged = AudioSegment.from_file(io.BytesIO(merged_raw))
    src = AudioSegment.from_file(io.BytesIO(original))
    a = np.array(src.get_array_of_samples()).astype(float)
    b = np.array(merged.get_array_of_samples()).astype(float)
    assert abs(len(a) - len(b)) <= 1  # whole-millisecond cuts can cost a sample at the very end
    n = (
        min(len(a), len(b)) - 1000
    )  # the 8 ms fade at the very start and end of the track is intended
    a, b = a[1000:n], b[1000:n]
    assert 10 * np.log10(np.mean(a**2) / max(np.mean((a - b) ** 2), 1e-12)) > 60


def test_a_new_split_clears_the_previous_job(gui):
    gui.json("POST", "/api/upload", _song(), area="input", name="song.wav")
    gui.json("POST", "/api/upload", b"x", area="processed", name="old_part01.wav")
    job = gui.wait(
        gui.json("POST", "/api/split", {"limit": "20", "format": "wav", "overlap": "2"})[1]["job"]
    )
    assert job["state"] == "done", job["error"]
    assert gui.json("GET", "/api/state")[1]["processed"] == []


def test_errors_reach_the_page_as_messages(gui):
    assert gui.json("POST", "/api/split", {})[0] == 400  # nothing dropped yet
    assert gui.json("POST", "/api/merge", {})[0] == 400  # no manifest
    gui.json("POST", "/api/upload", b"this is not audio", area="input", name="song.mp3")
    status, started = gui.json("POST", "/api/split", {"limit": "20"})
    job = gui.wait(started["job"])
    assert status == 200 and job["state"] == "error" and job["error"]
    assert gui.json("GET", "/api/job", id="nope")[0] == 404


@pytest.mark.parametrize("samplerate", ["0", "-44100", "0.4"])
def test_merge_rejects_a_sample_rate_that_is_not_positive(gui, samplerate):
    manifest = json.dumps({"overlap_ms": 0, "parts": ["a.wav"]}).encode()
    up = "/api/upload"
    assert gui.json("POST", up, manifest, area="manifest", name="manifest.json")[0] == 200
    assert gui.json("POST", up, b"x", area="processed", name="a.wav")[0] == 200

    status, body = gui.json("POST", "/api/merge", {"samplerate": samplerate})
    assert status == 400  # refused up front, not as a failed job afterwards
    assert "samplerate" in body["error"]
    assert not gui.server.jobs


def test_only_one_job_runs_at_a_time(gui):
    gui.json("POST", "/api/upload", _song(), area="input", name="song.wav")
    server = gui.server
    blocker = threading.Event()
    server.start_job("split", lambda job: blocker.wait(10))
    try:
        assert gui.json("POST", "/api/split", {"limit": "20"})[0] == 409
    finally:
        blocker.set()


# --- form values -------------------------------------------------------------------------


def test_build_split_config_derives_the_target_like_the_cli(tmp_path):
    config, limit = build_split_config({"limit": "59", "format": "mp3"}, str(tmp_path))
    assert limit == 59 and config.target_s == pytest.approx(59 - 3 - 4 - 0.25)
    config, _ = build_split_config({"limit": "59", "format": "FLAC"}, str(tmp_path))
    assert config.export_format == "flac" and config.target_s == pytest.approx(52)


def test_build_split_config_keeps_a_short_limit_valid(tmp_path):
    config, _ = build_split_config({"limit": "20", "format": "wav", "overlap": "2"}, str(tmp_path))
    assert config.before_slack_s < config.target_s  # the default 20 s slack was scaled down


@pytest.mark.parametrize(
    "params, message",
    [
        ({"limit": "6", "overlap": "4"}, "too short"),
        ({"format": "ogg"}, "format must be"),
        ({"limit": "abc"}, "must be a number"),
        ({"limit": "nan"}, "must be a number"),
    ],
)
def test_build_split_config_rejects_bad_values(tmp_path, params, message):
    with pytest.raises(ValueError, match=message):
        build_split_config(params, str(tmp_path))


def test_summarize_merge():
    report = MergeReport(
        boundaries=[
            BoundaryInfo(1, 1.5, 0.99, True, seam_ms=10_000.0, seam_match_db=40.0),
            BoundaryInfo(2, -3.0, 0.5, True, seam_ms=65_500.0, seam_match_db=12.0, flagged=True),
        ],
        warnings=["something"],
    )
    summary = summarize_merge(report)
    assert summary == {
        "seams": 2,
        "worst_match_db": 12.0,
        "flagged": ["1:05.5"],
        "largest_offset_ms": -3.0,
        "warnings": ["something"],
    }


# --- command line -------------------------------------------------------------------------


def test_gui_subcommand_parses():
    args = _build_parser().parse_args(["gui", "--port", "8123", "--no-browser", "--workdir", "x"])
    assert (args.command, args.port, args.no_browser, args.workdir) == ("gui", 8123, True, "x")
    assert _build_parser().parse_args(["gui"]).port == 0


# --- merge options, reports, ffmpeg ---------------------------------------------------------


def _split_and_hand_back(gui):
    """Split a song through the API and put every chunk in ``processed`` unchanged."""
    gui.json("POST", "/api/upload", _song(), area="input", name="song.wav")
    status, started = gui.json(
        "POST",
        "/api/split",
        {"limit": "20", "format": "wav", "overlap": "2", "before_slack": "8", "after_slack": "3"},
    )
    assert status == 200 and gui.wait(started["job"])["state"] == "done"
    parts = gui.json("GET", "/api/state")[1]["split"]["parts"]
    for name in parts:
        chunk = gui.call("GET", f"/download/chunks/{name}")[1]
        gui.json("POST", "/api/upload", chunk, area="processed", name=name)
    return parts


def _merge(gui, **params):
    status, started = gui.json("POST", "/api/merge", {"samplerate": "22050", **params})
    assert status == 200, started
    return gui.wait(started["job"])


def test_info_says_what_reads_and_writes_the_audio(gui):
    info = gui.json("GET", "/api/info")[1]
    assert info["audio_engine"]["pyav"] and info["audio_engine"]["ffmpeg"]
    assert "320k" in info["bitrates"] and info["seam_fade_ms"] > 0


def test_merge_applies_the_bitrate_and_seam_search_and_writes_a_report(gui):
    _split_and_hand_back(gui)

    low = _merge(gui, format="mp3", bitrate="128k", seam_search=True, seam_fade_ms="100")
    assert low["state"] == "done", low["error"]
    assert low["result"]["file"] == "song_merged.mp3"
    assert low["result"]["reports"] == ["song_merged_report.md", "song_merged_report.json"]
    small = len(gui.call("GET", "/download/merged/song_merged.mp3")[1])

    state = gui.json("GET", "/api/state")[1]
    assert [f["name"] for f in state["merged"]] == [
        "song_merged.mp3"
    ]  # the reports are not "the result"
    assert sorted(state["reports"]) == ["song_merged_report.json", "song_merged_report.md"]

    status, raw, _ = gui.call("GET", "/download/merged/song_merged_report.json")
    report = json.loads(raw)
    assert status == 200 and report["boundaries"]
    assert all(
        b["fade_ms"] == pytest.approx(100, abs=1) for b in report["boundaries"]
    )  # seam search ran
    markdown = gui.call("GET", "/download/merged/song_merged_report.md")[1].decode()
    assert "seam" in markdown.lower()

    high = _merge(gui, format="mp3", bitrate="320k")
    assert high["state"] == "done", high["error"]
    # the test song is 22.05 kHz, where MP3 tops out at 160 kbps: 320k can only give 160k (1.25x)
    assert len(gui.call("GET", "/download/merged/song_merged.mp3")[1]) > 1.2 * small
    report = json.loads(gui.call("GET", "/download/merged/song_merged_report.json")[1])
    assert all(
        b["fade_ms"] > 1000 for b in report["boundaries"]
    )  # this time the fade spans the overlap


@pytest.mark.parametrize(
    "params, message",
    [
        ({"bitrate": "999k"}, "bitrate must be one of"),
        ({"bitrate": "320000"}, "bitrate must be one of"),
        ({"seam_fade_ms": "0"}, "seam_fade_ms"),
        ({"seam_fade_ms": "abc"}, "seam_fade_ms"),
    ],
)
def test_merge_rejects_bad_bitrate_and_seam_fade_up_front(gui, params, message):
    manifest = json.dumps({"overlap_ms": 0, "parts": ["a.wav"]}).encode()
    gui.json("POST", "/api/upload", manifest, area="manifest", name="manifest.json")
    gui.json("POST", "/api/upload", b"x", area="processed", name="a.wav")
    status, body = gui.json("POST", "/api/merge", params)
    assert status == 400 and message in body["error"]
    assert not gui.server.jobs


# --- cancelling -------------------------------------------------------------------------------


def test_cancelling_a_split_stops_it_and_removes_the_partial_chunks(gui, monkeypatch):
    gui.json("POST", "/api/upload", _song(), area="input", name="song.wav")
    chunks = gui.server.workspace.path("chunks")
    reached, go = threading.Event(), threading.Event()

    def fake_split(path, config, progress=None):
        (chunks / "song_part01.wav").write_bytes(b"partial")
        progress(1, 3, "song_part01.wav")  # not cancelled yet: carries on
        reached.set()
        assert go.wait(10)
        progress(2, 3, "song_part02.wav")  # cancel was asked for: raises
        raise AssertionError("the split should have stopped")

    monkeypatch.setattr(gui_module, "split_audio", fake_split)
    job_id = gui.json("POST", "/api/split", {"limit": "20"})[1]["job"]
    assert reached.wait(10)

    assert gui.json("POST", "/api/cancel", id=job_id)[0] == 200
    assert gui.json("GET", "/api/job", id=job_id)[1]["cancelling"] is True
    go.set()
    job = gui.wait(job_id)
    assert job["state"] == "cancelled" and job["error"] is None and job["cancelling"] is False
    assert gui.json("GET", "/api/state")[1]["chunks"] == []

    monkeypatch.undo()  # the page can start something new afterwards
    assert (
        gui.wait(gui.json("POST", "/api/split", {"limit": "20", "format": "wav"})[1]["job"])[
            "state"
        ]
        == "done"
    )


def _upload_a_manifest_and_one_chunk(gui):
    manifest = json.dumps({"overlap_ms": 0, "parts": ["a.wav", "b.wav"]}).encode()
    gui.json("POST", "/api/upload", manifest, area="manifest", name="manifest.json")
    gui.json("POST", "/api/upload", b"x", area="processed", name="a.wav")


def test_cancelling_a_merge_between_chunks(gui, monkeypatch):
    _upload_a_manifest_and_one_chunk(gui)
    reached, go = threading.Event(), threading.Event()
    seen = []

    def fake_merge(processed_dir, manifest_path, output_path, progress=None, **kwargs):
        progress(1, 2, "a.wav")
        seen.append(kwargs["bitrate"])
        reached.set()
        assert go.wait(10)
        progress(2, 2, "b.wav")
        raise AssertionError("the merge should have stopped")

    monkeypatch.setattr(gui_module, "merge_audio", fake_merge)
    job_id = gui.json("POST", "/api/merge", {})[1]["job"]
    assert reached.wait(10)
    assert gui.json("GET", "/api/job", id=job_id)[1]["progress"]["name"] == "a.wav"
    gui.json("POST", "/api/cancel", id=job_id)
    go.set()
    assert gui.wait(job_id)["state"] == "cancelled"
    assert gui.json("GET", "/api/state")[1]["merged"] == []
    assert seen == ["320k"]


def test_a_cancel_that_arrives_while_the_result_is_being_written_discards_it(gui, monkeypatch):
    _upload_a_manifest_and_one_chunk(gui)
    reached, go = threading.Event(), threading.Event()

    def fake_merge(processed_dir, manifest_path, output_path, progress=None, **kwargs):
        progress(1, 2, "a.wav")
        with open(output_path, "wb") as f:  # the encode, which cannot be interrupted
            f.write(b"merged")
        reached.set()
        assert go.wait(10)
        return AudioSegment.silent(100)

    monkeypatch.setattr(gui_module, "merge_audio", fake_merge)
    job_id = gui.json("POST", "/api/merge", {})[1]["job"]
    assert reached.wait(10)
    gui.json("POST", "/api/cancel", id=job_id)
    go.set()
    assert gui.wait(job_id)["state"] == "cancelled"
    assert gui.json("GET", "/api/state")[1]["merged"] == []  # the file was removed


def test_cancel_of_a_finished_or_unknown_job(gui):
    _split_and_hand_back(gui)
    job = _merge(gui, format="wav")
    assert job["state"] == "done"
    assert gui.json("POST", "/api/cancel", id=job["id"])[0] == 200  # nothing to stop
    assert gui.json("GET", "/api/job", id=job["id"])[1]["state"] == "done"
    assert gui.json("POST", "/api/cancel", id="nope")[0] == 404
    assert gui.call("POST", "/api/cancel", id=job["id"], token=False)[0] == 403


# --- service presets and renamed results -------------------------------------------


@pytest.fixture
def no_user_presets(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("SEAMSPLIT_SERVICES", raising=False)
    return tmp_path / "config" / "seamsplit"


def test_the_page_is_offered_the_service_presets(gui, no_user_presets):
    info = gui.json("GET", "/api/info")[1]
    (xminus,) = info["services"]
    assert xminus["name"] == "xminus" and xminus["limit_s"] == 59.0
    assert xminus["format"] is None and xminus["overlap_s"] is None
    assert info["services_error"] is None
    page = gui.call("GET", "/")[1].decode()
    assert 'id="service"' in page and "setupServices" in page


def test_the_users_presets_are_offered_too_and_a_broken_file_does_not_stop_the_page(
    gui, no_user_presets
):
    no_user_presets.mkdir(parents=True)
    path = no_user_presets / "services.json"
    path.write_text(json.dumps({"mine": {"limit_s": 25, "format": "wav", "overlap_s": 1}}))
    names = [s["name"] for s in gui.json("GET", "/api/info")[1]["services"]]
    assert names == ["mine", "xminus"]
    path.write_text("{broken")
    info = gui.json("GET", "/api/info")[1]
    assert [s["name"] for s in info["services"]] == ["xminus"]
    assert "is not valid JSON" in info["services_error"]


def test_results_with_lost_names_are_put_in_order_by_the_merge_in_the_page(gui):
    original = _song()
    gui.json("POST", "/api/upload", original, area="input", name="song.wav")
    job = gui.wait(
        gui.json(
            "POST",
            "/api/split",
            {
                "limit": "20",
                "format": "wav",
                "overlap": "2",
                "before_slack": "8",
                "after_slack": "3",
            },
        )[1]["job"]
    )
    assert job["state"] == "done", job["error"]
    parts = gui.json("GET", "/api/state")[1]["split"]["parts"]
    assert len(parts) >= 3
    # the service gives them back with names that say nothing, and in another order
    for k, name in enumerate(reversed(parts)):
        _, chunk, _ = gui.call("GET", f"/download/chunks/{name}")
        gui.json("POST", "/api/upload", chunk, area="processed", name=f"download ({k + 1}).wav")
    started = gui.json("POST", "/api/merge", {"format": "wav", "samplerate": "22050"})[1]
    job = gui.wait(started["job"])
    assert job["state"] == "done", job["error"]
    assert any("were not in part order" in w for w in job["result"]["warnings"])
    assert job["result"]["flagged"] == []
