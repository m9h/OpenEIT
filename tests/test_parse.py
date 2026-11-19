"""Serial line parsing, checked against frames captured from a Spectra board."""
from pathlib import Path

from OpenEIT.backend.serialhandler import parse_any_line

FRAMES = (Path(__file__).parent / "data" / "spectra_16e_frames.txt").read_text().splitlines()


def test_real_16_electrode_frames_parse():
    # stock firmware, mode d: 16 electrodes, opposition drive, 192 values per frame
    assert FRAMES
    for line in FRAMES:
        values = parse_any_line(line, "d")
        assert len(values) == 192
        assert all(v > 0 for v in values)


def test_time_series_and_spectroscopy_formats():
    assert parse_any_line("1.5, 2.5,3", "a") == [1.5, 2.5, 3.0]
    assert parse_any_line("magnitudes: 200;10.5;500;11", "b") == [200.0, 10.5, 500.0, 11.0]


def test_malformed_lines_are_rejected():
    assert parse_any_line("OpenEIT", "d") is None
    assert parse_any_line("magnitudes: 1,2,x", "d") is None
