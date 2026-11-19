"""Each reconstruction wrapper should turn a change in the data into an image."""
from pathlib import Path

import numpy as np
import pytest

from OpenEIT.backend.serialhandler import parse_any_line
from OpenEIT.reconstruction import BpReconstruction, GreitReconstruction, JacReconstruction

FRAMES = (Path(__file__).parent / "data" / "spectra_16e_frames.txt").read_text().splitlines()


@pytest.mark.parametrize("cls", [JacReconstruction, BpReconstruction, GreitReconstruction])
def test_perturbed_frame_gives_nonzero_image(cls):
    baseline = np.array(parse_any_line(FRAMES[0], "d"))
    perturbed = baseline.copy()
    perturbed[40:60] *= 1.2

    recon = cls(n_el=16)
    recon.update_reference(baseline)
    recon.eit_reconstruction(baseline)          # first frame becomes the reference
    img = np.asarray(recon.eit_reconstruction(perturbed), dtype=float)

    assert img.size > 0
    assert np.nanmax(np.abs(img)) > 0
