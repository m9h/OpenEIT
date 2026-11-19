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


@pytest.mark.parametrize("n_el", [8, 16, 32])
@pytest.mark.parametrize("cls", [JacReconstruction, BpReconstruction, GreitReconstruction])
def test_simulated_target_is_located(cls, n_el):
    """A conductive disc at (0.4, 0.3) should give the image its peak in that quadrant.

    The data come from pyEIT's forward model with the same protocol the firmware
    scans (opposition drive, adjacent measurement), so this also pins the
    measurement ordering the wrappers assume.
    """
    from pyeit import mesh
    from pyeit.eit.fem import EITForward
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle

    recon = cls(n_el=n_el)
    fine = mesh.create(n_el, h0=0.06)
    fwd = EITForward(fine, recon.protocol)
    v0 = fwd.solve_eit()
    target = mesh.set_perm(fine, anomaly=PyEITAnomaly_Circle(center=[0.4, 0.3], r=0.2, perm=10.0), background=1.0)
    v1 = fwd.solve_eit(perm=target.perm)
    assert len(v0) == {8: 32, 16: 192, 32: 896}[n_el]

    recon.update_reference(v0)
    recon.eit_reconstruction(v0)
    img = np.asarray(recon.eit_reconstruction(v1), dtype=float)

    if cls is GreitReconstruction:
        iy, ix = np.unravel_index(np.nanargmax(np.abs(img)), img.shape)
        x, y = recon.gx[iy, ix], recon.gy[iy, ix]
    else:
        x, y = recon.mesh_obj.node[np.nanargmax(np.abs(img)), :2]
    assert x > 0 and y > 0
    assert np.hypot(x - 0.4, y - 0.3) < 0.5  # back projection with 8 electrodes is coarse
