import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def conn(tmp_path_factory):
    """Synthetic test fixture: cortex-shaped random wiring. Not mouse data."""
    import make_synthetic
    from mousedev import connectome
    p = tmp_path_factory.mktemp("brain") / "connectome.npz"
    rid, pos, types, pre, post, nsyn = make_synthetic.build(6000, 0, 40, 70.0)
    connectome.save(str(p), rid, pos, types, pre, post, nsyn,
                    {"source": "SYNTHETIC TEST FIXTURE (not a brain)", "synthetic": True})
    return connectome.load(str(p))


@pytest.fixture()
def cfg(tmp_path):
    from mousedev.config import Config
    c = Config()
    c.brain.w_syn, c.brain.inh_gain = 1.9125, 2.0          # calibrated for the fixture above
    c.rig.slot_ref_hz = [38.7, 17.9, 36.4, 36.5, 45.8, 39.9, 25.0, 34.9]
    c.state_dir = str(tmp_path / "state")
    c.web_dir = str(tmp_path / "web")
    (tmp_path / "web").mkdir()
    return c
