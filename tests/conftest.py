import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from binsim.config import load_config  # noqa: E402

CONFIG = os.path.join(REPO, "configs", "default.yaml")


@pytest.fixture
def cfg(tmp_path):
    return load_config(CONFIG, [f"assets.cache_dir={tmp_path / 'cache'}", f"run.output_dir={tmp_path / 'out'}"])


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    from binsim.assets import prepare_all
    d = tmp_path_factory.mktemp("cache")
    c = load_config(CONFIG, [f"assets.cache_dir={d}"])
    return prepare_all(c)
