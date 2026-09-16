"""Runs regardless of whether `implicit` is installed — verifies the friendly error."""
import sys
from unittest.mock import patch

import pandas as pd
import pytest


def test_clear_error_when_implicit_not_installed():
    from src.providers.als_provider import ALSProvider

    provider = ALSProvider()
    ratings = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [1.0]})

    # Also patch implicit.als, not just implicit — if any earlier test in the same
    # process already imported implicit (e.g. test_als_provider.py, when the real
    # package is installed), that submodule is cached in sys.modules and patching only
    # the parent name to None does not invalidate it, letting the real import succeed.
    with patch.dict(sys.modules, {"implicit": None, "implicit.als": None}):
        with pytest.raises(ImportError, match="implicit"):
            provider.fit(ratings)
