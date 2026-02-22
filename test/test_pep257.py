# Copyright 2025 Nigel Hayward-Smith
#
# Licensed under the MIT License.

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    rc = main(argv=["--add-ignore", "D100,D101,D102,D103,D104,D105,D106,D107"])
    assert rc == 0, "Found code style errors / warnings"
