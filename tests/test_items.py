# TODO: écrire les tests pour les items de base

import pytest
from llmtasker.items.base import BaseItem


def test_base_item_uncorrect_input():
    with pytest.raises(Exception):
        BaseItem[str, str](id="1", input=1, output=None)
