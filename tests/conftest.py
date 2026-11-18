import pytest

from turkmorph.presets import demo_tokenizer


@pytest.fixture(scope="session")
def tok():
    return demo_tokenizer()
