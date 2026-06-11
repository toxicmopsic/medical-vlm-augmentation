from omegaconf import OmegaConf

from thesis_rrg.data.registry import DATA_BUILDER_REGISTRY


def test_registry_has_expected_builders():
    keys = set(DATA_BUILDER_REGISTRY.keys())
    assert {"baseline", "h1", "h2", "h3"}.issubset(keys)


def test_registry_lookup():
    fn = DATA_BUILDER_REGISTRY.get("baseline")
    assert callable(fn)
