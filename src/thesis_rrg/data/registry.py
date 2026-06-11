from __future__ import annotations

from thesis_rrg.data.builders.baseline import build_baseline_bundle
from thesis_rrg.data.builders.hypothesis1 import build_h1_bundle
from thesis_rrg.data.builders.hypothesis2 import build_h2_bundle
from thesis_rrg.data.builders.hypothesis3 import build_h3_bundle
from thesis_rrg.data.builders.hypothesis4 import build_h4_bundle
from thesis_rrg.registry import Registry

DATA_BUILDER_REGISTRY = Registry("data_builder")
DATA_BUILDER_REGISTRY.register("baseline", build_baseline_bundle)
DATA_BUILDER_REGISTRY.register("h1", build_h1_bundle)
DATA_BUILDER_REGISTRY.register("h2", build_h2_bundle)
DATA_BUILDER_REGISTRY.register("h3", build_h3_bundle)
DATA_BUILDER_REGISTRY.register("h4", build_h4_bundle)