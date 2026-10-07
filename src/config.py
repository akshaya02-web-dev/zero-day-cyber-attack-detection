"""Central configuration system for the cyber-threat XAI project.

This module stores all project settings in one place using dataclasses so the
training pipeline, anomaly detector, and fusion layer can read configuration
without hardcoded attack names or scattered constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Dataset-specific configuration."""

    dataset_path: str = "data/raw"
    label_column: str = "label"
    attack_column: str = "attack_category"
    normal_label_names: tuple[str, ...] = ("NORMAL", "normal", "Normal", "BENIGN", "benign")
    known_attack_categories: tuple[str, ...] = ("DoS", "DDoS", "PortScan")
    held_out_unknown_attack_categories: tuple[str, ...] = ("Bot", "Web Attack")
    feature_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplittingConfig:
    """Train/validation/test split configuration."""

    random_state: int = 42
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(frozen=True)
class SupervisedModelConfig:
    """Configuration for known attack detection models."""

    enabled_models: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")
    handle_class_imbalance: bool = True
    class_weight: str | None = "balanced"


@dataclass(frozen=True)
class AnomalyDetectionConfig:
    """Configuration for Isolation Forest anomaly detection."""

    contamination: float = 0.01
    n_estimators: int = 200
    random_state: int = 42


@dataclass(frozen=True)
class DecisionFusionConfig:
    """Configuration for the known-vs-unknown decision engine."""

    known_attack_probability_threshold: float = 0.6
    anomaly_score_threshold: float = 0.0
    normal_confidence_threshold: float = 0.5
    known_attack_confidence_threshold: float = 0.6
    unknown_confidence_threshold: float = 0.5


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Enable or disable feature engineering preprocessing steps."""

    enabled: bool = True


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Feature selection configuration."""

    enabled: bool = True
    top_n_features: int = 20


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    supervised_model: SupervisedModelConfig = field(default_factory=SupervisedModelConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    decision_fusion: DecisionFusionConfig = field(default_factory=DecisionFusionConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    @property
    def dataset_path(self) -> Path:
        """Return dataset path as a Path object."""
        return Path(self.data.dataset_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a nested dictionary."""
        return asdict(self)


DEFAULT_CONFIG = ProjectConfig()


def get_default_config() -> ProjectConfig:
    """Return the default runtime configuration."""
    return ProjectConfig()


def build_config_from_dict(config_dict: dict[str, Any] | None = None) -> ProjectConfig:
    """Construct a project configuration from a dictionary.

    This helper enables overriding only the values needed for a specific run
    without editing the defaults in multiple files.
    """
    base = get_default_config()
    if not config_dict:
        return base

    merged = asdict(base)
    nested_update(merged, config_dict)

    return ProjectConfig(
        data=DataConfig(**merged["data"]),
        splitting=SplittingConfig(**merged["splitting"]),
        supervised_model=SupervisedModelConfig(**merged["supervised_model"]),
        anomaly_detection=AnomalyDetectionConfig(**merged["anomaly_detection"]),
        decision_fusion=DecisionFusionConfig(**merged["decision_fusion"]),
        feature_engineering=FeatureEngineeringConfig(**merged["feature_engineering"]),
        feature_selection=FeatureSelectionConfig(**merged["feature_selection"]),
    )


def nested_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively update nested dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested_update(base[key], value)
        else:
            base[key] = value


__all__ = [
    "DataConfig",
    "SplittingConfig",
    "SupervisedModelConfig",
    "AnomalyDetectionConfig",
    "DecisionFusionConfig",
    "FeatureEngineeringConfig",
    "FeatureSelectionConfig",
    "ProjectConfig",
    "DEFAULT_CONFIG",
    "get_default_config",
    "build_config_from_dict",
]
