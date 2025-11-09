"""Пакет с интеграциями внешних сервисов."""

from .shelter_cloud import (
    ShelterCloudAuthenticationError,
    ShelterCloudAvailabilityError,
    ShelterCloudConfig,
    ShelterCloudOfflineService,
    ShelterCloudService,
)

__all__ = [
    "ShelterCloudAuthenticationError",
    "ShelterCloudAvailabilityError",
    "ShelterCloudConfig",
    "ShelterCloudService",
    "ShelterCloudOfflineService",
]
