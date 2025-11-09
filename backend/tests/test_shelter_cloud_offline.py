from backend.app.services import ShelterCloudOfflineService


def test_offline_service_returns_offers_sorted_by_price():
    service = ShelterCloudOfflineService()

    offers = service.fetch_availability(
        check_in="2024-05-01",
        check_out="2024-05-03",
        adults=2,
        children=0,
        children_ages=[],
    )

    assert service.is_configured() is True
    assert offers
    assert offers[0]["name"].lower().startswith("стандарт")
    assert offers[0]["price"] == 19800
    assert offers[0]["breakfast_included"] is True
