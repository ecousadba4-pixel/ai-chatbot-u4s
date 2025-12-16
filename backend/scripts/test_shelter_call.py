import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.booking.models import Guests
from app.booking.shelter_client import ShelterCloudService


async def main() -> None:
    service = ShelterCloudService()
    today = datetime.now(ZoneInfo("UTC")).date()
    check_in = today + timedelta(days=7)
    check_out = check_in + timedelta(days=2)
    guests = Guests(adults=2, children=0)
    try:
        offers = await service.fetch_availability(
            check_in=check_in.isoformat(),
            check_out=check_out.isoformat(),
            guests=guests,
        )
    finally:
        await service.close()

    print("Check-in:", check_in)
    print("Check-out:", check_out)
    for offer in offers:
        print(offer)


if __name__ == "__main__":
    asyncio.run(main())
