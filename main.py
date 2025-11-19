import os
from typing import List, Optional
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load .env variables
load_dotenv()

CALENDLY_API_TOKEN = os.getenv("CALENDLY_API_TOKEN")
CALENDLY_SCHEDULING_LINK = os.getenv("CALENDLY_SCHEDULING_LINK")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Winnipeg")

if not CALENDLY_API_TOKEN:
    raise RuntimeError("CALENDLY_API_TOKEN is not set in environment.")
if not CALENDLY_SCHEDULING_LINK:
    raise RuntimeError("CALENDLY_SCHEDULING_LINK is not set in environment.")

CALENDLY_BASE_URL = "https://api.calendly.com"

app = FastAPI()

# ------------------ Models for Vapi tools ------------------ #

class AvailabilityRequest(BaseModel):
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    preferred_time_of_day: Optional[str] = None


class TimeSlot(BaseModel):
    start: str
    end: str


class AvailabilityResponse(BaseModel):
    slots: List[TimeSlot]


class BookAppointmentRequest(BaseModel):
    start_time: str
    end_time: str
    invitee_name: str
    invitee_email: str


class BookAppointmentResponse(BaseModel):
    success: bool
    event_uri: Optional[str] = None
    message: Optional[str] = None


class CancelAppointmentRequest(BaseModel):
    event_uri: str


class CancelAppointmentResponse(BaseModel):
    success: bool
    message: Optional[str] = None


# ------------------ Helper functions ------------------ #

async def get_calendly_headers() -> dict:
    return {
        "Authorization": f"Bearer {CALENDLY_API_TOKEN}",
        "Content-Type": "application/json",
    }


# ------------------ Routes ------------------ #

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/check-availability", response_model=AvailabilityResponse)
async def check_availability(body: AvailabilityRequest):
    """
    TEMP simple availability logic:
    - If no date_start provided, use tomorrow (UTC)
    - Returns 4 half-hour slots between 1pm and 3pm on that date.

    This is just for the AI agent to have something to offer.
    You can later replace this with real availability logic.
    """
    today = datetime.utcnow().date()
    target_date = today + timedelta(days=1)

    if body.date_start:
        try:
            target_date = datetime.fromisoformat(body.date_start).date()
        except ValueError:
            # If parsing fails, keep default (tomorrow)
            pass

    slots: List[TimeSlot] = []
    start_hour = 13  # 1pm (UTC-ish, adjust later if needed)

    for i in range(4):
        slot_start = datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=start_hour + (i // 2),
            minute=30 * (i % 2),
        )
        slot_end = slot_start + timedelta(minutes=30)
        slots.append(
            TimeSlot(
                start=slot_start.isoformat() + "Z",
                end=slot_end.isoformat() + "Z",
            )
        )

    return AvailabilityResponse(slots=slots)


@app.post("/book-appointment", response_model=BookAppointmentResponse)
async def book_appointment(body: BookAppointmentRequest):
    """
    CURRENT VERSION (no direct Calendly booking):
    - Collects user info + requested time.
    - Returns your Calendly link so the caller can confirm the booking.
    - Vapi can send the link by SMS or read it out loud.

    Calendly's public API with a Personal Access Token does not support
    creating events, so we let the user finalize via the normal Calendly page.
    """
    message = (
        f"I've recorded your appointment request for {body.start_time} "
        f"for {body.invitee_name}. "
        f"To finalize the booking, please use this Calendly link: "
        f"{CALENDLY_SCHEDULING_LINK}"
    )

    return BookAppointmentResponse(
        success=True,
        event_uri=None,
        message=message,
    )


@app.post("/cancel-appointment", response_model=CancelAppointmentResponse)
async def cancel_appointment(body: CancelAppointmentRequest):
    """
    Cancels a Calendly scheduled event using the event_uri.

    Calendly's v2 API uses a dedicated Cancel Event endpoint:
    POST /scheduled_events/{uuid}/cancellation

    Body:
    {
      "cancellation": {
        "canceled_by": "string",
        "reason": "string"
      }
    }
    """

    event_uri = body.event_uri.rstrip("/")
    event_uuid = event_uri.split("/")[-1]

    if not event_uuid:
        raise HTTPException(
            status_code=400,
            detail="Invalid event_uri. Could not extract event UUID.",
        )

    payload = {
        "cancellation": {
            "canceled_by": "Host",  # or "Ashammeet Gill"
            "reason": "Cancelled via NexDirection AI assistant",
        }
    }

    async with httpx.AsyncClient(base_url=CALENDLY_BASE_URL) as client:
        resp = await client.post(
            f"/scheduled_events/{event_uuid}/cancellation",
            headers=await get_calendly_headers(),
            json=payload,
        )

    if resp.status_code in (200, 201, 204):
        return CancelAppointmentResponse(
            success=True,
            message="Appointment cancelled successfully.",
        )

    return CancelAppointmentResponse(
        success=False,
        message=f"Error cancelling appointment: {resp.status_code} {resp.text}",
    )

