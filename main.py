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

# ------------------ Models ------------------ #

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


# ---- New Models for Appointment Lookup ---- #

class FindAppointmentsRequest(BaseModel):
    invitee_email: str


class AppointmentSummary(BaseModel):
    event_uri: str
    start_time: str
    end_time: str
    status: Optional[str] = None
    event_name: Optional[str] = None


class FindAppointmentsResponse(BaseModel):
    events: List[AppointmentSummary]


# ------------------ Helper ------------------ #

async def get_calendly_headers() -> dict:
    return {
        "Authorization": f"Bearer {CALENDLY_API_TOKEN}",
        "Content-Type": "application/json",
    }


# ------------------ Routes ------------------ #

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- Check Availability (temporary logic) ---- #

@app.post("/check-availability", response_model=AvailabilityResponse)
async def check_availability(body: AvailabilityRequest):
    today = datetime.utcnow().date()
    target_date = today + timedelta(days=1)

    if body.date_start:
        try:
            target_date = datetime.fromisoformat(body.date_start).date()
        except:
            pass

    slots: List[TimeSlot] = []
    start_hour = 13  # 1pm UTC-ish

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


# ---- Book Appointment ---- #

@app.post("/book-appointment", response_model=BookAppointmentResponse)
async def book_appointment(body: BookAppointmentRequest):
    message = (
        f"I've recorded your appointment request for {body.start_time} "
        f"for {body.invitee_name}. "
        f"To finalize the booking, please use this link: "
        f"{CALENDLY_SCHEDULING_LINK}"
    )

    return BookAppointmentResponse(
        success=True,
        event_uri=None,
        message=message,
    )


# ---- Cancel Appointment ---- #

@app.post("/cancel-appointment", response_model=CancelAppointmentResponse)
async def cancel_appointment(body: CancelAppointmentRequest):
    event_uri = body.event_uri.rstrip("/")
    event_uuid = event_uri.split("/")[-1]

    if not event_uuid:
        raise HTTPException(
            status_code=400,
            detail="Invalid event_uri. Could not extract UUID.",
        )

    payload = {
        "cancellation": {
            "canceled_by": "Host",
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
            message="Appointment cancelled successfully."
        )

    return CancelAppointmentResponse(
        success=False,
        message=f"Error cancelling appointment: {resp.status_code} {resp.text}",
    )


# ---- NEW: Find Appointments By Email ---- #

@app.post("/find-appointments-by-email", response_model=FindAppointmentsResponse)
async def find_appointments_by_email(body: FindAppointmentsRequest):
    """
    Lookup Calendly events for a given invitee email.
    This is the recommended method by Calendly for identifying appointments.
    """

    headers = await get_calendly_headers()

    async with httpx.AsyncClient(base_url=CALENDLY_BASE_URL) as client:

        # 1) Get organization URI
        user_resp = await client.get("/users/me", headers=headers)
        if user_resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching Calendly user: {user_resp.text}",
            )

        user_data = user_resp.json()
        org_uri = user_data["resource"]["current_organization"]

        # 2) Get scheduled events filtered by email
        params = {
            "organization": org_uri,
            "invitee_email": body.invitee_email,
            "sort": "start_time:asc",
        }

        events_resp = await client.get("/scheduled_events", headers=headers, params=params)
        if events_resp.status_code != 200:
            raise HTTPException(
                status_code=events_resp.status_code,
                detail=f"Error fetching events: {events_resp.text}",
            )

        data = events_resp.json()

        events: List[AppointmentSummary] = []

        for ev in data.get("collection", []):
            # Only return active/upcoming events
            if ev.get("status") != "active":
                continue

            events.append(
                AppointmentSummary(
                    event_uri=ev["uri"],
                    start_time=ev["start_time"],
                    end_time=ev["end_time"],
                    status=ev.get("status"),
                    event_name=ev.get("name"),
                )
            )

    return FindAppointmentsResponse(events=events)
