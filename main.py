# main.py
import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ============================================================
# ENV & CONSTANTS
# ============================================================
load_dotenv()

DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "America/Winnipeg").strip()
TENANTS_FILE = Path(os.getenv("TENANTS_FILE", "./tenants.json"))
OBFUSCATE_KEYS = os.getenv("OBFUSCATE_KEYS", "false").lower() in {"1", "true", "yes"}

CAL_API_BASE = "https://api.cal.com/v2"
CAL_SLOTS_API_VERSION = "2024-09-04"      # required header for /v2/slots
CAL_BOOKINGS_API_VERSION = "2024-08-13"   # required header for /v2/bookings

app = FastAPI(title="NexDirection Multi-tenant Cal.com Backend")

# ============================================================
# MODELS
# ============================================================

class TenantUpsert(BaseModel):
    tenant_id: str = Field(..., description="Unique slug, e.g. 'sunrise-massage'")
    business_name: Optional[str] = None
    cal_api_key: str = Field(..., description="Cal.com API key (prefixed with cal_)")
    cal_team_id: Optional[int] = Field(None, description="Cal.com Team ID if the client uses teams")
    cal_event_types: str = Field(..., example="Deep Tissue:111,Relaxation:222")
    cal_default_service: Optional[str] = Field(None, description="Name from cal_event_types to use if none supplied")
    timezone: Optional[str] = Field(None, description="IANA TZ, e.g. America/Winnipeg")
    did_numbers: Optional[List[str]] = Field(default=None, description="Optional list of phone DIDs used to route calls")

class TenantPublic(BaseModel):
    tenant_id: str
    business_name: Optional[str] = None
    cal_team_id: Optional[int] = None
    cal_event_types: str
    cal_default_service: Optional[str] = None
    timezone: Optional[str] = None
    did_numbers: Optional[List[str]] = None

class AvailabilityRequest(BaseModel):
    date_start: Optional[str] = None  # YYYY-MM-DD or ISO; defaults to tomorrow if omitted
    date_end: Optional[str] = None
    preferred_time_of_day: Optional[str] = None
    service_name: Optional[str] = None

class TimeSlot(BaseModel):
    start: str
    end: str

class AvailabilityResponse(BaseModel):
    slots: List[TimeSlot]

class BookAppointmentRequest(BaseModel):
    start_time: str                    # preferred with offset e.g. 2025-12-01T10:00:00-06:00
    end_time: str                      # ignored by Cal.com (kept for Vapi compatibility)
    invitee_name: str
    invitee_phone: str                 # REQUIRED
    invitee_email: Optional[str] = None  # OPTIONAL (we synthesize if missing)
    service_name: Optional[str] = None

class BookAppointmentResponse(BaseModel):
    success: bool
    event_uri: Optional[str] = None
    message: Optional[str] = None

class CancelAppointmentRequest(BaseModel):
    uid: Optional[str] = None
    event_uri: Optional[str] = None           # full URI or uid at the end
    invitee_email: Optional[str] = None
    phone: Optional[str] = None
    cancellation_reason: Optional[str] = None
    cancel_subsequent_bookings: Optional[bool] = None  # for non-seated recurring
    seat_uid: Optional[str] = None                     # for seated bookings

class CancelAppointmentResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class FindAppointmentsRequest(BaseModel):
    invitee_email: str

class FindByPhoneRequest(BaseModel):
    phone: str

class AppointmentSummary(BaseModel):
    event_uri: str
    start_time: str
    end_time: Optional[str] = None
    status: Optional[str] = None
    event_name: Optional[str] = None

class FindAppointmentsResponse(BaseModel):
    events: List[AppointmentSummary]

class RescheduleRequest(BaseModel):
    # how to find the existing booking (priority: uid > event_uri > email > phone)
    uid: Optional[str] = None
    event_uri: Optional[str] = None
    invitee_email: Optional[str] = None
    phone: Optional[str] = None

    # the new time to move to (caller can say local time; we convert)
    new_start_time: str                    # e.g. "2025-12-03T15:00:00-06:00"
    service_name: Optional[str] = None     # optional override; if omitted we reuse old service

# ============================================================
# SIMPLE "DB" (JSON FILE) + HELPERS
# ============================================================

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()

def _unb64(s: str) -> str:
    return base64.b64decode(s.encode()).decode()

def _load_tenants() -> Dict[str, dict]:
    if not TENANTS_FILE.exists():
        return {}
    try:
        return json.loads(TENANTS_FILE.read_text())
    except Exception:
        return {}

def _save_tenants(data: Dict[str, dict]) -> None:
    TENANTS_FILE.write_text(json.dumps(data, indent=2))

TENANTS: Dict[str, dict] = _load_tenants()

def upsert_tenant(t: TenantUpsert) -> None:
    key = t.tenant_id.strip()
    stored = TENANTS.get(key, {})
    api_key_to_store = _b64(t.cal_api_key) if OBFUSCATE_KEYS else t.cal_api_key

    stored.update({
        "tenant_id": key,
        "business_name": t.business_name,
        "cal_api_key": api_key_to_store,
        "cal_team_id": t.cal_team_id,
        "cal_event_types": t.cal_event_types,
        "cal_default_service": t.cal_default_service,
        "timezone": t.timezone or DEFAULT_TIMEZONE,
        "did_numbers": t.did_numbers or [],
        "obfuscated": OBFUSCATE_KEYS
    })
    TENANTS[key] = stored
    _save_tenants(TENANTS)

def get_tenant_by_id(tenant_id: str) -> dict:
    t = TENANTS.get(tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"Unknown tenant '{tenant_id}'. Add it via /tenants/upsert.")
    return t

def get_tenant_by_did(did: str) -> dict:
    for t in TENANTS.values():
        if did in (t.get("did_numbers") or []):
            return t
    raise HTTPException(status_code=404, detail=f"No tenant mapped to DID '{did}'.")

def resolve_tenant(request: Request, x_tenant_id: Optional[str], x_did: Optional[str]) -> dict:
    # 1) Header X-Tenant-ID
    if x_tenant_id:
        return get_tenant_by_id(x_tenant_id.strip())
    # 2) Query ?tenant=
    q_tenant = request.query_params.get("tenant")
    if q_tenant:
        return get_tenant_by_id(q_tenant.strip())
    # 3) Header X-DID
    if x_did:
        return get_tenant_by_did(x_did.strip())
    # 4) Query ?did=
    q_did = request.query_params.get("did")
    if q_did:
        return get_tenant_by_did(q_did.strip())
    raise HTTPException(status_code=400, detail="No tenant specified. Send X-Tenant-ID or ?tenant= (or DID via X-DID / ?did=).")

def service_map_str_to_dict(raw: str) -> Dict[str, str]:
    """
    'Deep Tissue:111,Relaxation:222' -> {'Deep Tissue': '111', 'Relaxation': '222'}
    """
    out: Dict[str, str] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, id_str = part.split(":", 1)
        name, id_str = name.strip(), id_str.strip()
        if name and id_str.isdigit():
            out[name] = id_str
    return out

def resolve_event_type(tenant: dict, service_name: Optional[str]) -> Tuple[str, str]:
    mapping = service_map_str_to_dict(tenant.get("cal_event_types", ""))
    if not mapping:
        raise HTTPException(status_code=400, detail="Tenant has no CAL event types configured.")
    default_name = tenant.get("cal_default_service") or (list(mapping.keys())[0])
    chosen = (service_name or default_name).strip()
    etid = mapping.get(chosen)
    if not etid:
        for k, v in mapping.items():
            if k.lower() == chosen.lower():
                etid = v
                chosen = k
                break
    if not etid:
        raise HTTPException(status_code=400, detail=f"Unknown service '{service_name}'. Available: {', '.join(mapping.keys())}")
    return etid, chosen

def auth_headers_for(tenant: dict) -> Dict[str, str]:
    raw_key = tenant.get("cal_api_key", "")
    if tenant.get("obfuscated"):
        raw_key = _unb64(raw_key)
    if not raw_key:
        raise HTTPException(status_code=400, detail="Tenant has no Cal.com API key configured.")
    return {
        "Authorization": f"Bearer {raw_key}",
        "Content-Type": "application/json"
    }

def as_date(iso_or_date: Optional[str]) -> datetime.date:
    if not iso_or_date:
        return (datetime.now(timezone.utc) + timedelta(days=1)).date()
    try:
        if len(iso_or_date) == 10:
            return datetime.strptime(iso_or_date, "%Y-%m-%d").date()
        dt = datetime.fromisoformat(iso_or_date.replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        return (datetime.now(timezone.utc) + timedelta(days=1)).date()

def to_utc_z(iso_str: str) -> str:
    """
    Convert '2025-12-01T10:00:00-06:00' -> '2025-12-01T16:00:00Z'
    If input has no offset, assume DEFAULT_TIMEZONE (not recommended).
    """
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TIMEZONE))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        d = datetime.strptime(iso_str[:10], "%Y-%m-%d").date()
        return datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "provider": "cal.com", "tenants": list(TENANTS.keys())}

# ------------------ ADMIN: Tenants ------------------ #

@app.post("/tenants/upsert")
async def tenants_upsert(body: TenantUpsert):
    upsert_tenant(body)
    t = get_tenant_by_id(body.tenant_id)
    public = TenantPublic(
        tenant_id=t["tenant_id"],
        business_name=t.get("business_name"),
        cal_team_id=t.get("cal_team_id"),
        cal_event_types=t.get("cal_event_types", ""),
        cal_default_service=t.get("cal_default_service"),
        timezone=t.get("timezone"),
        did_numbers=t.get("did_numbers") or []
    )
    return {"ok": True, "tenant": public}

@app.get("/tenants")
async def tenants_list():
    items = []
    for t in TENANTS.values():
        items.append(TenantPublic(
            tenant_id=t["tenant_id"],
            business_name=t.get("business_name"),
            cal_team_id=t.get("cal_team_id"),
            cal_event_types=t.get("cal_event_types", ""),
            cal_default_service=t.get("cal_default_service"),
            timezone=t.get("timezone"),
            did_numbers=t.get("did_numbers") or []
        ).dict())
    return {"tenants": items}

# ------------------ PUBLIC: Services for a Tenant ------------------ #

@app.get("/services")
async def list_services(
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)
    mapping = service_map_str_to_dict(tenant.get("cal_event_types", ""))
    default_name = tenant.get("cal_default_service") or (list(mapping.keys())[0] if mapping else None)
    return {
        "tenant": tenant["tenant_id"],
        "services": [{"name": k, "id": v} for k, v in mapping.items()],
        "default": default_name
    }

# ============================================================
# UTIL
# ============================================================

@app.get("/now")
async def now_endpoint(
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    t = resolve_tenant(request, x_tenant_id, x_did)
    tz = t.get("timezone") or DEFAULT_TIMEZONE
    dt = datetime.now(ZoneInfo(tz))
    return {
        "tenant": t["tenant_id"],
        "timezone": tz,
        "iso": dt.isoformat(),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M"),
        "weekday": dt.strftime("%A")
    }

# ============================================================
# VAPI-COMPATIBLE BOOKING ROUTES (Cal.com)
# ============================================================

@app.post("/check-availability", response_model=AvailabilityResponse)
async def check_availability(
    body: AvailabilityRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)
    event_type_id, _ = resolve_event_type(tenant, body.service_name)

    # Cal.com /v2/slots requires start & end (UTC ISO date or datetime).
    start_date = as_date(body.date_start)
    end_date = as_date(body.date_end) if body.date_end else (start_date + timedelta(days=1))

    start_iso = start_date.strftime("%Y-%m-%d")
    end_iso = end_date.strftime("%Y-%m-%d")

    tz = tenant.get("timezone") or DEFAULT_TIMEZONE

    params = {
        "eventTypeId": int(event_type_id),
        "start": start_iso,
        "end": end_iso,
        "timeZone": tz,
        "format": "range",
    }
    if tenant.get("cal_team_id"):
        params["teamId"] = str(tenant["cal_team_id"])  # <-- correct param for v2

    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_SLOTS_API_VERSION
    }

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.get("/slots", params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=f"Cal.com availability error: {r.text}")
        raw = r.json().get("data", {})

    # Parse returned map {date: [{start, end}, ...], ...}
    slots: List[TimeSlot] = []
    for _, entries in raw.items():
        for s in entries:
            if "start" in s and "end" in s:
                slots.append(TimeSlot(start=s["start"], end=s["end"]))

    return AvailabilityResponse(slots=slots)

@app.post("/book-appointment", response_model=BookAppointmentResponse)
async def book_appointment(
    body: BookAppointmentRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)
    event_type_id, resolved_name = resolve_event_type(tenant, body.service_name)

    # ---- Normalize phone to E.164 (+1…) so caller never has to say "+1"
    phone_raw = (body.invitee_phone or "").strip()
    digits = "".join(ch for ch in phone_raw if ch.isdigit())
    if not digits:
        raise HTTPException(status_code=400, detail="Phone number is required.")
    phone_e164 = f"+{digits}" if digits.startswith("1") else f"+1{digits}"

    # ---- Synthesize email if missing
    invitee_email = (body.invitee_email or f"{(digits[-10:] or 'client')}@nexdirection.local").lower()

    # Cal.com booking requires UTC Z time
    start_utc_z = to_utc_z(body.start_time)
    tz = tenant.get("timezone") or DEFAULT_TIMEZONE

    payload = {
        "start": start_utc_z,
        "eventTypeId": int(event_type_id),
        "attendee": {
            "name": body.invitee_name,
            "email": invitee_email,
            "timeZone": tz,
            "phoneNumber": phone_e164  # normalized
        },
        "metadata": {
            "source": "vapi-voice",
            "serviceName": resolved_name
        }
        # Optionals you can add: lengthInMinutes, location, guests, etc.
    }

    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_BOOKINGS_API_VERSION
    }

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.post("/bookings", json=payload)
        if r.status_code >= 400:
            return BookAppointmentResponse(
                success=False,
                message=f"Cal.com booking error: {r.status_code} {r.text}"
            )
        resp = r.json()
        data = resp.get("data", resp)

    booking_uid = str(data.get("uid") or data.get("id") or "")
    start = data.get("start") or start_utc_z
    label = f" ({resolved_name})" if resolved_name else ""
    return BookAppointmentResponse(
        success=True,
        event_uri=f"{CAL_API_BASE}/bookings/{booking_uid}" if booking_uid else None,
        message=f"Booked{label} for {start} under {body.invitee_name}."
    )

# ============================================================
# INTERNAL HELPERS (for find/cancel/reschedule)
# ============================================================

async def _fetch_upcoming_for_tenant(tenant: dict) -> List[dict]:
    now_utc = datetime.now(timezone.utc)
    after_start = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    before_end = (now_utc + timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "status": "upcoming,unconfirmed",
        "afterStart": after_start,
        "beforeEnd": before_end,
        "take": 100,
        "skip": 0,
    }
    if tenant.get("cal_team_id"):
        params["teamId"] = str(tenant["cal_team_id"])

    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_BOOKINGS_API_VERSION,
    }

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.get("/bookings", params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=f"Cal.com list error: {r.text}")
        resp = r.json()
        return resp.get("data", [])

def _soonest(events: List[dict]) -> Optional[dict]:
    def _to_dt(e: dict) -> datetime:
        try:
            return datetime.fromisoformat((e.get("start") or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)
    if not events:
        return None
    return sorted(events, key=_to_dt)[0]

# ============================================================
# CANCELLATION (auto-resolves by uid, event_uri, email, or phone)
# ============================================================

@app.post("/cancel-appointment", response_model=CancelAppointmentResponse)
async def cancel_appointment(
    body: CancelAppointmentRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)
    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_BOOKINGS_API_VERSION,
    }

    # 1) Resolve the booking UID
    booking_uid = (body.uid or "").strip()
    if not booking_uid and body.event_uri:
        booking_uid = body.event_uri.rstrip("/").split("/")[-1].strip()

    target_event: Optional[dict] = None

    # 2) If still missing, try resolve by invitee_email (soonest upcoming)
    if not booking_uid and body.invitee_email:
        all_events = await _fetch_upcoming_for_tenant(tenant)
        email_lower = body.invitee_email.strip().lower()
        email_matches = []
        for ev in all_events:
            attendees = ev.get("attendees") or []
            for a in attendees:
                if (a.get("email") or "").strip().lower() == email_lower:
                    email_matches.append(ev)
                    break
        target_event = _soonest(email_matches)
        if target_event:
            booking_uid = str(target_event.get("uid") or target_event.get("id") or "").strip()

    # 3) If still missing, try resolve by phone suffix (soonest upcoming)
    if not booking_uid and body.phone:
        all_events = await _fetch_upcoming_for_tenant(tenant)
        wanted = "".join(ch for ch in body.phone if ch.isdigit())
        phone_matches = []
        for ev in all_events:
            attendees = ev.get("attendees") or []
            for a in attendees:
                pn = (a.get("phoneNumber") or "")
                pn_norm = "".join(ch for ch in pn if ch.isdigit())
                if pn_norm and wanted and pn_norm.endswith(wanted):
                    phone_matches.append(ev)
                    break
        target_event = _soonest(phone_matches)
        if target_event:
            booking_uid = str(target_event.get("uid") or target_event.get("id") or "").strip()

    if not booking_uid:
        return CancelAppointmentResponse(success=False, message="No matching booking found to cancel.")

    # 4) Build cancel payload
    cancel_body: Dict[str, object] = {}
    if body.seat_uid:
        cancel_body["seatUid"] = body.seat_uid
    if body.cancellation_reason:
        cancel_body["cancellationReason"] = body.cancellation_reason
    if body.cancel_subsequent_bookings is not None:
        cancel_body["cancelSubsequentBookings"] = body.cancel_subsequent_bookings

    # 5) Call Cal.com
    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.post(f"/bookings/{booking_uid}/cancel", json=cancel_body or None)
        if r.status_code >= 400:
            return CancelAppointmentResponse(success=False, message=f"Cal.com cancel error: {r.status_code} {r.text}")

    return CancelAppointmentResponse(success=True, message="Appointment cancelled.")

# ============================================================
# LOOKUPS (email / phone)
# ============================================================

@app.post("/find-appointments-by-email", response_model=FindAppointmentsResponse)
async def find_appointments_by_email(
    body: FindAppointmentsRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)

    params = {
        "attendeeEmail": body.invitee_email,
        "status": "upcoming,unconfirmed",
        "take": 100,
        "skip": 0,
    }
    if tenant.get("cal_team_id"):
        params["teamId"] = str(tenant["cal_team_id"])

    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_BOOKINGS_API_VERSION,
    }

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.get("/bookings", params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=f"Cal.com list error: {r.text}")
        resp = r.json()
        items = resp.get("data", [])  # per docs

    events: List[AppointmentSummary] = []
    for ev in items:
        status = (ev.get("status") or "").lower()
        if status == "cancelled":
            continue
        bid = ev.get("uid") or ev.get("id")
        if not bid:
            continue
        events.append(AppointmentSummary(
            event_uri=f"{CAL_API_BASE}/bookings/{bid}",
            start_time=ev.get("start"),
            end_time=ev.get("end"),
            status=ev.get("status"),
            event_name=(ev.get("title") or (ev.get("eventType") or {}).get("slug"))
        ))
    return FindAppointmentsResponse(events=events)

@app.post("/find-appointments-by-phone", response_model=FindAppointmentsResponse)
async def find_appointments_by_phone(
    body: FindByPhoneRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)

    # Strategy: pull upcoming + unconfirmed (next 90 days) and filter locally by phone suffix.
    now_utc = datetime.now(timezone.utc)
    after_start = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    before_end = (now_utc + timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "status": "upcoming,unconfirmed",
        "afterStart": after_start,
        "beforeEnd": before_end,
        "take": 100,
        "skip": 0,
    }
    if tenant.get("cal_team_id"):
        params["teamId"] = str(tenant["cal_team_id"])

    headers = {
        **auth_headers_for(tenant),
        "cal-api-version": CAL_BOOKINGS_API_VERSION,
    }

    wanted = "".join(ch for ch in body.phone if ch.isdigit())

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r = await client.get("/bookings", params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=f"Cal.com list error: {r.text}")
        resp = r.json()
        items = resp.get("data", [])

    events: List[AppointmentSummary] = []
    for ev in items:
        status = (ev.get("status") or "").lower()
        if status == "cancelled":
            continue

        attendees = ev.get("attendees") or []
        matched = False
        for a in attendees:
            pn = (a.get("phoneNumber") or "")
            if not pn:
                continue
            pn_norm = "".join(ch for ch in pn if ch.isdigit())
            if pn_norm and wanted and pn_norm.endswith(wanted):
                matched = True
                break
        if not matched:
            continue

        bid = ev.get("uid") or ev.get("id")
        if not bid:
            continue
        events.append(AppointmentSummary(
            event_uri=f"{CAL_API_BASE}/bookings/{bid}",
            start_time=ev.get("start"),
            end_time=ev.get("end"),
            status=ev.get("status"),
            event_name=(ev.get("title") or (ev.get("eventType") or {}).get("slug"))
        ))
    return FindAppointmentsResponse(events=events)

# ============================================================
# RESCHEDULE (book-new-first, then cancel old; rollback on failure)
# ============================================================

@app.post("/reschedule-appointment")
async def reschedule_appointment(
    body: RescheduleRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None),
    x_did: Optional[str] = Header(default=None)
):
    tenant = resolve_tenant(request, x_tenant_id, x_did)
    headers = { **auth_headers_for(tenant), "cal-api-version": CAL_BOOKINGS_API_VERSION }

    # ---- 1) Resolve target booking (uid > event_uri > email > phone)
    booking_uid = (body.uid or "").strip()
    target_event: Optional[dict] = None

    if not booking_uid and body.event_uri:
        booking_uid = body.event_uri.rstrip("/").split("/")[-1].strip()

    if not booking_uid and (body.invitee_email or body.phone):
        all_events = await _fetch_upcoming_for_tenant(tenant)

        if body.invitee_email and not target_event:
            email_lower = body.invitee_email.strip().lower()
            email_matches = []
            for ev in all_events:
                for a in (ev.get("attendees") or []):
                    if (a.get("email") or "").strip().lower() == email_lower:
                        email_matches.append(ev); break
            target_event = _soonest(email_matches)

        if body.phone and not target_event:
            wanted = "".join(ch for ch in body.phone if ch.isdigit())
            phone_matches = []
            for ev in all_events:
                for a in (ev.get("attendees") or []):
                    pn = "".join(ch for ch in (a.get("phoneNumber") or "") if ch.isdigit())
                    if pn and wanted and pn.endswith(wanted):
                        phone_matches.append(ev); break
            target_event = _soonest(phone_matches)

        if target_event:
            booking_uid = str(target_event.get("uid") or target_event.get("id") or "").strip()

    if not booking_uid:
        raise HTTPException(status_code=404, detail="No matching booking found to reschedule.")

    if not target_event:
        all_events = await _fetch_upcoming_for_tenant(tenant)
        for ev in all_events:
            if str(ev.get("uid") or ev.get("id") or "") == booking_uid:
                target_event = ev; break
    if not target_event:
        raise HTTPException(status_code=404, detail="Unable to load booking to reschedule.")

    # ---- 2) Determine eventTypeId (prefer existing)
    evtype_obj = target_event.get("eventType") or {}
    evtype_id = evtype_obj.get("id")
    if not evtype_id:
        if not body.service_name:
            raise HTTPException(status_code=400, detail="Cannot infer service from booking; please provide service_name.")
        evtype_id, _ = resolve_event_type(tenant, body.service_name)

    # ---- 3) Reuse attendee details
    attendees = target_event.get("attendees") or []
    primary = attendees[0] if attendees else {}
    invitee_name = primary.get("name") or "Guest"
    phoneNumber = primary.get("phoneNumber") or ""
    email_raw = (primary.get("email") or "").lower()

    # synthesize email if missing
    digits_from_existing = "".join(ch for ch in phoneNumber if ch.isdigit())
    email_final = email_raw or f"{(digits_from_existing[-10:] or 'client')}@nexdirection.local"

    # normalize phone if present
    phone_e164 = None
    if digits_from_existing:
        phone_e164 = f"+{digits_from_existing}" if digits_from_existing.startswith("1") else f"+1{digits_from_existing}"

    # ---- 4) Book the new slot FIRST
    start_utc_z = to_utc_z(body.new_start_time)
    tz = tenant.get("timezone") or DEFAULT_TIMEZONE
    book_payload = {
        "start": start_utc_z,
        "eventTypeId": int(evtype_id),
        "attendee": {
            "name": invitee_name,
            "email": email_final,
            "timeZone": tz,
            **({"phoneNumber": phone_e164} if phone_e164 else {})
        },
        "metadata": {"source": "vapi-voice", "rescheduleFrom": booking_uid}
    }

    async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
        r_new = await client.post("/bookings", json=book_payload)
        if r_new.status_code >= 400:
            raise HTTPException(status_code=r_new.status_code, detail=f"Cal.com booking error: {r_new.text}")
        data_new = (r_new.json().get("data") or {})

    new_uid = str(data_new.get("uid") or data_new.get("id") or "").strip()
    if not new_uid:
        raise HTTPException(status_code=502, detail="Booked new time but no uid returned by provider.")

    # ---- 5) Cancel the old booking (rollback on failure)
    try:
        async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
            r_cancel = await client.post(f"/bookings/{booking_uid}/cancel", json={"cancellationReason": "Reschedule to new time"})
            if r_cancel.status_code >= 400:
                # rollback: cancel the new one we just created
                await client.post(f"/bookings/{new_uid}/cancel", json={"cancellationReason": "Rollback after failed reschedule cancel"})
                raise HTTPException(status_code=r_cancel.status_code, detail=f"Cal.com cancel error: {r_cancel.text}")
    except Exception:
        async with httpx.AsyncClient(base_url=CAL_API_BASE, headers=headers, timeout=30) as client:
            await client.post(f"/bookings/{new_uid}/cancel", json={"cancellationReason": "Rollback after unexpected error"})
        raise

    return {
        "success": True,
        "old_booking_uid": booking_uid,
        "new_booking_uri": f"{CAL_API_BASE}/bookings/{new_uid}",
        "new_start": data_new.get("start") or start_utc_z
    }
