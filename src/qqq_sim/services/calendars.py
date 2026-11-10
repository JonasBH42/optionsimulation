"""Load explicit local calendar records without silently guessing fixing times."""
from __future__ import annotations
from datetime import date,datetime
from pathlib import Path
import json
from qqq_sim.replay.calendar import CalendarService, SessionRecord, CalendarError, local_datetime


def load_calendar(path=None):
    if path is None:
        return CalendarService()
    raw=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(raw,dict):
        raw=raw["sessions"]
    rows=[]
    for record in raw:
        row=dict(record)
        row["local_date"]=date.fromisoformat(row["local_date"])
        for name in ("exchange_open","underlying_close","option_close","fixing_at","exercise_instruction_cutoff"):
            if row.get(name) is not None:
                stamp=datetime.fromisoformat(row[name])
                if stamp.tzinfo is None:
                    raise CalendarError("CALENDAR_TIMESTAMP_UNVERIFIED",f"{name} requires an explicit timezone offset")
                row[name]=local_datetime(stamp)
        if not row["exchange_open"]<row["underlying_close"]<=row["option_close"]:
            raise CalendarError("INVALID_SESSION_INTERVAL",str(row["local_date"]))
        if any(row[name].date()!=row["local_date"] for name in ("exchange_open","underlying_close","option_close")):
            raise CalendarError("SESSION_DATE_MISMATCH",str(row["local_date"]))
        if row.get("status","OPEN") not in ("OPEN","CLOSED"):
            raise CalendarError("INVALID_SESSION_STATUS",str(row.get("status")))
        if "scenario_flag" in row and not isinstance(row["scenario_flag"],bool):
            raise CalendarError("INVALID_SCENARIO_FLAG","scenario_flag must be boolean")
        rows.append(SessionRecord(**row))
    if len({row.local_date for row in rows})!=len(rows):
        raise CalendarError("DUPLICATE_SESSION_RECORD","Each local date must have one calendar record")
    return CalendarService(rows)
