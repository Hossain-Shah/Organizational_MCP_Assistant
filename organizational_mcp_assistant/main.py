import asyncio
from graph.workflow import app
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from graph.nodes import NER_SERVER, call_tool_json
from org_rag.agent import org_rag_agent
import re

# ---------------- REQUIRED ENTITIES ---------------- #

ROOM_BOOKING_REQUIRED = [
    "Room_ID", "Pickup_time", "Drop_time",
    "Pickup_date", "Participant", "Meeting_purpose"
]

VEHICLE_BOOKING_REQUIRED = [
    "Pickup_point", "Pickup_time", "Drop_time",
    "Pickup_date", "Drop_date", "Participant", "Destination"
]

LEAVE_REQUEST_REQUIRED = [
    "Pickup_date", "Drop_date", "Leave_type",
    "Leave_reason", "Leave_initiate", "Leave_terminate"
]

def build_generic_cluster_records(user_input: str):
    text = user_input.strip()

    if "generic cluster" not in text.lower():
        return None

    student_match = re.search(r"student\s+([A-Za-z0-9_-]+)", text, re.I)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    status_match = re.search(r"attendance_status\s+([01])", text, re.I)
    school_match = re.search(r"school\s+([A-Za-z0-9_-]+)", text, re.I)
    country_match = re.search(r"country\s+([A-Za-z ]+)", text, re.I)

    if not student_match or not date_match or not status_match:
        return []

    return [
        {
            "student_id": student_match.group(1),
            "attendance_date": date_match.group(0),
            "attendance_status": int(status_match.group(1)),
            "absence_reason": None,
            "school_id": school_match.group(1) if school_match else None,
            "year": int(date_match.group(0)[:4]),
            "month": int(date_match.group(0)[5:7]),
            "country_name": country_match.group(1).strip() if country_match else None,
            "academic_year": date_match.group(0)[:4],
        }
    ]

CONF_THRESHOLD = 0.8


# ---------------- DIRECT NER (NO GRAPH) ---------------- #

async def extract_entities_directly(text: str) -> dict:
    async with stdio_client(NER_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            return await call_tool_json(
                session,
                "extract_entities",
                {"text": text}
            )


# ---------------- MAIN LOOP ---------------- #

async def main():
    session_state = {
        "current_intent": None,
        "entities": {},
        "missing_entities": [],
        "expected_entity": None
    }

    print("Bot: Hi there! How can I help you today?")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"bye", "exit", "quit"}:
            print("Bot: See you later!")
            break

        # 🔒 HARD LOCK — skip graph if already in task
                # 🔒 HARD LOCK — skip graph if already in task
        if session_state["current_intent"] in {
            "Room Booking System", "VMS", "HR"
        }:
            result = {}
        else:
            generic_cluster_records = build_generic_cluster_records(user_input)

            payload = {"user_input": user_input}

            if generic_cluster_records is not None:
                payload["generic_cluster_records"] = generic_cluster_records

            result = await app.ainvoke(payload)
            intent = result.get("intent")

            if intent == "GENERIC_CLUSTER":
                response = result.get("response")

                if response:
                    print("Bot:", response)
                else:
                    print("Bot: Generic cluster prediction could not be completed.")

                continue

            if intent in {"Room Booking System", "VMS", "HR"}:
                session_state["current_intent"] = intent

            elif result.get("response"):
                print("Bot:", result["response"])
                continue

            else:
                response = await org_rag_agent(user_input)
                print("Bot:", response)
                continue

        intent = session_state["current_intent"]

        # ---------------- ROOM BOOKING ---------------- #
        if intent == "Room Booking System":

            entities = await extract_entities_directly(user_input)
            expected = session_state.get("expected_entity")

            if expected and expected in entities:
                session_state["entities"][expected] = entities[expected]
                session_state["expected_entity"] = None

            session_state["missing_entities"] = [
                e for e in ROOM_BOOKING_REQUIRED
                if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                e = session_state["missing_entities"][0]
                session_state["expected_entity"] = e
                print(f"Bot: Please provide {e.replace('_', ' ')}.")

            else:
                print("Bot: Room booking request received:")
                print(session_state["entities"])

                session_state = {
                    "current_intent": None,
                    "entities": {},
                    "missing_entities": [],
                    "expected_entity": None
                }

        # ---------------- VEHICLE BOOKING ---------------- #
        elif intent == "VMS":

            entities = await extract_entities_directly(user_input)
            expected = session_state.get("expected_entity")

            if expected and expected in entities:
                session_state["entities"][expected] = entities[expected]
                session_state["expected_entity"] = None

            session_state["missing_entities"] = [
                e for e in VEHICLE_BOOKING_REQUIRED
                if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                e = session_state["missing_entities"][0]
                session_state["expected_entity"] = e
                print(f"Bot: Please provide {e.replace('_', ' ')}.")

            else:
                print("Bot: Vehicle booking request received:")
                print(session_state["entities"])

                session_state = {
                    "current_intent": None,
                    "entities": {},
                    "missing_entities": [],
                    "expected_entity": None
                }

        # ---------------- LEAVE REQUEST ---------------- #
        elif intent == "HR":

            entities = await extract_entities_directly(user_input)
            expected = session_state.get("expected_entity")

            if expected and expected in entities:
                session_state["entities"][expected] = entities[expected]
                session_state["expected_entity"] = None

            session_state["missing_entities"] = [
                e for e in LEAVE_REQUEST_REQUIRED
                if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                e = session_state["missing_entities"][0]
                session_state["expected_entity"] = e
                print(f"Bot: Please provide {e.replace('_', ' ')}.")

            else:
                print("Bot: Leave request received:")
                print(session_state["entities"])

                session_state = {
                    "current_intent": None,
                    "entities": {},
                    "missing_entities": [],
                    "expected_entity": None
                }

        # ---------------- GENERAL / CHAT ---------------- #
        else:
            response = result.get("response")
            if response:
                print("Bot:", response)
            else:
                print("Bot: Could you please clarify?")


if __name__ == "__main__":
    asyncio.run(main())
