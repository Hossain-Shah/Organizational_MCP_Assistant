import asyncio
from graph.workflow import app

# Define required entities for room booking
ROOM_BOOKING_REQUIRED = ["Room_ID", "Pickup_time", "Drop_time", "Pickup_date", "Participant", "Meeting_purpose"]
VEHICLE_BOOKING_REQUIRED = ["Pickup_point", "Pickup_time", "Drop_time", "Pickup_date", "Drop_date", "Participant", "Destination"]
LEAVE_REQUEST_REQUIRED = ["Drop_date", "Leave_initiate", "Leave_terminate", "Pickup_date", "Leave_type", "Leave_reason"]

async def main():
    session_state = {
        "current_intent": None,
        "entities": {},
        "missing_entities": []
    }

    print("Bot: Hi there! How can I help you today?")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["bye", "exit", "quit", "see you later"]:
            print("Bot: See you later, thanks for visiting")
            break

        # Pass user input to your workflow
        state = {"user_input": user_input}
        result = await app.ainvoke(state)
        response = result.get("response", "Sorry, I didn't understand that.")

        # Determine current intent (session-aware)
        task_intents = result.get("task_intents", {})  # make sure your workflow returns this
        TASK_CONF_THRESHOLD = 0.9
        if task_intents:
            best_task_intent = max(task_intents, key=task_intents.get, default=None)
            if best_task_intent and task_intents[best_task_intent] >= TASK_CONF_THRESHOLD:
                session_state["current_intent"] = best_task_intent
            else:
                session_state["current_intent"] = "IRRELEVANT"

        # Handle Room Booking multi-turn
        if session_state["current_intent"] == "Room Booking System":
            entities = result.get("entities", {})
            session_state["entities"].update(entities)

            session_state["missing_entities"] = [
                e for e in ROOM_BOOKING_REQUIRED if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                next_entity = session_state["missing_entities"][0]
                print(f"Bot: Please provide {next_entity.replace('_', ' ')}.")
            else:
                print("Bot: Room booking request received:", session_state["entities"])
                session_state = {"current_intent": None, "entities": {}, "missing_entities": []}

        # Handle Vehicle Booking multi-turn
        elif session_state["current_intent"] == "VMS":
            entities = result.get("entities", {})
            session_state["entities"].update(entities)

            session_state["missing_entities"] = [
                e for e in VEHICLE_BOOKING_REQUIRED if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                next_entity = session_state["missing_entities"][0]
                print(f"Bot: Please provide {next_entity.replace('_', ' ')}.")
            else:
                print("Bot: Vehicle booking request received:", session_state["entities"])
                session_state = {"current_intent": None, "entities": {}, "missing_entities": []}

        # Handle Leave Request multi-turn
        elif session_state["current_intent"] == "HR":
            entities = result.get("entities", {})
            session_state["entities"].update(entities)

            session_state["missing_entities"] = [
                e for e in LEAVE_REQUEST_REQUIRED if e not in session_state["entities"]
            ]

            if session_state["missing_entities"]:
                next_entity = session_state["missing_entities"][0]
                print(f"Bot: Please provide {next_entity.replace('_', ' ')}.")
            else:
                print("Bot: Leave request received:", session_state["entities"])
                session_state = {"current_intent": None, "entities": {}, "missing_entities": []}

        else:
            # For other intents, just respond normally
            print("Bot:", response)

if __name__ == "__main__":
    asyncio.run(main())
