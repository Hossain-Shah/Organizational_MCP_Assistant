class ResponseFFN:
    def generate(self, intent, entities):

        if intent == "Room Booking System":
            return f"Room booking request received: {entities}"

        if intent == "VMS":
            return f"Vehicle booking request received: {entities}"

        if intent == "HR":
            return f"Leave request received: {entities}"

        return "I couldn't understand your request."
