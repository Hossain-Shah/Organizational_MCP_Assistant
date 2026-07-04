import spacy

class CustomNER:
    def __init__(self):
        self.nlp = spacy.load("models/artifacts/booking_ner_finetuned_model/model-best")

    def extract(self, text: str):
        doc = self.nlp(text)
        return {ent.label_: ent.text for ent in doc.ents}
