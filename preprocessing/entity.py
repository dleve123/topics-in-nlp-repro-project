from typing import List, Tuple
import stanza

stanza.download('en')
nlp = stanza.Pipeline('en')

EntityList = List[Tuple[str, str]]

def extract_entities(text) -> EntityList:
    """
    Given a text, return entities with type
    """
    doc = nlp(text)
    return [(ent.type, ent.text) for ent in doc.entities]


def entity_overlap(ents1: EntityList, ents2: EntityList):
    """
        Returns overlapping entities for two entity lists
    """
    return set(ents1).intersection(set(ents2))

if __name__ == "__main__":
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    print(extract_entities(doc))