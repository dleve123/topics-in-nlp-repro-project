from preprocessing.entity import extract_entities
from preprocessing.entity import entity_overlap

source = """
He was re-elected for a second term by the UN General Assembly, \
unopposed and unanimously, on 21 \
June 2011, with effect from 1 January 2012. Mr. Ban \
describes his priorities as mobilising world leaders to deal \
with climate change, economic upheaval, pandemics and \
increasing pressures involving food, energy and water
"""
summary = """
The United Nations Secretary-General Ban Ki-moon was elected \
for a second term in 21 June 2011.
"""

def test_entity_extraction():
    entities_source = extract_entities(source)
    assert entities_source[0] == ('ORDINAL', 'second')
    assert entities_source[1] == ('ORG', 'the UN General Assembly')
    entities_summary = extract_entities(summary)
    assert entities_summary[0] == ('ORG', 'United Nations')
    assert entities_summary[1] == ('PERSON', 'Ban Ki-moon')

def test_entity_overlap():
    entities_source = extract_entities(source)
    entities_summary = extract_entities(summary)
    entities_overlap = entity_overlap(entities_source, entities_summary)
    assert len(entities_overlap) == 2
    assert ('DATE', '21 June 2011') in entities_overlap
    assert ('ORDINAL', 'second') in entities_overlap