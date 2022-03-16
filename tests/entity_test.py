from preprocessing.entity import extract_entities
from preprocessing.entity import entity_overlap

import unittest

source = """
He was re-elected for a second term by the UN
General Assembly, unopposed and unanimously, on 21
June 2011, with effect from 1 January 2012. Mr. Ban
describes his priorities as mobilising world leaders to deal
with climate change, economic upheaval, pandemics and
increasing pressures involving food, energy and water
"""
summary = """
The United Nations Secretary-General
Ban Ki-moon was elected for a second term in 21 June
2011.
"""

class TestEntityPreprocessing(unittest.TestCase):

    def test_entity_extraction(self):
        entities_source = extract_entities(source)
        entities_summary = extract_entities(summary)
        self.assertEqual(entities_source[0], ('TODO: Type', 'UN General Assembly'))

    def test_entity_overlap(self):
        entities_source = extract_entities(source)
        entities_summary = extract_entities(summary)
        entities_overlap = entity_overlap(entities_source, entities_summary)
        self.assertEqual(len(entities_overlap), 1)

if __name__ == '__main__':
    unittest.main()