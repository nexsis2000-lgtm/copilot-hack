import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('flight_demo', Path(__file__).with_name('app.py'))
        cls.module = importlib.util.module_from_spec(spec)
        # Never unpickle or run the repository's trained model in unit tests.
        with patch('pickle.load', return_value=Mock()):
            spec.loader.exec_module(cls.module)
        cls.client = cls.module.app.test_client()

    def test_prediction_preserves_numeric_probabilities(self):
        for vector in ([0.8, 0.2], [1.0, 0.0], [0.0, 1.0]):
            with self.subTest(vector=vector):
                self.module.model.predict_proba.return_value = [vector]
                response = self.client.get('/predict?day_of_week=3&airport_id=123')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json, {'certainty':vector[0], 'delay':vector[1]})

    def test_invalid_parameters_return_client_error(self):
        for query in ('', '?day_of_week=no&airport_id=123', '?day_of_week=3', '?day_of_week=8&airport_id=123'):
            self.assertEqual(self.client.get('/predict'+query).status_code, 400)

    def test_airport_csv_preserves_quoted_commas_and_last_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'airports.csv').write_text('id,name\n1,"London, UK"\n2,Paris', encoding='utf-8')
            with patch.object(self.module, 'ASSET_DIR', folder):
                response = self.client.get('/airports')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, [{'id':1,'name':'London, UK'}, {'id':2,'name':'Paris'}])
