import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
import json
from Airflow.dags.src.data_prep import data_overview, data_validation, data_cleaning
from Airflow.dags.src.label_encode import encode_data
from Airflow.dags.data_prep_dag import data_overview_callable, data_validation_callable, data_cleaning_callable, encode_data_callable
from airflow.models import TaskInstance

class TestDagTasks(unittest.TestCase):

    def setUp(self):
        # Load real data from AmesHousing.csv
        self.real_data = pd.read_csv('Methodology/AmesHousing.csv').head(5).to_json()

    def test_data_overview_task(self):
        # Create a mock for TaskInstance
        ti_mock = MagicMock(spec=TaskInstance)

        # Simulate pulling real data from XCom
        ti_mock.xcom_pull.return_value = self.real_data

        # Run the callable function
        kwargs = {'ti': ti_mock}
        data_overview_callable(**kwargs)

        # Assert data was pushed to XCom
        ti_mock.xcom_push.assert_called_once()

    def test_data_validation_task(self):
        ti_mock = MagicMock(spec=TaskInstance)

        ti_mock.xcom_pull.return_value = self.real_data

        kwargs = {'ti': ti_mock}
        data_validation_callable(**kwargs)

        ti_mock.xcom_push.assert_called_once()

    def test_data_cleaning_task(self):
        ti_mock = MagicMock(spec=TaskInstance)

        ti_mock.xcom_pull.return_value = self.real_data

        kwargs = {'ti': ti_mock}
        data_cleaning_callable(**kwargs)

        ti_mock.xcom_push.assert_called_once()

    def test_encode_data_task(self):
        ti_mock = MagicMock(spec=TaskInstance)

        ti_mock.xcom_pull.return_value = self.real_data

        kwargs = {'ti': ti_mock}
        encode_data_callable(**kwargs)

         # Check if data was pushed to XCom with the expected keys
        ti_mock.xcom_push.assert_any_call(key='encoded_result', value=unittest.mock.ANY)
        # ti_mock.xcom_push.assert_any_call(key='remaining_mappings', value=unittest.mock.ANY)

if __name__ == '__main__':
    unittest.main()
