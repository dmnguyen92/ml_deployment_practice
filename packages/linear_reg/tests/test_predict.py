import math

from linear_reg.predict import make_prediction
from linear_reg.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='data_test.csv')
    single_test_json = test_data[0:1]

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    #assert math.ceil(subject.get('predictions')[0]) == 112476


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='data_test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 3

    # We expect some rows to be filtered out
    #assert len(subject.get('predictions')) != original_data_length