import unittest
import pandas as pd
import numpy as np
# Import will fail initially because the functions are not yet exported/extracted
from feature_pipeline import clean_duplicates, calculate_ien_per_sensor

class TestFeaturePipeline(unittest.TestCase):
    
    def test_clean_duplicates(self):
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6]
        })
        expected_len = 3
        result = clean_duplicates(df)
        self.assertEqual(len(result), expected_len)
        self.assertTrue((result.iloc[1] == [2, 5]).all())

    def test_calculate_ien_per_sensor(self):
        # Formula: Beta / (Alpha + Theta + epsilon)
        df = pd.DataFrame({
            'Beta_0': [10.0, 20.0],
            'Alpha_0': [5.0, 5.0],
            'Theta_0': [5.0, 5.0]
        })
        # IEN = 10 / (5 + 5 + 1e-6) ~= 1.0 (approx)
        # 20 / (5 + 5 + 1e-6) ~= 2.0 (approx)
        result = calculate_ien_per_sensor(df, 0)
        np.testing.assert_allclose(result, [1.0, 2.0], rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
