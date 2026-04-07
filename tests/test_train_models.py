import unittest
import pandas as pd
import numpy as np
from train_models import prepare_data

class TestTrainModels(unittest.TestCase):
    
    def test_prepare_data(self):
        df = pd.DataFrame({
            'feat1': [1, 2, 3, 4],
            'feat2': [10, 20, 30, 40],
            'Label': [0, 1, 0, 1]
        })
        X_scaled, y, scaler = prepare_data(df)
        
        self.assertEqual(X_scaled.shape, (4, 2))
        self.assertEqual(len(y), 4)
        # Check if scaling worked (mean should be close to 0)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_scaled.std(), 1, places=0) # with 4 samples, std might be slightly off due to degrees of freedom but should be close to 1

if __name__ == '__main__':
    unittest.main()
