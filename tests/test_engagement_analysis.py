import unittest
import pandas as pd
from engagement_analysis import validate_hierarchy, calculate_normalization_params

class TestEngagementAnalysis(unittest.TestCase):
    
    def test_validate_hierarchy_success(self):
        # Conc (2.0) > Neutro (0.0) > Relax (1.0)
        trend = pd.Series({2.0: 0.8, 0.0: 0.5, 1.0: 0.2})
        self.assertTrue(validate_hierarchy(trend))
        
    def test_validate_hierarchy_failure(self):
        trend = pd.Series({2.0: 0.2, 0.0: 0.5, 1.0: 0.8})
        self.assertFalse(validate_hierarchy(trend))

    def test_calculate_normalization_params(self):
        df = pd.DataFrame({
            'IEN_Global': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        # 0.05 quantile of [0.1...1.0] is roughly 0.145
        # 0.95 quantile of [0.1...1.0] is roughly 0.955
        params = calculate_normalization_params(df)
        self.assertIn('ien_min', params)
        self.assertIn('ien_max', params)
        self.assertAlmostEqual(params['ien_min'], df['IEN_Global'].quantile(0.05))
        self.assertAlmostEqual(params['ien_max'], df['IEN_Global'].quantile(0.95))

if __name__ == '__main__':
    unittest.main()
