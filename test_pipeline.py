import sys
import unittest
from database import init_db
from models import XGBoostModel

class TestAITrading(unittest.TestCase):
    def test_database_init(self):
        # tests if db initializes without crashing (using in-memory db)
        Session = init_db('sqlite:///:memory:')
        self.assertIsNotNone(Session)
        
    def test_xgboost_init(self):
        # tests if model initializes
        model = XGBoostModel()
        self.assertIsNotNone(model.model)
        self.assertFalse(model.is_trained)

if __name__ == '__main__':
    unittest.main()
