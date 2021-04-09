#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
import model




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model.train_model('cs-train', test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model_load = model.load_model(model.ridgeRegressionModel, 'Australia', test=True)
        
        self.assertTrue('predict' in dir(model_load))
        self.assertTrue('fit' in dir(model_load))

       
    def test_03_predict(self):
        """
        test the predict function input
        """


        day = '03'
        month = '09'
        year = '2019'
        country = 'United Kingdom'
    

        result = model.model_predict(day, month, year, country, 'ridge', test=True)
        y_pred = result['y_pred']
        self.assertTrue(len(y_pred) == 1)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
