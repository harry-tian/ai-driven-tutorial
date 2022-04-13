import unittest
import numpy as np
import evals.embed_evals as evals

test_embed_2class = (np.array([
    [0,0],
    [0,1],
    [0,3],
    [0,6]
]),np.array([0,0,1,1]))

test_embed_2class = (np.array([
    [0,0,1],
    [0,1,1],
    [0,3,1],
    [0,6,1]
]),np.array([0,0,1,1]))

class Test_class_1NN_idx(unittest.TestCase):
    def test_2class(self):
        evals.class_1NN_idx(**test_embed_2class,)
        self.assertEqual(func(5), 6)
        self.assertNotEqual(func(5), 6)

if __name__ == '__main__':
    unittest.main()