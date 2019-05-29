import unittest

from scripts.omd_train_mnist import run_experiment_full_test
from scripts.utils import preprocess_config


class TestMnistOMD(unittest.TestCase):

    def test_no_logging_3_shot(self):
        config = {
            'dataset' : 'mnist',
            'way': '10',
            'train_shot': '3',
            'test_shot': '1',
            'loss': 'MSE',
            'epochs': '5',
            'trials': '1',
            'silent': '1',
            'split': 'test',
            'x_dim': '28',
            'z_dim': '784',
            'initialization': 'xavier_normal',
            'optimizer': 'adam',
            'lr': '0.001',
            'channels': '1',
            'gpu': '0',
            'test_batch': '2000',
            'log_test_accuracy': 0
        }
        config = preprocess_config(config)
        run_experiment_full_test(config)

    def test_test_accuracy_logging_3_shot(self):
        config = {
            'dataset' : 'mnist',
            'way': '10',
            'train_shot': '3',
            'test_shot': '1',
            'loss': 'MSE',
            'epochs': '5',
            'trials': '1',
            'silent': '1',
            'split': 'test',
            'x_dim': '28',
            'z_dim': '784',
            'initialization': 'xavier_normal',
            'optimizer': 'adam',
            'lr': '0.001',
            'channels': '1',
            'gpu': '0',
            'test_batch': '2000',
            'log_test_accuracy': 1
        }
        config = preprocess_config(config)
        run_experiment_full_test(config)

    def test_1_shot(self):
        config = {
            'dataset': 'mnist',
            'way': '10',
            'train_shot': '1',
            'test_shot': '1',
            'loss': 'MSE',
            'epochs': '5',
            'trials': '1',
            'silent': '1',
            'split': 'test',
            'x_dim': '28',
            'z_dim': '784',
            'initialization': 'xavier_normal',
            'optimizer': 'adam',
            'lr': '0.001',
            'channels': '1',
            'gpu': '0',
            'test_batch': '2000',
            'log_test_accuracy': 0
        }
        config = preprocess_config(config)
        run_experiment_full_test(config)

    def test_10_shot(self):
        config = {
            'dataset': 'mnist',
            'way': '10',
            'train_shot': '10',
            'test_shot': '1',
            'loss': 'MSE',
            'epochs': '5',
            'trials': '1',
            'silent': '1',
            'split': 'test',
            'x_dim': '28',
            'z_dim': '784',
            'initialization': 'xavier_normal',
            'optimizer': 'adam',
            'lr': '0.001',
            'channels': '1',
            'gpu': '0',
            'test_batch': '2000',
            'log_test_accuracy': 0
        }
        config = preprocess_config(config)
        run_experiment_full_test(config)