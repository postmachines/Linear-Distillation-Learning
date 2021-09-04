"""
Miscellaneous functions for training procedure
"""

import pandas as pd


def preprocess_config(c):
    conf_dict = {}
    int_params = ['way', 'train_shot', 'test_shot', 'epochs', 'trials',
                  'silent', 'x_dim', 'z_dim', 'channels', 'gpu',
                  'test_batch', 'save_data', 'log_test_accuracy']
    float_params = ['lr', 'lr_target', 'lr_predictor']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


class Logger(object):
    def __init__(self, filepath, columns):
        if not filepath.endswith('.csv'):
            filepath += '.csv'
        self.path = filepath
        df = pd.DataFrame(columns=columns)
        df.to_csv(filepath, index=False)
        print(f"Logging results can be find at {filepath}")

    def log(self, dict):
        df = pd.read_csv(self.path)
        df = df.append(dict, ignore_index=True)
        df.to_csv(self.path, index=False)