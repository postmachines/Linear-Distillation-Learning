def preprocess_config(c):
    conf_dict = {}
    int_params = ['way', 'train_shot', 'test_shot', 'epochs', 'trials',
                  'silent', 'x_dim', 'z_dim', 'channels', 'gpu',
                  'test_batch', 'save_data', 'log_accuracy']
    float_params = ['lr', 'lr_target', 'lr_predictor']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict