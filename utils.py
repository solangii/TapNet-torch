import argparse

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def experiment_name_generator(config, info=None):
    name = f"{str(config.dataset)}_{str(config.n_class_train)}way_{str(config.n_shot)}shot"

    if info is not None:
        name = f"{name}+{str(info)}"

    return name
