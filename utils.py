def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def experiment_name_generator(config):
    #few_shot_setting = f"{str(config.n_way)}way_{str(config.n_shot)}shot_{str(config.n_query)}query"
    #training_params = f"{str(config.glocal_layers)}Layers_{str(config.graph_node_dim)}dim"
    #name = f"{few_shot_setting}_{training_params}"
    return None