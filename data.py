from torchmeta.datasets.helpers import miniimagenet, omniglot, tieredimagenet
from torchmeta.utils.data import BatchMetaDataLoader

import torchvision
from torchvision import transforms
import torch


# torch meta : https://github.com/tristandeleu/pytorch-meta

#Todo tiered, omniglot normalize 값

def data_loader(config):
    if config.dataset == 'mini':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # ToDo : shuffle, metasplit 각 옵션 정확하게 숙지
        # Todo : way를 test와 동일하게 하나
        train_set = miniimagenet(config.data_root, ways=config.n_class_train, shots= config.n_shot, test_shots=config.n_query_train,
                                meta_split='train', shuffle=True, download=True,
                                transform=transform)
        val_set = miniimagenet(config.data_root, ways=config.n_class_test, shots=config.n_shot, test_shots=config.n_query_test,
                              meta_split='val', shuffle=True, download = True,
                              transform=transform)
        test_set = miniimagenet(config.data_root, ways=config.n_class_test, shots=config.n_shot, test_shots=config.n_query_test,
                               meta_split='test', shuffle=True, download =True,
                               transform=transform)

    elif config.dataset =='tiered':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((),())
        ])

        train_set = tieredimagenet(config.data_root, ways=config.n_class_train, shots=config.n_shot,
                                 test_shots=config.n_query_train,
                                 meta_split='train', shuffle=True, download=True,
                                 transform=transform)
        val_set = tieredimagenet(config.data_root, ways=config.n_class_test, shots=config.n_shot,
                               test_shots=config.n_query_test,
                               meta_split='val', shuffle=True, download=True,
                               transform=transform)
        test_set = tieredimagenet(config.data_root, ways=config.n_class_test, shots=config.n_shot,
                                test_shots=config.n_query_test,
                                meta_split='test', shuffle=True, download=True,
                                transform=transform)

    elif config.dataset == 'omniglot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((), ())
        ])
        train_set = omniglot(config.data_root, ways=config.n_class_train, shots=config.n_shot,
                                 test_shots=config.n_query_train,
                                 meta_split='train', shuffle=True, download=True,
                                 transform=transform)
        val_set = omniglot(config.data_root, ways=config.n_class_test, shots=config.n_shot,
                               test_shots=config.n_query_test,
                               meta_split='val', shuffle=True, download=True,
                               transform=transform)
        test_set = omniglot(config.data_root, ways=config.n_class_test, shots=config.n_shot,
                                test_shots=config.n_query_test,
                                meta_split='test', shuffle=True, download=True,
                                transform=transform)
    else:
        raise ValueError

    train_loader = BatchMetaDataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = BatchMetaDataLoader(val_set, batch_size = config.batch_size, shuffle=True)
    test_loader = BatchMetaDataLoader(test_set, batch_size = config.batch_size, shuffle=True)

    dataloloader = {}
    dataloloader['meta-train'] = train_loader
    dataloloader['meta-val'] = val_loader
    dataloloader['meta-test'] = test_loader

    return dataloloader

