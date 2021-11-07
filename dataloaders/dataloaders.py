import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)


    # lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
    # subsetA, subsetB = random_split(init_dataset, lengths)
    dataset_supervised = file.__dict__[dataset_name].__dict__[dataset_name](opt,for_metrics = False ,for_supervision = True)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt,dataset_supervised=dataset_supervised, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))


    dataloader_supervised = torch.utils.data.DataLoader(dataset_supervised, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
   
    # dataloader_train =  dataloader_train.remove(dataloader_supervised.dataset.paths[0],1)

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False)
    
    return dataloader_train,dataloader_supervised, dataloader_val