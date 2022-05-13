
class DefualtConfig(object):

    ###################################################################
    # Model
    model_name = 'ViT'
    model_path = './model.pth'
    load_model = False
    num_classes = 14

    # Model : ViT
    pretrained_model = 'google/vit-base-patch16-224-in21k'

    ###################################################################
    # Training
    start_epoch = 0
    num_epochs = 15
    earlyStop_interval = 3

    batch_size = 32
    lr = 2e-5
    lr_warmup_epoch = 5

    ###################################################################
    # GPU Settings
    # use_gpu = True
    use_gpu_index = 0

    ###################################################################
    # DataLoader
    num_workers = 6

    # Dataset
    trainset_path = './data/dataset/train'
    testset_path = './data/dataset/test'

    train_valid_split = 0.3  # ratio of valid set

    ###################################################################

    def __init__(self) -> None:
        '''
        '''
        pass

    def parse(self, kwargs):
        '''
        '''
        print('User config : ')
        pass
