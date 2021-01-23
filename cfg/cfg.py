class cfg:
    # project
    NAME = 'BBN.cassava.new_cv20.resnext50.SCE.SGD.cosine.RandAugment.50epoch'
    OUTPUT_DIR = './output/cassava/new_cv20'
    CPU_MODE = False
    GPU_ID = '0'
    K_FOLD = [4, 3, 2, 1, 0]
    CLS_NUM = 5

    # train
    BATCH_SIZE = 256
    SHUFFLE = True
    NUM_WORKERS = 32
    PIN_MEMORY = True
    MAX_EPOCH = 100
    TENSORBOARD_ENABLE = True

    # OPTIMIZER
    OPTIMIZER_TYPE = 'SGD'
    BASE_LR = 3e-5
    MOMENTUM = 0.9
    WEIGHT_DECAY = 2e-5

    # LR_SCHEDULER
    LOSS_TYPE = 'CrossEntropy'
    # LOSS_TYPE = 'SymmetricCrossEntropy'
    #  LOSS_TYPE = 'LDAMLoss'
    #  LOSS_TYPE = 'CSCE'
    #  LOSS_TYPE = 'LabelSmoothingCrossEntropy'


    SCHEDULER_TYPE = 'cosine'
    # TYPE = 'multistep'
    # TYPE = 'warmup'
    LR_STEP = [15, 25]
    LR_FACTOR = 0.1
    COSINE_DECAY_END = 0
    WARM_EPOCH = 5

    SHOW_STEP = 50
    SAVE_STEP = 1
    VALID_STEP = 1
    SEED = 2021

    # data
    DATASET = 'IMBALANCECASSAVA'
    BACKBONE = 'TRANSFORMERS'
    mean = [0.430316, 0.496727, 0.313513]
    std = [0.238024, 0.240754, 0.228859]
    COLOR_SPACE = 'RGB'
    INPUT_SIZE = 512
    INPUT_CHANNEL = 3