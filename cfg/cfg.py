class cfg:
    # project
    NAME = 'ViT.cassava.new_cv20.CE.SGD.cosine.RandAugment.80epoch'
    OUTPUT_DIR = './output/cassava/new_cv20'
    CPU_MODE = False
    GPU_ID = '0'
    K_FOLD = [4, 3, 2, 1, 0]
    CLS_NUM = 5

    # train
    BATCH_SIZE = 128
    NUM_WORKERS = 16
    PIN_MEMORY = True
    MAX_EPOCH = 80
    TENSORBOARD_ENABLE = True

    # OPTIMIZER
    OPTIMIZER_TYPE = 'SGD'
    # BASE_LR = 3e-5
    BASE_LR = 3e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 2e-5

    # LR_SCHEDULER
    LOSS_TYPE = 'CrossEntropy'
    # LOSS_TYPE = 'SymmetricCrossEntropy'
    #  LOSS_TYPE = 'LDAMLoss'
    #  LOSS_TYPE = 'CSCE'
    #  LOSS_TYPE = 'LabelSmoothingCrossEntropy'

    # SCHEDULER_TYPE = 'cosine'
    # SCHEDULER_TYPE = 'multistep'
    SCHEDULER_TYPE = 'warmup'
    LR_STEP = [30, 50, 70]
    LR_FACTOR = 1.0 / 3
    COSINE_DECAY_END = 0
    WARM_EPOCH = 10

    SHOW_STEP = 5
    SAVE_STEP = 5
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
