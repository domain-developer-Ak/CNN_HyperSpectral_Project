class Config:
    # Data parameters
    DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    NUM_CLASSES = 5
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32

    # Model parameters
    LEARNING_RATE = 0.001
    EPOCHS = 20
    DROPOUT_RATE = 0.5

    # Paths
    MODEL_SAVE_PATH = "saved_model.pth"
    LOG_DIR = "logs"
