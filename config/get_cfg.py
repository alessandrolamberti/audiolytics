import os
from utils.utils import create_model

GENDER_MODEL_PATH = os.getenv('GENDER_MODEL_PATH', "./weights/model.h5")

model = create_model()
model.load_weights(GENDER_MODEL_PATH)
