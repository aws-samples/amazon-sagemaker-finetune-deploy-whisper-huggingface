import logging
import json
import torch
import librosa
import numpy as np
import io

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_path = '/opt/ml/model'
logger.info("Libraries are loaded")


def model_fn(model_dir):
    device = get_device()

    # model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model = pipeline(task='automatic-speech-recognition', model=model_dir, device=device)
    logger.info("Model is loaded")

    return model


def input_fn(input_data, content_type):
    # input_data = json.loads(json_request_data)
    logger.info("Input data is processed")
    speech_array, sampling_rate = librosa.load(io.BytesIO(input_data), sr=16000)

    return np.asarray(speech_array)


def predict_fn(input_data, model):
    logger.info("Starting inference.")

    transcript = model(input_data)["text"]

    return transcript


def output_fn(transcript, accept='application/json'):
    return json.dumps(transcript)


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

