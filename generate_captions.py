import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image
import matplotlib.pyplot as plt

# Load necessary files
WORKING_DIR = './'
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# Load tokenizer
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Load trained model
model = load_model(os.path.join(WORKING_DIR, 'caption_model.h5'))

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate captions
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Generate caption for an image
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(WORKING_DIR, "Images", image_name)
    image = Image.open(img_path)
    y_pred = predict_caption(model, features[image_id], tokenizer, 34)

    print('Predicted Caption:', y_pred)
    plt.imshow(image)
    plt.show()
