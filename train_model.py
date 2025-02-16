import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = './'
WORKING_DIR = './'

# Load VGG16 model for feature extraction
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Extract image features
features = {}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image).reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Save extracted features
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb') as f:
    pickle.dump(features, f)

# Load captions
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# Create mapping between image ID and captions
mapping = {}
for line in tqdm(captions_doc.split('
')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], " ".join(tokens[1:])
    image_id = image_id.split('.')[0]
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Clean captions
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower().replace('[^A-Za-z]', '').replace('\s+', ' ')
            captions[i] = 'startseq ' + caption + ' endseq'

clean(mapping)

# Tokenize captions
all_captions = [caption for key in mapping for caption in mapping[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

# Define LSTM-based captioning model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
epochs = 20
batch_size = 32
steps = len(mapping) // batch_size

for i in range(epochs):
    model.fit([features, tokenizer], steps_per_epoch=steps, epochs=1, verbose=1)

# Save trained model
model.save(os.path.join(WORKING_DIR, 'caption_model.h5'))
