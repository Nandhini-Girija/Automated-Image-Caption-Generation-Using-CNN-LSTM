# Automated-Image-Caption-Generation-Using-CNN-LSTM

## **Project Overview**  
This project implements an **Image Caption Generator** that automatically generates textual descriptions for images using a hybrid **CNN-LSTM** deep learning model.  
- **CNN (VGG16)** extracts visual features from images.  
- **LSTM (Long Short-Term Memory)** generates meaningful captions based on extracted image features.  
- Trained on the **Flickr8K dataset**, which contains 8000 images with multiple captions.  
- Evaluated using **BLEU scores** to measure caption accuracy.  

## **Features**  
✔ **Feature Extraction**: Uses **VGG16** (pre-trained on ImageNet) to extract image features.  
✔ **Caption Generation**: Uses an **LSTM-based encoder-decoder model** to generate captions.  
✔ **BLEU Score Evaluation**: Measures caption accuracy.  
✔ **Preprocessing**: Tokenization, text cleaning, and sequence padding.  
✔ **Visualization**: Displays generated captions with images.  

## **Workflow**  
1. **Feature Extraction using CNN (VGG16)**  
   - Loads images and resizes them to 224×224 pixels.  
   - Extracts feature vectors from the last convolutional layer of **VGG16**.  
   - Stores features in a dictionary and serializes them for efficient processing.  

2. **Caption Preprocessing**  
   - Cleans captions (lowercasing, removing special characters).  
   - Adds `startseq` and `endseq` tokens to mark the beginning and end of captions.  
   - Tokenizes text and creates a vocabulary.  

3. **Model Training (LSTM-based Encoder-Decoder)**  
   - **Encoder**: Processes image features with a Dense layer.  
   - **Decoder**: Processes text sequences with an Embedding layer and LSTM.  
   - Merges both representations and predicts the next word in the caption.  

4. **Caption Generation & Evaluation**  
   - Uses the trained model to generate captions for new images.  
   - Evaluates performance using **BLEU-1** and **BLEU-2** scores.  

## **Installation & Setup**  

### **1. Prerequisites**  
Ensure you have **Python** installed with the necessary dependencies:  

```sh
pip install tensorflow keras numpy pandas matplotlib nltk opencv-python
```

### **2. Dataset Preparation**  
- Download the **Flickr8K dataset** and store images in the `Images/` folder.  
- Store captions in `captions.txt`.  

### **3. Run the Training Script**  
Execute the script to extract features and train the model:  

```sh
python train_model.py
```

### **4. Generate Captions for Images**  
To predict captions for new images:  

```sh
python generate_caption.py --image <image_path>
```

## **Results**  
✔ **BLEU Score Evaluation**  
   - **BLEU-1** (unigram precision).  
   - **BLEU-2** (bigram precision).  
✔ **Generated captions are contextually relevant** to the input images.  

## **Future Enhancements**  
🔹 **Use Transformer-based models** (e.g., GPT, BERT) for better language modeling.  
🔹 **Train on larger datasets** (e.g., MS-COCO).  
🔹 **Deploy as a web-based image captioning tool**.  

## **Authors**  
- **Nandhini K (M230788EC)**  
National Institute of Technology Calicut, India  
