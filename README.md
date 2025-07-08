CURRENCY NOTE RECOGNITION PROJECT
===============================

PROJECT OVERVIEW:
-----------------
The Currency Note Recognition project is a machine learning-based web application designed to identify Indian currency notes from images. Built using TensorFlow and Keras, it employs a Convolutional Neural Network (CNN) to classify images of currency notes into seven denominations: 10, 20, 50, 100, 200, 500, and 2000 Indian Rupees. The project leverages data augmentation and image preprocessing to enhance model performance, making it suitable for real-world applications such as automated currency recognition systems.

FILES INCLUDED:
---------------
1. CNN_currency_notes.ipynb – Main Jupyter Notebook containing the code for dataset extraction, model building, training, and evaluation
2. CNN_Dataset.zip – Compressed dataset containing training and testing images of Indian currency notes
3. README.txt – This file

FEATURE DESCRIPTION:
--------------------
- **Purpose**: Classifies images of Indian currency notes into their respective denominations.
- **Technology**: Utilizes TensorFlow and Keras for building a CNN model, with data augmentation using ImageDataGenerator.
- **Dataset**: Contains 153 training images and 42 testing images, organized into seven classes (1Hundrednote, 2Hundrednote, 2Thousandnote, 5Hundrednote, Fiftynote, Tennote, Twentynote).
- **Model Architecture**:
  - Three Conv2D layers with ReLU activation (32, 64, and 128 filters, respectively).
  - MaxPooling2D layers for dimensionality reduction.
  - Flatten layer followed by a Dense layer (128 units, ReLU activation) with 50% Dropout for regularization.
  - Output Dense layer with softmax activation for 7-class classification.
- **Input**: Images of size 128x128 pixels, preprocessed with rescaling, rotation, zoom, and horizontal flipping for training.
- **Output**: Predicted currency denomination based on the input image.
- **Dependencies**: tensorflow, keras, matplotlib, numpy, zipfile, os.

SETUP INSTRUCTIONS:
-------------------
1. **Install Required Python Libraries**:
   Install the necessary dependencies using pip:
   ```bash
   pip install tensorflow keras matplotlib numpy
   ```

2. **Ensure Files are in the Same Directory**:
   Place the following files in the project folder:
   - CNN_currency_notes.ipynb
   - CNN_Dataset.zip

3. **Extract the Dataset**:
   The notebook includes code to extract `CNN_Dataset.zip` into a `dataset` folder with `Train` and `Test` subdirectories. Ensure the zip file is in the same directory as the notebook.

4. **Run the Jupyter Notebook**:
   Launch Jupyter Notebook and open `CNN_currency_notes.ipynb`:
   ```bash
   jupyter notebook
   ```
   Execute the cells sequentially to:
   - Extract the dataset.
   - Load and preprocess images using ImageDataGenerator.
   - Build and compile the CNN model.
   - Train the model for 27 epochs.
   - Evaluate model performance on the test set.

WEB APPLICATION FEATURES:
-------------------------
- **Image Preprocessing**: Uses ImageDataGenerator for rescaling (1./255), rotation (20 degrees), zoom (20%), and horizontal flipping to augment training data and improve model robustness.
- **Model Training**: Trains the CNN model for 27 epochs using the Adam optimizer (learning rate 0.001) and categorical crossentropy loss.
- **Model Evaluation**: Validates performance on a separate test set with 42 images across 7 classes.
- **GPU Support**: Configured to leverage GPU acceleration (e.g., T4 GPU in Colab) for faster training.

DEPENDENCIES:
-------------
- Python 3.8+
- tensorflow
- keras
- matplotlib
- numpy
- zipfile (standard library)
- os (standard library)

PROJECT OUTPUT:
---------------
- **Trained Model**: A CNN model capable of classifying Indian currency notes with high accuracy.
- **Performance Metrics**: Outputs training and validation accuracy/loss per epoch, accessible via the `history` object in the notebook.
- **Class Mapping**: Maps predictions to denominations (e.g., {'1Hundrednote': 0, '2Hundrednote': 1, ...}).

LIMITATIONS:
------------
- **Dataset Size**: The dataset is relatively small (153 training images, 42 testing images), which may limit generalization to diverse real-world conditions.
- **Image Quality**: Model performance depends on image clarity and lighting; poor-quality images may reduce accuracy.
- **Denomination Scope**: Limited to seven Indian currency denominations; other denominations or currencies are not supported.
- **Environment**: Designed for execution in a GPU-enabled environment like Google Colab; local execution may require additional setup.

FUTURE IMPROVEMENTS:
--------------------
- Expand the dataset with more images to improve model robustness and generalization.
- Implement real-time image capture and prediction using a webcam or mobile device.
- Add support for additional currencies or denominations.
- Integrate the model into a web or mobile application using frameworks like Flask or Streamlit for user-friendly deployment.
- Enhance preprocessing with advanced techniques (e.g., edge detection, noise reduction) to handle varied image conditions.

CONCLUSION:
-----------
The Currency Note Recognition project demonstrates the application of deep learning for image classification, specifically tailored to identifying Indian currency denominations. By leveraging a CNN with data augmentation, it achieves reliable performance on a small dataset. This project serves as a foundation for further development into a deployable application for automated currency recognition, suitable for educational purposes or real-world financial systems.

Developed as part of a Machine Learning and Computer Vision portfolio project by Saanya Lakhani. Feel free to use for educational purpose.
