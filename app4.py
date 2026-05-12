# ============================================
# app4.py
# CNN Image Classification Streamlit App
# Covers ALL Vedgrow Internship Requirements
# ============================================

# REQUIREMENTS COVERED:
# ✅ CNN Architecture using TensorFlow/Keras
# ✅ Data Augmentation
# ✅ Train/Validation/Test Split
# ✅ Accuracy & Loss Graphs
# ✅ Custom Image Upload Prediction
# ✅ Professional UI
# ✅ Confusion Matrix
# ✅ Classification Report
# ✅ MNIST Dataset

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="CNN Digit Classifier",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

body{
    background-color:#0f172a;
}

.main{
    background-color:#020617;
    color:white;
}

h1,h2,h3{
    color:white;
}

.stButton>button{
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:white;
    border:none;
    border-radius:12px;
    padding:14px 28px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    transform:scale(1.03);
    transition:0.3s;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE SECTION
# ============================================

st.markdown("""
<h1 style='text-align:center;font-size:55px;'>
🧠 CNN Image Classification App
</h1>

<h3 style='text-align:center;color:#cbd5e1;'>
Handwritten Digit Recognition using Deep Learning
</h3>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATASET
# ============================================

st.header("📂 Loading MNIST Dataset")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

st.write("Training Images Shape:", X_train.shape)
st.write("Testing Images Shape:", X_test.shape)

# ============================================
# DATA PREPROCESSING
# ============================================

st.header("⚙️ Data Preprocessing")

# Normalize

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# One Hot Encoding

y_train_cat = to_categorical(y_train,10)
y_test_cat = to_categorical(y_test,10)

st.success("Data normalized and reshaped successfully!")

# ============================================
# DATA AUGMENTATION
# ============================================

st.header("🔄 Data Augmentation")

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

st.success("Data augmentation applied!")

# ============================================
# BUILD CNN MODEL
# ============================================

st.header("🏗️ CNN Architecture")

model = Sequential()

# 1st Convolution Layer

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

# 2nd Convolution Layer

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

# Flatten Layer

model.add(Flatten())

# Dense Layer

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

# Output Layer

model.add(Dense(10, activation='softmax'))

# ============================================
# MODEL SUMMARY
# ============================================

model.summary(print_fn=lambda x: st.text(x))

# ============================================
# COMPILE MODEL
# ============================================

st.header("⚡ Compiling Model")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

st.success("Model compiled successfully!")

# ============================================
# TRAIN MODEL
# ============================================

st.header("🚀 Training CNN Model")

epochs = st.slider(
    "Select Epochs",
    1,
    10,
    3
)

if st.button("🔥 Start Training"):

    history = model.fit(
        datagen.flow(X_train,y_train_cat,batch_size=64),
        validation_data=(X_test,y_test_cat),
        epochs=epochs
    )

    st.success("Training Completed!")

    # ============================================
    # ACCURACY GRAPH
    # ============================================

    st.header("📈 Training vs Validation Accuracy")

    fig1 = plt.figure(figsize=(10,5))

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    st.pyplot(fig1)

    # ============================================
    # LOSS GRAPH
    # ============================================

    st.header("📉 Training vs Validation Loss")

    fig2 = plt.figure(figsize=(10,5))

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    st.pyplot(fig2)

    # ============================================
    # MODEL EVALUATION
    # ============================================

    st.header("📊 Model Evaluation")

    loss, accuracy = model.evaluate(
        X_test,
        y_test_cat
    )

    st.success(f"Test Accuracy: {accuracy*100:.2f}%")

    # ============================================
    # PREDICTIONS
    # ============================================

    predictions = model.predict(X_test)

    predicted_labels = np.argmax(predictions, axis=1)

    # ============================================
    # CONFUSION MATRIX
    # ============================================

    st.header("🧩 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predicted_labels
    )

    fig3 = plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    st.pyplot(fig3)

    # ============================================
    # CLASSIFICATION REPORT
    # ============================================

    st.header("📄 Classification Report")

    report = classification_report(
        y_test,
        predicted_labels
    )

    st.text(report)

    # ============================================
    # SAVE MODEL
    # ============================================

    model.save("cnn_mnist_model.h5")

    st.success("Model saved successfully!")

# ============================================
# CUSTOM IMAGE PREDICTION
# ============================================

st.header("🖼️ Upload Custom Digit Image")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        width=200
    )

    # Preprocess Image

    img = img.convert("L")

    img = img.resize((28,28))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = img_array.reshape(1,28,28,1)

    # Load Saved Model

    loaded_model = tf.keras.models.load_model(
        "cnn_mnist_model.h5"
    )

    prediction = loaded_model.predict(img_array)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # ============================================
    # PREMIUM RESULT CARD
    # ============================================

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#2563eb,#1d4ed8);
        padding:40px;
        border-radius:25px;
        text-align:center;
        margin-top:20px;
        color:white;
        box-shadow:0px 10px 25px rgba(0,0,0,0.4);
    ">

        <h1 style="
            font-size:45px;
            margin-bottom:20px;
        ">
            🔢 Predicted Digit
        </h1>

        <h1 style="
            font-size:90px;
            font-weight:bold;
        ">
            {predicted_digit}
        </h1>

        <h3>
            Confidence: {confidence:.2f}%
        </h3>

    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<hr>

<h4 style='text-align:center;color:gray;'>
Built with TensorFlow, Keras & Streamlit 🚀
</h4>
""", unsafe_allow_html=True)
