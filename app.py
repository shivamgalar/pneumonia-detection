
import os
import io
import base64

import gdown
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template_string
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Google Drive model configuration
MODEL_PATH = "best_pneumonia_model.keras"
FILE_ID = "1ClhVFJr3K6movHQ9ECD_YZP_8a2GxCxk"

# Prediction threshold (kept from your original model setup)
THRESHOLD = 0.75

# Upload configuration
UPLOAD_FOLDER = "/tmp/pneumonia_uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Download model if it is not already present
if not os.path.exists(MODEL_PATH):
    downloaded_file = gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )
    if not downloaded_file or not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model download failed. Check the Google Drive sharing settings."
        )

# Load the trained model once at startup
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    compile=False,
    safe_mode=False
)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pneumonia Detection</title>
    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #e0f2fe, #f8fafc);
            color: #0f172a;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }

        .card {
            width: 100%;
            max-width: 560px;
            background: white;
            padding: 32px;
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(15, 23, 42, .12);
        }

        h1 { margin-top: 0; text-align: center; }

        .subtitle {
            text-align: center;
            color: #64748b;
        }

        input[type=file] {
            width: 100%;
            padding: 14px;
            border: 1px dashed #94a3b8;
            border-radius: 10px;
            margin: 20px 0;
        }

        button {
            width: 100%;
            border: 0;
            padding: 14px;
            border-radius: 10px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover { background: #1d4ed8; }

        .result {
            margin-top: 24px;
            padding: 18px;
            border-radius: 12px;
            background: #f1f5f9;
            text-align: center;
            overflow-wrap: anywhere;
        }

        .normal { color: #15803d; }
        .pneumonia { color: #b91c1c; }

        .warning {
            margin-top: 22px;
            font-size: 12px;
            line-height: 1.6;
            color: #64748b;
        }

        img {
            display: block;
            max-width: 100%;
            max-height: 300px;
            margin: 18px auto 0;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <main class="card">
        <h1>🩻 Pneumonia Detection</h1>
        <p class="subtitle">
            Chest X-ray classification using VGG16
        </p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="xray"
                   accept=".png,.jpg,.jpeg" required>
            <button type="submit">Analyze X-ray</button>
        </form>

        {% if error %}
            <div class="result">{{ error }}</div>
        {% endif %}

        {% if result %}
            <div class="result">
                <h2 class="{{ 'pneumonia' if result == 'PNEUMONIA' else 'normal' }}">
                    {{ result }}
                </h2>
                <p>Model pneumonia score: {{ score }}%</p>
                <img src="data:image/jpeg;base64,{{ image_data }}"
                     alt="Uploaded X-ray">
            </div>
        {% endif %}

        <p class="warning">
            Educational demonstration only. This model is not validated
            for clinical diagnosis. Do not use its output to make medical
            decisions. Consult a qualified healthcare professional.
        </p>
    </main>
</body>
</html>
"""


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None
    error = None
    image_data = None

    if request.method == "POST":
        file = request.files.get("xray")

        if not file or not file.filename:
            error = "Please select a chest X-ray image."

        elif not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."

        else:
            try:
                # Read and resize the uploaded image
                img = Image.open(file.stream).convert("RGB")
                img = img.resize((224, 224))

                # VGG16 preprocessing is already included in the model
                image = np.asarray(img, dtype=np.float32)
                image = np.expand_dims(image, axis=0)

                probability = float(
                    model.predict(image, verbose=0).ravel()[0]
                )

                result = (
                    "PNEUMONIA"
                    if probability >= THRESHOLD
                    else "NORMAL"
                )
                score = f"{probability * 100:.2f}"

                # Encode image for display on the results page
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                image_data = base64.b64encode(
                    buffer.getvalue()
                ).decode("utf-8")

            except Exception:
                app.logger.exception("Image prediction failed")
                error = (
                    "Could not process this image. "
                    "Please upload a valid chest X-ray."
                )

    return render_template_string(
        HTML,
        result=result,
        score=score,
        error=error,
        image_data=image_data
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
