import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import random
import datetime

UPLOAD_FOLDER = "static/uploaded"
IMG_FOLDER = "images"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load feature vectors and filenames
with open("model.pkl", "rb") as f:
    features, image_names = pickle.load(f)

# Load MobileNetV2 model
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def recommend(query_feature, top_k=15):
    similarities = cosine_similarity([query_feature], features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [image_names[i] for i in top_indices]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Extract feature from uploaded image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    query_feature = mobilenet.predict(img_array, verbose=0)[0]

    # Get recommended image filenames
    results = recommend(query_feature)

    # Copy recommended images to static/uploaded for display
    for rec_img in results:
        src = os.path.join(IMG_FOLDER, rec_img)
        dst = os.path.join(app.config['UPLOAD_FOLDER'], rec_img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    return render_template("results.html", uploaded=file.filename, results=results)

# Only one definition of /buy
@app.route('/buy', methods=['POST'])
def buy_now():
    item = request.form.get('item')
    price = request.form.get('price')
    style = request.form.get('style')
    return render_template("payment.html", item=item, price=price, style=style)

# Only one definition of /pay
@app.route('/pay', methods=['POST'])
def process_payment():
    item = request.form.get('item')
    price = request.form.get('price')
    style = request.form.get('style')
    size = request.form.get('size')
    quantity = int(request.form.get('quantity', 1))
    name = request.form.get('name')
    email = request.form.get('email')
    mobile = request.form.get('mobile')
    address = request.form.get('address')
    payment_method = request.form.get('payment_method')

    try:
        unit_price = float(price)
    except:
        unit_price = 0.0
    total = unit_price * quantity

    now = datetime.datetime.now()
    invoice_no = now.strftime("%Y%m%d%H%M%S") + str(random.randint(100, 999))

    return render_template("invoice.html",
                           item=item,
                           style=style,
                           size=size,
                           quantity=quantity,
                           unit_price=unit_price,
                           total=total,
                           name=name,
                           email=email,
                           mobile=mobile,
                           address=address,
                           payment_method=payment_method,
                           invoice_no=invoice_no)

if __name__ == "__main__":
    app.run(debug=True)
