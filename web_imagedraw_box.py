from flask import Flask, request, render_template, jsonify, json,send_file
from paddleocr import PaddleOCR, draw_ocr
import pytesseract as pytes
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import easyocr
import os
import re
import cv2
import io
import requests
import json
import base64
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
app=Flask(__name__,template_folder="template")
@app.route("/")
def home():
    return render_template("image.html")
@app.route("/pre",methods=["POST"])
def image_bounding_box():
    if request.method == "POST":
        image_file = request.files['image']
        image_data = image_file.read()
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(gray)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]
        font = ImageFont.load_default()
        font_path = r"C:\Users\VigneshSubramani\Music\Paddleocr\PaddleOCR\doc\fonts\latin.ttf"
        im_show = draw_ocr(image, boxes, font_path=font_path)
        pil_im_show = Image.fromarray(im_show)
        buffered = io.BytesIO()
        pil_im_show.save(buffered, format="PNG")
        image_data_uri = base64.b64encode(buffered.getvalue()).decode('utf-8')
    #return render_template("image.html", image_data_uri=image_data_uri)
    return render_template("image.html",**locals())
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)