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
ocr = PaddleOCR(use_angle_cls=True, lang='en')
app=Flask(__name__)
@app.route("/ml-service/draw_box/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/draw_box", methods=["POST"])
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
        font_path = r"latin.ttf"
        im_show = draw_ocr(image, boxes, font_path=font_path)
        pil_im_show = Image.fromarray(im_show)
        image_buffer = io.BytesIO()
        pil_im_show.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        return send_file(image_buffer, mimetype='image/png')
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)