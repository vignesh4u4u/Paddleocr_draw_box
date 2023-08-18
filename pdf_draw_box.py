from flask import Flask,request,render_template,jsonify,json,send_file
from pdf2image import convert_from_path
from paddleocr import PaddleOCR,draw_ocr
import pytesseract as pytes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import re
import cv2
from PIL import Image
import requests
import json
ocr = PaddleOCR(use_angle_cls=True,lang='en')
poppler_path=r"C:\Users\VigneshSubramani\Documents\poppler-23.08.0\Library\bin"
output_dir=r"C:\Users\VigneshSubramani\Desktop\MAIN PROJECT\OCR_METHOD\Image"
app=Flask(__name__)
@app.route("/ml-service/draw_box/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/draw_box", methods=["POST"])
def image_bounding_box():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "temp.pdf"
        file.save(file_path)
        pages = convert_from_path(pdf_path=file_path, poppler_path=poppler_path)
        image_path_list = []
        for i, page in enumerate(pages):
            image_path = os.path.join(output_dir, f"page_{i + 1}.png")
            page.save(image_path, "PNG")
            image_path_list.append(image_path)
        detected_text_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            font_path = r"C:\Users\VigneshSubramani\Music\Paddleocr\PaddleOCR\doc\fonts\latin.ttf"
            im_show = draw_ocr(image, boxes, font_path=font_path)
            pil_im_show = Image.fromarray(im_show)
            image_buffer = io.BytesIO()
            pil_im_show.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            return send_file(image_buffer, mimetype='image/png')
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)