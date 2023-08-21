from flask import Flask,request,render_template,jsonify,json,send_file
from paddleocr import PaddleOCR,draw_ocr
import pytesseract as pytes
import numpy as np
import pandas as pd
import seaborn as sns
import pypdfium2 as pdfium
import matplotlib.pyplot as plt
import os
import io
import re
import cv2
from PIL import Image
import requests
import json
ocr = PaddleOCR(use_angle_cls=True,lang='en')
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
        script_directory = os.path.dirname(os.path.abspath(__file__))
        temp_images_subdirectory = "temp_images"
        output_dir = os.path.join(script_directory, temp_images_subdirectory)
        os.makedirs(output_dir, exist_ok=True)
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        image_path_list = []
        os.chdir(output_dir)
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render(scale=300/72).to_pil()
            image_path = f"image_{page_number + 1}.png"
            pil_image.save(image_path)
            image_path_list.append(image_path)
        detected_text_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            font_path = r"latin.ttf"
            im_show = draw_ocr(image, boxes, font_path=font_path)
            pil_im_show = Image.fromarray(im_show)
            image_buffer = io.BytesIO()
            pil_im_show.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            return send_file(image_buffer, mimetype='image/png')
        os.remove(file_path)
        os.remove(temp_images_subdirectory)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)