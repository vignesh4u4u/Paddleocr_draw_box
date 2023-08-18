from flask import Flask,request,render_template,jsonify,json
from pdf2image import convert_from_path
from paddleocr import PaddleOCR,draw_ocr
import pytesseract as pytes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import easyocr
import os
import re
import cv2
from PIL import Image
import requests
import json
ocr = PaddleOCR(use_angle_cls=True,lang='en')
pytes.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path=r"C:\Users\VigneshSubramani\Documents\poppler-23.08.0\Library\bin"
output_dir=r"C:\Users\VigneshSubramani\Desktop\MAIN PROJECT\OCR_METHOD\Image"
app=Flask(__name__,template_folder="template")
@app.route("/ml-service/PdfOCR/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/PdfOCR", methods=["POST"])
def extract_text_information_pdf():
    if request.method == 'POST':
        file = request.files["file"]
        fields_input = request.form["fields"]
        #table_pattern_input = request.form["table_pattern"]
        file_path = "temp.pdf"
        file.save(file_path)
        with open(file_path,'rb') as f:
            pass
        pages=convert_from_path(pdf_path=file_path,poppler_path=poppler_path)
        detected_text_list = []
        for i, page in enumerate(pages):
            image_path = os.path.join(output_dir, f"page_{i + 1}.png")
            page.save(image_path, "PNG")
            gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    detected_text_list.append(line[1][0])
            text = ' '.join(detected_text_list)
            #print(text)
            data = {}
            for field in fields:
                key = field.get("key")
                pattern = field.get("pattern")
                repeatable = field.get("repeatable", True)
                if pattern:
                    matches = re.findall(pattern, text, flags=re.IGNORECASE)
                    if matches:
                        if repeatable:
                            data[key] = matches
                        else:
                            data[key] = matches[0]
            if data:
                return jsonify(data)
            else:
                return jsonify({"error": "No matching data found"})
        os.remove(file_path)
    #return render_template("che.html", **locals())
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)