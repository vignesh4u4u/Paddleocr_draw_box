from flask import Flask,request,render_template,jsonify,json
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
app=Flask(__name__)
@app.route("/ml-service/OCR/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/OCR", methods=["POST"])
def image_text_conversion():
    if request.method == "POST":
        image_file = request.files['Image']
        image_data = image_file.read()
        fields_input = request.form["fields"]
        #print(fields_input)
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        fields = json.loads(fields_input)
        result = ocr.ocr(gray)
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
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
    #return render_template("new.html", **locals())
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)