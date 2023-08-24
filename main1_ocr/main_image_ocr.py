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
app=Flask(__name__)
@app.route("/ml-service/ocr/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/ocr", methods=["POST"])
def image_text_conversion():
    if request.method == "POST":
        image_file = request.files['Image']
        image_data = image_file.read()
        fields_input = request.form["fields"]
        table_pattern_input = request.form["table_pattern"]
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        #fields = json.loads(fields_input)
        result = ocr.ocr(gray)
        detected_text_list = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                detected_text_list.append(line[1][0])
        text = ' '.join(detected_text_list)
        data = {}
        if fields_input:
            fields = json.loads(fields_input)
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
        if table_pattern_input:
            table_pattern = json.loads(table_pattern_input)
            matched_data = {}
            for f in table_pattern:
                key1 = f.get("key")
                pattern1 = f.get("pattern")
                repeatable1 = f.get("repeatable", True)
                if pattern1:
                    matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
                    if matches1:
                        matched_data[key1] = matches1
            output_data = []
            keys = list(matched_data.keys())
            if keys:
                max_entries = max(len(matched_data[key]) for key in keys)
                for i in range(max_entries):
                    entry = {}
                    for key in keys:
                        if i < len(matched_data[key]):
                            entry[key] = matched_data[key][i]
                    output_data.append(entry)
                data['table_data'] = output_data
            else:
                data['table_data'] = "No matching data found for table patterns."
        if data:
            return jsonify(data)
        else:
            return jsonify({"error": "No matching data found"})
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)