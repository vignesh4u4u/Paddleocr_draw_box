from flask import Flask,request,render_template
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import easyocr
import seaborn as sns
import requests
import cv2
import numpy as np
import re
import os
import io
reader=easyocr.Reader(lang_list=["en"])
image_path=r"C:\Users\VigneshSubramani\Desktop\VS CODE\image to text\image to text main\sample1.jpg"
image_url = r"https://imgv3.fotor.com/images/blog-cover-image/How-to-Make-Text-Stand-Out-And-More-Readable.jpg"
response = requests.get(image_url).content
image1 = Image.open(io.BytesIO(response))
image = cv2.imread(image_path)
results=reader.readtext(image)
text =""
for result in results:
    text=result[1]
    print(text)