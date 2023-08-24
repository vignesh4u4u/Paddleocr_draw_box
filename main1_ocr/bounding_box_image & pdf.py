from flask import Flask, request,send_file
from paddleocr import PaddleOCR,draw_ocr
import cv2
import numpy as np
import pypdfium2 as pdfium
import os
import io
from PIL import Image
ocr = PaddleOCR(use_angle_cls=True,lang='en')
app = Flask(__name__)
@app.route("/ml-service/draw_box/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/draw_box", methods=["POST"])
def bounding_box():
    file = request.files["pdf_file"]
    image_file = request.files['image_file']
    if file:
        file = request.files["pdf_file"]
        file_path = "temp.pdf"
        file.save(file_path)
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        image_path_list = []
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render(scale=3).to_pil()
            image_path = f"image_{page_number + 1}.png"
            pil_image.save(image_path)
            image_path_list.append(image_path)
        processed_images = []
        pdf.close()
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            font_path = "latin.ttf"
            im_show = draw_ocr(image, boxes, font_path=font_path)
            pil_im_show = Image.fromarray(im_show)
            image_buffer = io.BytesIO()
            pil_im_show.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            processed_images.append(image_buffer)
            os.remove(image_path)
        os.remove(file_path)
        return send_file(processed_images[0], mimetype='image/png')
    if image_file:
        image_file = request.files['image_file']
        image_data = image_file.read()
        gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(gray)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]
        font_path = "latin.ttf"
        im_show = draw_ocr(image, boxes, font_path=font_path)
        pil_im_show = Image.fromarray(im_show)
        image_buffer = io.BytesIO()
        pil_im_show.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        return send_file(image_buffer, mimetype='image/png')
    else:
       return None
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
