from pdf2jpg import pdf2jpg
from pdf2image import convert_from_path
poppler_path=r"C:\Users\VigneshSubramani\Documents\poppler-23.08.0\Library\bin"
pdf_path=r"C:\Users\VigneshSubramani\Pictures\invoice pdf\sample1.pdf"
pages=convert_from_path(pdf_path=pdf_path,poppler_path=poppler_path)
output_dir = r"C:\Users\VigneshSubramani\Documents\IMAGE"
for i, page in enumerate(pages):
    image_path = f"{output_dir}\\page_{i + 1}.png"
    page.save(image_path, "PNG")
print("Images saved successfully.")
print(image_path)