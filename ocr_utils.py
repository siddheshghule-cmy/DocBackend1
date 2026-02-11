from paddleocr import PaddleOCR
import tempfile
import os

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

def extract_text_from_image(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = ocr.ocr(tmp_path)

        texts = []

        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])

        return " ".join(texts).lower()

    finally:
        os.remove(tmp_path)
