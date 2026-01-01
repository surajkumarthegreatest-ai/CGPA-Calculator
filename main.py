from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import numpy as np
import cv2
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/parse-grades")
async def parse_grades(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    hor_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=2)
    ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=2)

    joints = cv2.bitwise_and(hor_lines, ver_lines)

    config = '--psm 6'
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

    rows = {}
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and len(text) > 1:
            y = data['top'][i] // 10  
            if y not in rows:
                rows[y] = []
            rows[y].append(text)

    parsed_courses = []
    for y in sorted(rows.keys()):
        row_text = " ".join(rows[y])
        words = row_text.split()
        if len(words) >= 3:
            grade = words[-1].upper()
            credits = words[-2]
            name = " ".join(words[:-2])
            
            if grade in ['S','A','B','C','D','E','F'] and credits.isdigit():
                parsed_courses.append({
                    "name": name,
                    "credits": int(credits),
                    "grade": grade
                })

    return {"status": "success", "data": parsed_courses}