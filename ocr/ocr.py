
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import os
import torch
import gc
import google.generativeai as genai
from PIL import Image

gc.collect()
torch.cuda.empty_cache()

app = FastAPI()

# Configure Google Generative AI
API_KEY = "AIzaSyBqdDrkQ1kcnaKZjLBjaVthxaAPOlS60xk"
genai.configure(api_key=API_KEY)

def ocr_with_paddle(img):
    finaltext = ''
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    result = ocr.ocr(img)
    
    for i in range(len(result[0])):
        text = result[0][i][1][0]
        finaltext += ' ' + text
    return finaltext

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text_results = []
    
    for page_number, page in enumerate(pages):
        image_path = f'temp_page_{page_number}.jpg'
        page.save(image_path, 'JPEG')
        page_text = ocr_with_paddle(image_path)
        text_results.append(page_text)
        os.remove(image_path)
    
    return text_results

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return ocr_with_paddle(image_path)

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), process_type: str = Form(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    file_ext = os.path.splitext(file_location)[1].lower()
    if file_ext in ['.pdf']:
        texts = extract_text_from_pdf(file_location)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        texts = [extract_text_from_image(file_location)]
    else:
        return JSONResponse(content={"message": "Unsupported file type."}, status_code=400)

    os.remove(file_location)

    model = genai.GenerativeModel('gemini-1.5-flash')

    if process_type == "work_permit_fee":
        prompt = f''' {texts} ; create a single layer json of Here are all the keys (names on the left side) from the provided JSON:

        1. Amount
        2. Attempted On
        3. Employee
        4. Employer
        5. Employee ID
        6. Employer ID
        7. Paid At
        8. Paid Date
        9. Paid On
        10. Payment Duration (Months)
        11. Payment From
        12. Payment Method
        13. Payment Number
        14. Payment Set Details
        15. Payment To
        16. Payment Type
        17. Payment Type Description
        18. Print Date
        19. Receipt Number
        20. Remarks
        21. Site Name
        22. Status
        23. Total Amount
        24. Work Permit Fee
        25. Work Permit Fee Description
        26. Work Permit Number in alphabetical order with above data '''

    elif process_type == "work_permit_card":
        prompt = f''' {texts} ; create a single layer in alphabetical order in json of Here are all the keys (names on the left side) from the provided JSON:

        1. Name
        2. Nationality
        3. Gender (Male M or Female F)
        4. Date of Birth
        5. Passport No. and its a 8 digit alphanumeric
        6. Profession
        7. Work Permit No.
        8. Card Issued Date
        '''

    elif process_type == "work_permit_entry_pass":
         prompt = f''' {texts} ; create a single layer in alphabetical order in json of Here are all the keys (names on the left side) from the provided JSON:

        1. Employee Name
        2. Passport
        3. Gender
        4. Date of Birth
        5. Nationality
        6. Entry Pass Number
        7. Employer
        8. Employer Registeration Number
        9. Occupation
        10. Basic Salary
        11. Work Site
        12. Accomodation Address
        13. Entry Pass Issued Date
        14. Last Entry Allowed
        15. Printed On
 '''

    elif process_type == "insurance":
         prompt = f''' {texts} ; create a single layer in alphabetical order in json of Here are all the keys (names on the left side) from the provided JSON:

1. CLASS OF TAKAFUL
2. Date of Issue
3. CERTIFICATE NO
4. THE PARTICIPANT
5. BUSINESS REGN. NO
6. PERIOD OF TAKAFUL 
7. TOTAL SUM COVERED
8. TOTAL TAKAFUL CONTRIBUTION
9. CURRENCY
10. DESCRIPTION OF COVER
11. PERSON COVERED 
12. PASSPORT NO / NIC NO
13. D.O.B 
14. NATIONALITY
 '''
    elif process_type == "insurance_recipt":
         prompt = f''' {texts} ; create a single layer in alphabetical order in json of Here are all the keys (names on the left side) from the provided JSON:

1. Name as per in the Policy Certificate
2. Payment Date
3. NIC /Bus. Registration No
4. Contact No
5. Certification No 
6. Amount
 '''
    elif process_type == "passport":
        prompt = f''' {texts} ; create a single layer in alphabetical order in json of Here are all the keys (names on the left side) from the provided JSON:

1. Visa No ( 5digit/year )
2. Valid from 
3. Valid until 
4. No of Entries
5. Visa Type
6. Place of Issue
7. Name of Person
8. Gender
9. Nationality
10. Passport No
11. Date of Birth
12. Details 
 '''
    response = model.generate_content(prompt)

    import json
    import re

    json_regex = re.compile(r'json\n({.*?})\n', re.DOTALL)
    match = json_regex.search(response.text)

    if match:
        json_str = match.group(1)
        data = json.loads(json_str)

        storage_id = 1
        # while os.path.exists(f'storage_{storage_id}.json'):
            # storage_id += 1

        with open(f'{process_type}_{storage_id}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return JSONResponse(content={"message": f"JSON data saved to {process_type}_{storage_id}.json"})

    else:
        return JSONResponse(content={"message": "No JSON object found in the text."}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# to run app use this command : uvicorn ocr:app --reload