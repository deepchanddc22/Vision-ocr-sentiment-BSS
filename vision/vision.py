
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
import re
from PIL import Image
import cv2
import json
from pydantic import BaseModel
import mysql.connector
from mysql.connector import errorcode
from fuzzywuzzy import process

app = FastAPI()

# Define device as CPU
device = torch.device("cpu")

# Model and processor configuration
model_id = "google/paligemma-3b-mix-224"

# Load model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Move model to CPU
model.to(device)

# Input text
input_text = '''Name the fruits/vegetables/products present in image and how many are they
'''

# Global variables to manage camera state and results
cap = None
camera_active = False
results_list = []

def capture_image():
    global cap, camera_active, results_list
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Webcam not accessible")
    
    while camera_active:
        ret, frame = cap.read()
        cv2.imshow('Press q to capture, Press e to exit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                result_dict = process_image(captured_image)
                results_list.append(result_dict)
            except Exception as e:
                print(f"Error processing image: {e}")
        elif key == ord('e'):
            camera_active = False
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_item_count(text):
    # Regular expression to match patterns like "6 oranges", "orange 6", "banana 7"
    pattern = r'(?P<count>\d+)\s*(?P<name>\w+)|(?P<name2>\w+)\s*(?P<count2>\d+)'
    match = re.search(pattern, text)
    
    if match:
        count = match.group('count') or match.group('count2')
        name = match.group('name') or match.group('name2')
        return name, int(count)
    else:
        raise ValueError("Input does not match the expected format")

def save_results_to_json(results, filename='results.json'):
    # Save the results to the file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def process_image(input_image):
    try:
        # Prepare inputs
        inputs = processor(text=input_text, images=input_image, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad():
            output = model.generate(**inputs, max_length=496)

        # Decode the entire output
        full_text = processor.decode(output[0], skip_special_tokens=True)

        # Split the text into input and output
        _, _, result = full_text.partition('\n\n')

        # Parse the result
        name, count = parse_item_count(result.strip())

        # Create result dictionary
        result_dict = {"name": name, "count": count}
        print(result_dict)

        return result_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_camera")
def start_camera(background_tasks: BackgroundTasks):
    global cap, camera_active, results_list
    if camera_active:
        raise HTTPException(status_code=400, detail="Camera is already active")

    cap = cv2.VideoCapture(0)
    camera_active = True
    results_list = []  # Reset the results list for a new session
    background_tasks.add_task(capture_image)
    return {"message": "Camera started"}

@app.post("/stop_camera")
def stop_camera():
    global camera_active, results_list
    if not camera_active:
        raise HTTPException(status_code=400, detail="Camera is not active")

    camera_active = False
    save_results_to_json(results_list)
    sku_finder()  # Save the accumulated results to JSON
    return {"message": "Camera stopped"}

class Item(BaseModel):
    item_name: str
    item_price: float
    item_sku: str

# MySQL connection details
db_config = {
    'user': 'deepchandoa',
    'password': '2050',
    'host': 'localhost',
    'database': 'items_database'
}

# Ensure database and table are created at startup
@app.on_event("startup")
def startup_event():
    try:
        cnx = mysql.connector.connect(
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host']
        )
        cursor = cnx.cursor()

        # Create database
        try:
            cursor.execute("CREATE DATABASE IF NOT EXISTS items_database")
        except mysql.connector.Error as err:
            print(err.msg)

        # Use the database
        cursor.execute("USE items_database")

        # Create table if it does not exist
        create_table_query = (
            "CREATE TABLE IF NOT EXISTS items ("
            "  item_id INT AUTO_INCREMENT PRIMARY KEY,"
            "  item_name VARCHAR(255) NOT NULL,"
            "  item_price DECIMAL(10, 2) NOT NULL,"
            "  item_sku VARCHAR(100) NOT NULL UNIQUE"
            ") ENGINE=InnoDB")

        try:
            cursor.execute(create_table_query)
        except mysql.connector.Error as err:
            print(err.msg)

        cnx.commit()
        cursor.close()
        cnx.close()
    except mysql.connector.Error as err:
        print(err)

@app.post("/data-entry")
async def create_item(item: Item):
    try:
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor()

        # Delete existing entry if it exists
        delete_query = "DELETE FROM items WHERE item_sku = %s"
        cursor.execute(delete_query, (item.item_sku,))

        # Insert new data
        insert_query = (
            "INSERT INTO items (item_name, item_price, item_sku) "
            "VALUES (%s, %s, %s)"
        )
        cursor.execute(insert_query, (item.item_name, item.item_price, item.item_sku))
        cnx.commit()

        cursor.close()
        cnx.close()
        return {"message": "Item inserted/replaced successfully."}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Error: {err}")

@app.post("/exit-data-entry")
async def exit_data_entry():
    return {"message": "Data entry operation completed."}


def sku_finder():

    # Load data from results.json
    with open('results.json', 'r') as file:
        items_list = json.load(file)

    # Connect to MySQL server
    try:
        cnx = mysql.connector.connect(
            user='deepchandoa',
            password='2050',
            host='localhost',
            database='items_database'
        )
        cursor = cnx.cursor()

        # Fetch all item names and SKUs from the database
        cursor.execute("SELECT item_name, item_sku FROM items")
        db_items = cursor.fetchall()
        db_items_dict = {item[0]: item[1] for item in db_items}

        # Update items with matched SKUs
        for item in items_list:
            item_name = item['name']

            # Find the closest match for item_name from the database
            matched_name, score = process.extractOne(item_name, db_items_dict.keys())
            if score > 80:  # Adjust threshold as needed
                matched_sku = db_items_dict[matched_name]
                item['sku'] = matched_sku
            else:
                print(f"No close match found for item '{item_name}'")
                item['sku'] = None

        # Save the updated items_list back to results.json
        with open('results.json', 'w') as file:
            json.dump(items_list, file, indent=4)

        print("Items updated with SKUs and saved to results.json.")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    finally:
        if cnx.is_connected():
            cursor.close()
            cnx.close()
            print("MySQL connection is closed.")


@app.post("/generate-bill")
async def generate_bill():
    with open('results.json', 'r') as file:
        items_list = json.load(file)

    # Connect to MySQL server
    try:
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor()
        total_amount = 0

        # Fetch all SKUs and prices from the database
        cursor.execute("SELECT item_sku, item_price FROM items")
        db_items = cursor.fetchall()
        db_items_dict = {item[0]: item[1] for item in db_items}

        for item in items_list:
            item_sku = item['sku']
            item_count = item['count']

            # Use the SKU to find the price from the database
            if item_sku in db_items_dict:
                item_price = db_items_dict[item_sku]
                total_amount += item_count * item_price
            else:
                print(f"No price found for item with SKU '{item_sku}'")

        cursor.close()
        cnx.close()
        return {"total_amount": total_amount}
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise HTTPException(status_code=500, detail="Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            raise HTTPException(status_code=500, detail="Database does not exist")
        else:
            raise HTTPException(status_code=500, detail=f"Error: {err}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)


