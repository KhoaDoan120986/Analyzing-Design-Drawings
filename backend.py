from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2, json, os, glob, base64
import time
from google import genai
from google.genai import types

def crop_from_original(resources, code):
    tiles_list = resources["mapping"][code]
    offsets = resources["metadata"]
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = 0, 0

    for tile_name in tiles_list:
        off = offsets[tile_name]
        x, y, w, h = off["x"], off["y"], off["width"], off["height"]

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    cropped = resources["image"][min_y:max_y, min_x:max_x]
    return cropped

def to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

def resize_for_web(img, max_size=768):
    h, w = img.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

app = FastAPI()

class ChatRequest(BaseModel):
    prompt_1: str
    prompt_2: str
    building_code: str
    apartment_number: int
    floor_number: int
    query: str

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
client = genai.Client(api_key=config["gemini_api_key"])
MODEL = config["MODEL"]
prompt_1 = config["prompt_1"]
prompt_2 = config["prompt_2"]

map_resources = {} 
with open("data/mapping.json", "r", encoding="utf-8") as f:
    map_resources["mapping"] = json.load(f)

with open("data/metadata.json", "r", encoding="utf-8") as f:
    map_resources["metadata"] = json.load(f)
    
map_resources["image"] = cv2.cvtColor(cv2.imread("data/map.JPG"), cv2.COLOR_BGR2RGB) 

root_dir = "data/design/"
resources = {}
for apartment in os.listdir(root_dir):
    apt_path = os.path.join(root_dir, apartment)
    if not os.path.isdir(apt_path):
        continue

    resources[apartment] = {}
    for floor in os.listdir(apt_path):
        floor_path = os.path.join(apt_path, floor)
        if not os.path.isdir(floor_path):
            continue

        res = {}
        mapping_file = os.path.join(floor_path, "mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r", encoding="utf-8") as f:
                res["mapping"] = json.load(f)

        metadata_file = os.path.join(floor_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                res["metadata"] = json.load(f)

        map_file = os.path.join(floor_path, "design.jpg")
        if os.path.exists(map_file):
            res["image"] = cv2.cvtColor(cv2.imread(map_file), cv2.COLOR_BGR2RGB) 

        resources[apartment][floor] = res

map_original_b64 = to_base64(resize_for_web(map_resources["image"]))

@app.post("/")
def chatbot_response(req: ChatRequest):
    try:
        floor_ranges = resources[req.building_code.replace(".", "")].keys()
        for i, fr in enumerate(floor_ranges):
            try:
                _, start, end = fr.split("_")
                start, end = int(start), int(end)
                if start <= req.floor_number <= end:
                    floor = fr
                    break
            except ValueError:
                    continue
            
        floor_resources = resources[req.building_code.replace(".", "")][floor]
        design_cropped = crop_from_original(floor_resources, f'CH{req.apartment_number:02d}')
        _, encoded_image = cv2.imencode('.jpg', design_cropped)
        design_part  = types.Part.from_bytes(data=encoded_image.tobytes(), mime_type="image/jpeg")

        floor_original_b64 = to_base64(resize_for_web(floor_resources["image"]))
        floor_cropped_b64 = to_base64(resize_for_web(design_cropped))

        prompt_step1 = prompt_1 if req.prompt_1 == "default" else req.prompt_1
        try: 
            response_1 = client.models.generate_content(
                model=MODEL,
                contents=[f"{prompt_step1}\nThông tin căn hộ cần phân tích: CH{req.apartment_number:02d}", design_part,],
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    top_p=0.9,
                    top_k=40,
                )
            )
            time.sleep(3)

        except Exception as e:
            print(e)

                
        map_cropped = crop_from_original(map_resources, req.building_code)
        map_cropped_b64 = to_base64(resize_for_web(map_cropped))

        _, encoded_image = cv2.imencode('.jpg', map_cropped)
        map_part  = types.Part.from_bytes(data=encoded_image.tobytes(), mime_type="image/jpeg")     
        
        prompt_step2 = prompt_2 if req.prompt_2 == "default" else req.prompt_2
        user_query = f"""{prompt_step2}
        Thông tin căn hộ cần phân tích:

        * Tòa nhà (Building): {req.building_code}
        * Căn hộ (Apartment): CH{req.apartment_number:02d}
        * Tầng (Floor): {req.floor_number}

        [MAP_IMAGE]: Đây là ảnh bản đồ tổng quan vị trí block/căn hộ trong khu đô thị.
        [FLOORPLAN_LAYOUT]: Đây là JSON mô tả bố cục thiết kế mặt bằng chi tiết của căn hộ CH{req.apartment_number:02d} trong Block {req.building_code}: {json.loads(response_1.text)}
        [USER_QUERY]: {req.query}
        """
        try: 
            response_2 = client.models.generate_content(
                model=MODEL,
                contents=[user_query, map_part,],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.9,
                    top_k=40,
                )
            )
        except Exception as e:
            print(e)
        return {
            "step1": response_1.text,
            "step2": response_2.text,
            "map_original": map_original_b64,
            "map_cropped": map_cropped_b64,
            "floorplan_original": floor_original_b64,
            "floorplan_cropped": floor_cropped_b64
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
if __name__ == "__main__":
    NGROK_AUTH_TOKEN = "31JYCDRSSloOw7lPnlEos7Y8sTv_5PUsnG81esTd4PAMccnDz"
    import uvicorn
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    tunnel = ngrok.connect(8000, bind_tls=True)
    print("🚀 ngrok public url:", tunnel.public_url)

    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"API_URL={tunnel.public_url}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)