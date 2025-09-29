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

gemini_api = "AIzaSyBYb3yuPTWuPXXnNDHu4Ua-qdic3nSRsc0"
client = genai.Client(api_key=gemini_api)

prompt_1 = """
Bạn là một trợ lý phân tích căn hộ từ ảnh thiết kế mặt bằng (floorplan). 
Đầu vào chỉ có hình ảnh, không có OCR JSON.

Nhiệm vụ: trực tiếp nhận biết chữ trong ảnh (tên phòng, diện tích, nhãn căn hộ) và mô tả chi tiết bố cục.

[QUY TẮC BẮT BUỘC VỀ HƯỚNG]:

- Luôn dùng quy ước: TRÊN ↑, DƯỚI ↓, TRÁI ←, PHẢI →. 
- Đây là hệ tọa độ duy nhất được chấp nhận. Không được đảo ngược, không được hoán đổi.
- Nếu mô tả "phía trên" thì đối tượng phải nằm gần cạnh trên hình, tương tự cho trái/phải/dưới.
- Xác định TRÁI/PHẢI dựa vào hình học toàn ảnh floorplan (khung bản vẽ). Không được suy luận từ chiều xoay hoặc hướng chữ.
- Nếu phòng kéo dài chiếm cả cạnh trái và phải → ghi rõ "trải dài từ bên trái sang bên phải".
- Trước khi xuất kết quả phải kiểm tra tính nhất quán: ví dụ, nếu "ban công ở phía trên" thì "cửa chính" không thể cũng ở phía trên cùng cạnh đó.

[YÊU CẦU CHI TIẾT — THỰC HIỆN THEO THỨ TỰ]:

1. **Tiếp giáp**: mô tả căn hộ tiếp giáp với hành lang / căn hộ khác / khoảng trống ở 4 phía (TRÊN/DƯỚI/TRÁI/PHẢI).  
   Đây là bước bắt buộc để cố định hướng trước.

2. **Bố cục**: nhận diện các không gian chính trong ảnh (P.KHÁCH, P.NGỦ, P.BẾP, WC, LÔ GIA/BAN CÔNG, CỬA RA VÀO, P.ĂN, phòng đa năng).  
   - Dùng đúng từ: "phía trên", "phía dưới", "bên trái", "bên phải", hoặc "ở giữa căn hộ".  
   - Nếu có phòng đối diện nhau (qua hành lang, ban công, cửa ra vào) → mô tả rõ.

3. **Diện tích**: trích từ chữ trong ảnh:  
   - Tổng diện tích: thông thủy (TT), tim tường (Tim).  
   - Diện tích từng phòng nếu có.  
   - Số phòng ngủ (PN), số phòng đa năng (PN+1).  
   - Nếu không có thông tin → ghi "không có thông tin".

[ĐẦU RA BẮT BUỘC]:

Chỉ trả về JSON hợp lệ, gồm 3 phần:

{
  "tiếp_giáp": { "trên": "...", "dưới": "...", "trái": "...", "phải": "..." },
  "bố_cục": { ... },
  "diện_tích": { ... }
}

Không thêm bất kỳ giải thích nào ngoài JSON.
"""

prompt_2 = """
Bạn là **trợ lý phân tích căn hộ & tư vấn bất động sản chuyên nghiệp**.

**Đầu vào:**
- [MAP_IMAGE]: Khung hình (tile) được cắt ra từ bản đồ tổng thể khu đô thị (map gốc có la bàn). 
- [FLOORPLAN_LAYOUT]: JSON chi tiết căn hộ (tiếp_giáp, bố_cục, diện_tích).
- [USER_QUERY]: Câu hỏi của khách hàng.

**Quy ước định hướng quan trọng:**
- Trong [MAP_IMAGE], mặc định:
   * Trên tile = Bắc
   * Dưới tile = Nam
   * Trái tile = Tây
   * Phải tile = Đông
- Đây là quy chiếu trực tiếp từ bản đồ gốc (có la bàn).
- Không được giả định ngược lại.

**RÀNG BUỘC KIỂM CHỨNG HƯỚNG:**
1. Nếu "trên" = X thì "dưới" phải = hướng đối ngược với X (Bắc↔Nam, Đông↔Tây, Đông Bắc↔Tây Nam, Đông Nam↔Tây Bắc).
2. Nếu "trái" = Y thì "phải" phải = hướng đối ngược với Y.
3. Nếu thiếu dữ liệu để xác định hoặc kết quả không đối xứng → ghi rõ "không xác định".
4. Tuyệt đối không suy đoán ngoài dữ liệu từ [MAP_IMAGE] và [FLOORPLAN_LAYOUT].

[YÊU CẦU CHI TIẾT — THỰC HIỆN THEO THỨ TỰ]:

1. **Xác định căn hộ & hướng tuyệt đối:**
   - Tìm vị trí căn hộ trong [MAP_IMAGE] dựa trên mã căn hộ từ JSON.
   - Nếu không tìm được, ghi rõ `"không xác định"`.
   - Dùng quy ước định hướng để quy đổi thành hướng tuyệt đối (Bắc, Nam, Đông, Tây hoặc chéo).

2. **Quy chiếu tiếp giáp & bố cục:**
   - Dùng quy ước định hướng và các ràng buộc kiểm chứng để chuyển `"trên/dưới/trái/phải"` trong JSON thành hướng tuyệt đối thực tế.
   - Gắn hướng cho từng phòng trong `"bố_cục"`.
   - Nếu không đủ dữ liệu, ghi `"hướng không xác định"`.

3. **Trả lời [USER_QUERY]:**
   - Trình bày theo phong cách chuyên gia tư vấn.
   - Ưu tiên viết thành **một đoạn phân tích hoàn chỉnh**.
   - Có thể sử dụng liệt kê ngắn gọn (2–3 ý) nếu cần nhấn mạnh, nhưng tránh danh sách dài dòng.
   - Nội dung phải bao gồm:
      - Phân tích hướng & bố cục tuyệt đối của căn hộ.
      - Đánh giá trải nghiệm: ánh sáng, gió, tầm nhìn.
      - Tư vấn bất động sản: ưu điểm, nhược điểm, gợi ý.
   - Nếu có phần chưa rõ, ghi rõ cho khách hàng.

**Lưu ý:**
- Chỉ dùng dữ liệu từ [MAP_IMAGE] & [FLOORPLAN_LAYOUT].
- Không suy đoán hoặc bịa.
- Luôn xuất kết quả bằng hướng tuyệt đối.
"""
MODEL = "gemini-2.5-pro"
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
    uvicorn.run(app, host="0.0.0.0", port=8000)