import gradio as gr
import requests, base64, cv2
import numpy as np
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()
API_URL = os.getenv("API_URL")
LOG_FILE = "logs.json"
def append_log(entry, log_file=LOG_FILE):
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []

    logs.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def query_backend(prompt1, prompt2, user_query, building, apartment, floor):
    payload = {
        "prompt_1": prompt1,
        "prompt_2": prompt2,
        "query": user_query.strip("\n").strip(" "),
        "building_code": building,
        "apartment_number": int(apartment),
        "floor_number": int(floor),
    }
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        step1 = data["step1"]
        step2 = data["step2"]

        def decode_img(b64):
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        imgs = [
            (decode_img(data["map_original"]), "Bản đồ gốc"),
            (decode_img(data["map_cropped"]), "Bản đồ phóng to"),
            (decode_img(data["floorplan_original"]), "Thiết kế gốc"),
            (decode_img(data["floorplan_cropped"]), "Thiết kế phóng to"),
        ]

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "building_code": building,
            "apartment_number": int(apartment),
            "floor_number": int(floor),
            "user_query": user_query,
            "ans1": step1,
            "ans2": step2,
        }
        append_log(log_entry)
        return f"```json\n{step1}\n```", step2, imgs
    else:
        return f"❌ Lỗi backend: {resp.text}", []

css_style = """
#scroll-md {
    max-height: 400px;
    overflow-y: auto;
    height: 600px;
    font-size: 16px;
}
"""
with gr.Blocks(css=css_style) as demo:
    with gr.Row():
        with gr.Column(scale=3): 
            building = gr.Textbox(label="Mã tòa nhà", value="S6.06")
            apartment = gr.Number(label="Số căn hộ", value=7, precision=0)
            floor = gr.Number(label="Số tầng", value=20, precision=0)

        with gr.Column(scale=4):
            prompt1 = gr.Textbox(label="Prompt Bước 1", value="default", lines=3)
            prompt2 = gr.Textbox(label="Prompt Bước 2", value="default", lines=3)

        with gr.Column(scale=3):
            query = gr.Textbox(label="Nhập câu hỏi", lines=3, value="Lô gia của căn hộ hướng ra khu vực nào")
            btn = gr.Button("Gửi", variant="primary")

    with gr.Row():
        with gr.Column(scale=5): 
            gallery = gr.Gallery(
                label="Hình ảnh", 
                columns=2, 
                height=500,
                object_fit="contain"
            )
            
        with gr.Column(scale=7):
            with gr.Tab("Phân tích căn hộ"):
                output1 = gr.Markdown(
                    value="### Bước 1: Phân tích căn hộ",
                    elem_id="scroll-md"
                )
            with gr.Tab("Đối chiếu bản đồ"):
                output2 = gr.Markdown(
                    value="### Bước 2: Tinh chỉnh trong bản đồ thực tế",
                    elem_id="scroll-md"
                )

    btn.click(
        query_backend,
        [prompt1, prompt2, query, building, apartment, floor],
        [output1, output2, gallery]
    )
    query.submit(
        query_backend,
        [prompt1, prompt2, query, building, apartment, floor],
        [output1, output2, gallery]
    )


if __name__ == "__main__":
    demo.launch(share=True)