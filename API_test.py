import base64, requests

# 读取本地图片转 base64
with open("demo.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": img_b64,
    "num_inference_steps": 50
}

res = requests.post("http://127.0.0.1:5013/shape/", json=payload)
data = res.json()

if data["code"] == 1:
    glb_b64 = data["glb"]
    with open("result.glb", "wb") as f:
        f.write(base64.b64decode(glb_b64))
    print("✅ 结果已保存 result.glb")
else:
    print("❌ 出错:", data.get("error"))
