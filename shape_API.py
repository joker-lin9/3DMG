# app_progress.py
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json, base64, io, os, sys, threading, time, re
from contextlib import redirect_stderr
import torch
from PIL import Image

sys.path.insert(0, './hunyuan3d-2.1/hy3dshape')
sys.path.insert(0, './hunyuan3d-2.1/hy3dpaint')

# from hy3dshape.rembg import BackgroundRemover
from background_remover import BackgroundRemover

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# ======================
# 模型初始化（全局一次）
# ======================
model_root = 'checkpoint/Shape'
subfolder  = 'hunyuan3d-dit-v2-1'
use_safetensors = False
variant = 'fp16'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.float16 if device == 'cuda' else torch.float32

ext = 'safetensors' if use_safetensors else 'ckpt'
variant_suffix = '' if variant is None else f'.{variant}'
ckpt_name  = f'model{variant_suffix}.{ext}'
model_dir  = os.path.join(model_root, subfolder)
ckpt_path  = os.path.join(model_dir, ckpt_name)
config_path= os.path.join(model_dir, 'config.yaml')

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
    ckpt_path, config_path, device=device, dtype=dtype, use_safetensors=use_safetensors
)
rembg = BackgroundRemover(use_gpu=True)

# ======================
# 任务状态
# ======================
JOBS = {}  # job_id -> dict(status, progress, phase, message, glb_path/base64, error)
JOBS_LOCK = threading.Lock()
JOB_COUNTER = 0

def new_job():
    global JOB_COUNTER
    with JOBS_LOCK:
        JOB_COUNTER += 1
        job_id = str(JOB_COUNTER)
        JOBS[job_id] = {
            "status": "pending",     # pending|running|done|error
            "phase":  "init",        # Diffusion Sampling / Volume Decoding / ...
            "progress": 0,           # 0-100
            "message": "",
            "glb_path": None,
            "error": None,
        }
    return job_id

def update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)

# 解析 tqdm 行的简单正则（示例适配你给的两行格式）
RE_PCT   = re.compile(r'(\d+)%\|')             # 匹配百分比
RE_PHASE = re.compile(r'^(.*?):\s')            # "Diffusion Sampling:: ..." / "Volume Decoding: ..."
RE_COUNT = re.compile(r'\s(\d+)/(\d+)\s')      # 7134/7134

class CustomTqdmCapture(io.TextIOBase):
    """Capture tqdm writes to stderr, focusing on updating job progress with just percentage."""
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
        self._buf = ""

    def write(self, s):
        self._buf += s
        if '\r' in s or '\n' in s:
            line = self._buf.strip()
            self._buf = ""
            if not line:
                return len(s)

            # Match phase (e.g., "Diffusion Sampling" or "Volume Decoding")
            m_phase = RE_PHASE.search(line)
            phase = m_phase.group(1).strip() if m_phase else JOBS[self.job_id].get("phase", "running")

            # Match percentage (e.g., 45%)
            m_pct = RE_PCT.search(line)
            if m_pct:
                pct = int(m_pct.group(1))  # Extract the percentage value
                if phase == "Diffusion Sampling:":
                    update_job(self.job_id, progress=pct, phase=phase)  # Update Diffusion progress
                elif phase == "Volume Decoding":
                    update_job(self.job_id, progress=pct, phase=phase)  # Update Decoding progress
            else:
                # In case no percentage is found, just update the message field
                update_job(self.job_id, phase=phase, message=line)

        return len(s)

    def flush(self):
        pass


# ======================
# 运行任务（后台线程）
# ======================
def run_job(job_id, image: Image.Image, num_inference_steps: int):
    update_job(job_id, status="running", phase="Diffusion Sampling", progress=0)
    out_path = f"demo_{job_id}.glb"

    try:
        # 捕获 tqdm 输出（tqdm 默认写 stderr）
        with redirect_stderr(CustomTqdmCapture(job_id)):
            if image.mode == 'RGB':
                image = rembg(image).convert("RGBA")
            else:
                image = image.convert("RGBA")

            with torch.inference_mode():
                mesh = pipeline(image=image, num_inference_steps=num_inference_steps)[0]

        mesh.export(out_path)
        update_job(job_id, status="done", progress=100, phase="done", glb_path=os.path.abspath(out_path))
    except Exception as e:
        update_job(job_id, status="error", error=str(e), phase="error")

# ======================
# Flask
# ======================
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route("/shape/", methods=["POST"])
def shape_api():
    """提交任务：JSON {image: base64, num_inference_steps: 50}"""
    try:
        payload = request.get_json(force=True)
        img_b64 = payload.get("image")
        steps   = int(payload.get("num_inference_steps", 50))

        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes))

        job_id = new_job()
        th = threading.Thread(target=run_job, args=(job_id, image, steps), daemon=True)
        th.start()

        return jsonify({"code": 1, "job_id": job_id})
    except Exception as e:
        return jsonify({"code": 0, "error": str(e)}), 400

@app.route("/progress/<job_id>", methods=["GET"])
def progress(job_id):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
        if not info:
            return jsonify({"code": 0, "error": "job_id not found"}), 404
        return jsonify({
            "code": 1,
            "status": info["status"],
            "phase": info["phase"],
            "progress": info["progress"],
            "message": info["message"],
        })

@app.route("/events/<job_id>")
def sse_events(job_id):
    """SSE 实时推送进度：前端可用 EventSource 订阅"""
    def gen():
        last_msg = None
        while True:
            with JOBS_LOCK:
                info = JOBS.get(job_id)
            if not info:
                yield "event: error\ndata: {\"error\":\"job not found\"}\n\n"
                break
            payload = json.dumps({
                "status": info["status"],
                "phase": info["phase"],
                "progress": info["progress"],
                "message": info["message"],
            })
            if payload != last_msg:
                yield f"data: {payload}\n\n"
                last_msg = payload
            if info["status"] in ("done", "error"):
                break
            time.sleep(0.3)  # 推送频率
    return Response(gen(), mimetype="text/event-stream")

@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
        if not info:
            return jsonify({"code": 0, "error": "job_id not found"}), 404
        if info["status"] != "done":
            return jsonify({"code": 0, "error": f"job not done: {info['status']}"}), 202
        path = info["glb_path"]

    # 方式A：返回 base64（与之前一致）
    with open(path, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("utf-8")
    return jsonify({"code": 1, "glb": glb_b64})

    # 方式B：如果你想直接返回文件，请改为：
    # from flask import send_file
    # return send_file(path, mimetype="model/gltf-binary", as_attachment=True, download_name="demo.glb")

if __name__ == "__main__":
    # 开发模式
    app.run(host="0.0.0.0", port=8000)
