"""
使用阿里云百炼（Model Studio / DashScope）的文生图 API：
- 官方为异步任务：先创建任务，再根据 task_id 轮询查询结果（PENDING→RUNNING→SUCCEEDED/FAILED）。
- 文档未提供“百分比进度”，因此我们**不显示进度条**，仅展示任务状态与耗时。
- 参考文档：
  * Step 1 创建任务：POST /services/aigc/text2image/image-synthesis（必须携带 X-DashScope-Async: enable）
  * Step 2 查询任务：GET   /tasks/{task_id}
"""
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import os, io, time, uuid, threading, requests
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import json

app = Flask(__name__)
CORS(app, resources=r'/*')

# ==================
# 配置
# ==================
# 地域：'intl' 新加坡；'cn' 北京（内地版）
REGION = os.getenv('DASHSCOPE_REGION', 'cn')  # 'intl' or 'cn'
BASE = 'https://dashscope-intl.aliyuncs.com/api/v1' if REGION=='intl' else 'https://dashscope.aliyuncs.com/api/v1'
API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-b83db532a7d3433faccfe191d70fc676')
MODEL  = os.getenv('DASHSCOPE_T2I_MODEL', 'wanx2.0-t2i-turbo')  # 如: wan2.2-t2i-flash / wan2.5-t2i-preview / wanx2.0-t2i-turbo
IMAGE_SIZE = os.getenv('DASHSCOPE_T2I_SIZE', '768*768')
SAVE_DIR = os.path.abspath('t2i_outputs')
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

# 任务表：job_id -> {status, total, done, items:[{subject, prompt, status(PENDING/RUNNING/SUCCEEDED/FAILED), started, finished, path, task_id, error}]}
JOBS = {}
LOCK = threading.Lock()


def _new_job(subjects, style):
    job_id = str(uuid.uuid4())
    with LOCK:
        JOBS[job_id] = {
            'status': 'pending',
            'total': len(subjects),
            'done': 0,
            'items': [
                {
                    'subject': s,
                    'prompt': f"{style}, {s}",
                    'status': 'PENDING',
                    'started': None,
                    'finished': None,
                    'path': None,
                    'task_id': None,
                    'error': None,
                } for s in subjects
            ]
        }
    return job_id


def _update_item(job_id, idx, **kw):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: return
        job['items'][idx].update(kw)
        # 汇总
        job['done'] = sum(1 for it in job['items'] if it['status'] in ('SUCCEEDED','FAILED'))
        if job['done'] == job['total']:
            job['status'] = 'error' if any(it['status']=='FAILED' for it in job['items']) else 'done'


# 简单 SSE 管道（仅在状态变化/阶段切换时推送）
from collections import defaultdict, deque
_waiters = defaultdict(list)  # job_id -> [deque]

def _sse_push(job_id, payload):
    # 附带 overall，便于前端关闭
    with LOCK:
        job = JOBS.get(job_id)
        if job:
            payload = {
                **payload,
                "overall": {"status": job["status"], "total": job["total"], "done": job["done"]}
            }
    for q in list(_waiters.get(job_id, [])):
        q.append(payload)


def _create_task(prompt: str, n=1):
    url = f"{BASE}/services/aigc/text2image/image-synthesis"
    headers = dict(HEADERS)
    headers['X-DashScope-Async'] = 'enable'  # 必须
    body = {
        'model': MODEL,
        'input': { 'prompt': prompt },
        'parameters': { 'size': IMAGE_SIZE, 'n': n }
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    j = r.json()
    if r.status_code != 200:
        raise RuntimeError(j.get('message') or f"HTTP {r.status_code}")
    out = j.get('output') or {}
    return out.get('task_id'), out.get('task_status')


def _query_task(task_id: str):
    url = f"{BASE}/tasks/{task_id}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    j = r.json()
    out = j.get('output') or {}
    return out  # 包含 task_status, results[], submit_time, scheduled_time, end_time 等


def _download_image_to_path(url: str, save_path: str):
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _worker(job_id, style, subjects):
    # 标记运行
    with LOCK:
        if JOBS.get(job_id):
            JOBS[job_id]['status'] = 'running'

    for idx, subject in enumerate(subjects):
        # 通用风格 + 主体，并附加便于抠图/分离的描述
        prompt = f"{style}, {subject}，有立体感, 背景简单，前景背景色差大, third dimension, simple background, high contrast between foreground and background"
        _update_item(job_id, idx, status='PENDING', started=time.time())
        _sse_push(job_id, {'item': {'index': idx, 'status': 'PENDING', 'subject': subject}})
        try:
            task_id, init_status = _create_task(prompt, n=1)
            _update_item(job_id, idx, task_id=task_id, status=init_status or 'PENDING')
            _sse_push(job_id, {'item': {'index': idx, 'status': init_status or 'PENDING', 'subject': subject}})

            # 轮询（官方建议 10s；这里采用 3s，更灵敏，若需降压可调大）
            last_status = init_status
            while True:
                time.sleep(3)
                out = _query_task(task_id)
                st = out.get('task_status') or 'UNKNOWN'
                if st != last_status:
                    _update_item(job_id, idx, status=st)
                    _sse_push(job_id, {'item': {'index': idx, 'status': st, 'subject': subject}})
                    last_status = st
                if st in ('SUCCEEDED','FAILED','CANCELED','UNKNOWN'):
                    break

            if last_status == 'SUCCEEDED':
                results = (out.get('results') or [])
                # 取第一张
                url = None
                for r in results:
                    if isinstance(r, dict) and r.get('url'):
                        url = r['url']; break
                if not url:
                    raise RuntimeError('no image url returned')
                fname = PurePosixPath(unquote(urlparse(url).path)).parts[-1]
                save_path = os.path.join(SAVE_DIR, f"{job_id}_{idx}_{fname}")
                _download_image_to_path(url, save_path)
                _update_item(job_id, idx, status='SUCCEEDED', finished=time.time(), path=save_path)
                _sse_push(job_id, {'item': {'index': idx, 'status': 'SUCCEEDED', 'subject': subject}})
            else:
                _update_item(job_id, idx, status='FAILED', finished=time.time(), error=out.get('message'))
                _sse_push(job_id, {'item': {'index': idx, 'status': 'FAILED', 'subject': subject, 'error': out.get('message')}})
        except Exception as e:
            _update_item(job_id, idx, status='FAILED', finished=time.time(), error=str(e))
            _sse_push(job_id, {'item': {'index': idx, 'status': 'FAILED', 'subject': subject, 'error': str(e)}})


@app.route('/t2i/', methods=['POST'])
def submit():
    data = request.get_json(force=True)
    style = (data.get('style') or '').strip()
    subjects = [str(s).strip() for s in (data.get('subjects') or []) if str(s).strip()]
    if not style:
        return jsonify({'code':0, 'error':'绘画风格(style)不能为空'}), 400
    if not subjects:
        return jsonify({'code':0, 'error':'至少需要一个主体对象(subject)'}), 400

    job_id = _new_job(subjects, style)

    # 准备 SSE 等待队列
    q = deque()
    _waiters[job_id].append(q)

    th = threading.Thread(target=_worker, args=(job_id, style, subjects), daemon=True)
    th.start()
    return jsonify({'code':1, 'job_id': job_id, 'total': len(subjects)})


@app.route('/t2i/progress/<job_id>')
def progress(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'code':0, 'error':'job_id not found'}), 404
        # 由于官方无百分比，这里仅返回状态与耗时（秒）
        def seconds(t):
            return round((time.time()-t),1) if t else 0
        items = []
        for it in job['items']:
            items.append({
                'subject': it['subject'],
                'prompt': it['prompt'],
                'status': it['status'],
                'elapsed': seconds(it['started']),
                'error': it['error'],
            })
        return jsonify({'code':1,'status':job['status'],'total':job['total'],'done':job['done'],'items':items})


@app.route('/t2i/events/<job_id>')
def events(job_id):
    def gen():
        if job_id not in _waiters:
            _waiters[job_id].append(deque())
        q = _waiters[job_id][-1]
        last = time.time()
        while True:
            while q:
                payload = q.popleft()
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"  # ✅ JSON + \n\n
                last = time.time()
                with LOCK:
                    job = JOBS.get(job_id)
                    if job and job['done'] == job['total']:
                        return
            if time.time()-last > 30:
                yield "data: " + json.dumps({"job_id": job_id, "heartbeat": int(time.time())}) + "\n\n"
                last = time.time()
            time.sleep(0.2)
    resp = Response(gen(), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route('/t2i/result/<job_id>/<int:idx>')
def get_result(job_id, idx):
    from flask import request
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'code':0, 'error':'job_id not found'}), 404
        if idx<0 or idx>=job['total']:
            return jsonify({'code':0, 'error':'index out of range'}), 400
        it = job['items'][idx]
        if it['status'] != 'SUCCEEDED' or not it['path']:
            return jsonify({'code':0, 'error': f"item not ready: {it['status']}"}), 202
        # 如果带 download=1，则按主体名导出文件名
        dl = request.args.get('download') == '1'
        fname = f"{it['subject']}.png"
        return send_file(it['path'], mimetype='image/png', as_attachment=dl, download_name=fname if dl else None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5023, debug=False)