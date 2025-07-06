from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import numpy as np
import sqlite3
import datetime
import time
import psutil
import onnxruntime

app = Flask(__name__)

# -------------------------
# 1. Inisialisasi Video File & Model ONNX
# -------------------------
video_path = "C:/Users/didan/skripsi/vid.mp4"
camera = cv2.VideoCapture(video_path)
if not camera.isOpened():
    print("[ERROR] Video file tidak bisa dibuka!")
else:
    print("[INFO] Video file dibuka dengan sukses.")

# Load ONNX model (2 kelas: fire=0, person=1)
session = onnxruntime.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

# Variabel global untuk event-based logging
last_person_count = 0
last_fire_detected = 0

# Dinamis input name ONNX
input_name = session.get_inputs()[0].name

# -------------------------
# 2. Helper: IOU & NMS
# -------------------------
def iou(boxA, boxB):
    x1A,y1A,x2A,y2A = boxA[:4]
    x1B,y1B,x2B,y2B = boxB[:4]
    interX1 = max(x1A, x1B); interY1 = max(y1A, y1B)
    interX2 = min(x2A, x2B); interY2 = min(y2A, y2B)
    interW = max(0, interX2 - interX1); interH = max(0, interY2 - interY1)
    interArea = interW * interH
    areaA = (x2A-x1A)*(y2A-y1A); areaB = (x2B-x1B)*(y2B-y1B)
    union = areaA + areaB - interArea
    return interArea/union if union>0 else 0

def nms_boxes(dets, iou_th=0.45):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    keep=[]
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best,d) < iou_th]
    return keep

# -------------------------
# 3. Preprocess & Deteksi ONNX (robust)
# -------------------------
def preprocess(frame, input_size=416):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1)[None,...].astype(np.float32) / 255.0
    return img

def detect_person_and_fire(frame, conf_th=0.2, iou_th=0.45, input_size=416):
    H, W = frame.shape[:2]
    scale_x = W / input_size
    scale_y = H / input_size

    # preprocess + infer
    inp = preprocess(frame, input_size)
    outputs = session.run(None, {input_name: inp})
    preds = outputs[0]
    if preds.ndim == 3:
        preds = preds[0]

    raw_boxes = []
    K = preds.shape[1]
    # branch raw YOLOv5 output (x,y,w,h in pixel of 416×416 + obj_conf + cls_conf...)
    if K >= 7:
        for det in preds:
            x, y, w, h, obj_conf, *cls_conf = det.tolist()
            cls_conf = np.array(cls_conf)
            cls_id = int(cls_conf.argmax())
            conf = obj_conf * cls_conf[cls_id]
            if conf < conf_th:
                continue
            # konversi center-xywh (416×416) → xyxy di frame asli
            x1 = int((x - w/2) * scale_x)
            y1 = int((y - h/2) * scale_y)
            x2 = int((x + w/2) * scale_x)
            y2 = int((y + h/2) * scale_y)
            raw_boxes.append([x1, y1, x2, y2, conf, cls_id])
    else:
        # (jika K==6 atau format lain, tambahkan branch di sini)
        return 0, 0, "normal"

    # NMS
    boxes = nms_boxes(raw_boxes, iou_th)

    # gambar & hitung
    person_count = 0
    fire_detected = 0
    for x1, y1, x2, y2, conf, cls in boxes:
        if cls == 1:
            person_count += 1
            color = (0,255,0); label = f"person {conf:.2f}"
        elif cls == 0:
            fire_detected = 1
            color = (0,0,255); label = f"fire {conf:.2f}"
        else:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Crowd status: 3 kategori ---
    if person_count <= 10:
        crowd_status = "Normal"
    elif person_count <= 20:
        crowd_status = "Less Crowded"
    else:
        crowd_status = "Crowded"

    return person_count, fire_detected, crowd_status


# -------------------------
# 4. Logging ke Database
# -------------------------
def log_detection(person_count, fire_detected, crowd_status):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    c.execute('''
        INSERT INTO log_detections (timestamp, person_count, fire_detected, crowd_status)
        VALUES (?, ?, ?, ?)
    ''', (ts, person_count, fire_detected, crowd_status))
    conn.commit()
    conn.close()
    print(f"[LOG] person={person_count}, fire={fire_detected}, status={crowd_status}")

# -------------------------
# 5. Streaming + Event Logging + FPS/Resource Overlay
# -------------------------
latencies = []

def gen_frames():
    global last_person_count, last_fire_detected, latencies
    prev_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # -----------------------------
        # 1. Ukur latency inferensi
        # -----------------------------
        t0 = time.time()
        p_cnt, f_det, status = detect_person_and_fire(frame)
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        # Batasi history agar tidak terlalu panjang
        if len(latencies) > 300:
            latencies.pop(0)

        # -----------------------------
        # 2. Hitung statistik latency & jitter
        # -----------------------------
        avg_latency = sum(latencies) / len(latencies)
        import statistics
        jitter = statistics.pstdev(latencies)

        # -----------------------------
        # 3. Hitung FPS & Resource
        # -----------------------------
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        # -----------------------------
        # 4. Overlay semua informasi
        # -----------------------------
        cv2.putText(frame, f"Person: {p_cnt}",    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}",     (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"CPU: {cpu:.1f}%",    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Mem: {mem:.1f}%",    (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        cv2.putText(frame, f"Latency: {avg_latency:.1f} ms", (20,135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Jitter: {jitter:.1f} ms",       (20,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # -----------------------------
        # 5. Event logging & reset counters
        #    (sesuai kode kamu sebelumnya)
        # -----------------------------
        diff    = p_cnt - last_person_count
        changed = (f_det != last_fire_detected)
        if abs(diff) >= 2 or changed:
            log_detection(p_cnt, f_det, status)
            last_person_count  = p_cnt
            last_fire_detected = f_det
        else:
            if p_cnt < last_person_count:  last_person_count  = p_cnt
            if changed:                    last_fire_detected = f_det

        # -----------------------------
        # 6. Encode & yield frame
        # -----------------------------
        ret2, buf = cv2.imencode('.jpg', frame)
        if not ret2:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# -------------------------
# 6. Flask Endpoints
# -------------------------
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/latest_log')
def api_latest_log():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT crowd_status, fire_detected FROM log_detections ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        cs, fd = row
        fs = "Fire detected!" if fd==1 else "No fire detected"
    else:
        cs, fs = "normal", "No fire detected"
    return jsonify({"crowd_status": cs, "fire_status": fs})

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/history')
def history():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("""SELECT id, timestamp, person_count, fire_detected, crowd_status
                 FROM log_detections ORDER BY id DESC LIMIT 20""")
    rows = c.fetchall(); conn.close()
    html = "<h1>Riwayat Deteksi</h1><table border=1><tr><th>ID</th><th>Waktu</th><th>Person</th><th>Fire</th><th>Status</th></tr>"
    for r in rows:
        fire_txt = "Fire detected!" if r[3]==1 else "No fire detected"
        html += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{fire_txt}</td><td>{r[4]}</td></tr>"
    html += "</table><p><a href='/dashboard'>Kembali ke Dashboard</a></p>"
    return html

# -------------------------
# 7. Main
# -------------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask on port 5000…")
    app.run(debug=True, host='127.0.0.1', port=5000)
