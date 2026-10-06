import os

# IP Webcam low-latency options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
    "framedrop;1|max_delay;0|reorder_queue_size;0|"
    "probesize;32|analyzeduration;0"
)

import cv2
import numpy as np
import json
import time
import threading
import tkinter as tk

from insightface.app import FaceAnalysis

# Optional: Emotion detection
try:
    from deepface import DeepFace
    EMOTION_AVAILABLE = True
except Exception:
    EMOTION_AVAILABLE = False

# Telegram
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


# ================= Config =================
DB_FILE      = "faces_db.json"
ENC_DIR      = "encodings"
TOLERANCE    = 0.45
TOL_SMALL    = 0.60
SCALE        = 0.5
DETECT_EVERY = 5
WIN_NAME     = "Face Recognition - InsightFace"
INIT_W, INIT_H = 960, 540

MIN_FACE_SIZE     = 12
SMALL_FACE_LIMIT  = 120
ENHANCE_SMALL     = True

SMOOTH_HISTORY  = 5
SMOOTH_MIN_VOTE = 3
SMOOTH_FORGET   = 8

COLOR_KNOWN     = (0, 255, 0)
COLOR_UNCERTAIN = (0, 215, 255)
COLOR_UNKNOWN   = (0, 0, 255)

# Auto-Update (Multiple Templates)
MAX_TEMPLATES          = 3
UPDATE_MIN_DIST        = 0.22
UPDATE_MIN_VOTES       = 5
UPDATE_QUALITY_MARGIN  = 1.15
CROSS_CHECK_DIST       = 0.35

# Distance Estimate
AVG_FACE_WIDTH_CM   = 15.0
CAMERA_FOV_DEG      = 60.0

# Emotion Detection
EMOTION_INTERVAL    = 2.0

# Telegram
TELEGRAM_ENABLED    = True
TELEGRAM_TOKEN      = "8994893005:AAF1lXFdZJ1x-_hfxCk1qQ09Yr1UZLT5J14"                    
TELEGRAM_CHAT_ID    = "1833513500"                   
ALERT_COOLDOWN      = 30
UNKNOWN_ALERT       = True
KNOWN_ALERT         = False
SEND_PHOTO          = True

os.makedirs(ENC_DIR, exist_ok=True)

# ================= InsightFace =================
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(480, 480))


# ================= Distance Estimate =================
def estimate_distance(face_px_width, frame_w):
    """Rough distance in meters based on face pixel width."""
    try:
        fov_rad = np.radians(CAMERA_FOV_DEG)
        focal_px = frame_w / (2 * np.tan(fov_rad / 2))
        distance_cm = (AVG_FACE_WIDTH_CM * focal_px) / max(face_px_width, 1)
        return distance_cm / 100.0
    except Exception:
        return 0.0


# ================= Emotion Detection =================
def detect_emotion(face_crop):
    """Return (emotion_str, confidence) or (None, 0)."""
    if not EMOTION_AVAILABLE:
        return None, 0.0
    try:
        result = DeepFace.analyze(
            face_crop, actions=['emotion'],
            enforce_detection=False, silent=True
        )
        if isinstance(result, list):
            result = result[0]
        emo = result['dominant_emotion']
        conf = float(result['emotion'][emo])
        return emo, conf
    except Exception:
        return None, 0.0


# ================= Telegram =================
_tg_last_alert = {}
_tg_lock = threading.Lock()


def _tg_send(text, photo_bgr=None):
    """Send Telegram message in a background thread."""
    if not REQUESTS_AVAILABLE: return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return

    def _worker():
        try:
            base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
            if photo_bgr is not None and SEND_PHOTO:
                _, buf = cv2.imencode('.jpg', photo_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                requests.post(
                    f"{base}/sendPhoto",
                    data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text},
                    files={'photo': ('alert.jpg', buf.tobytes(), 'image/jpeg')},
                    timeout=15
                )
            else:
                requests.post(
                    f"{base}/sendMessage",
                    data={'chat_id': TELEGRAM_CHAT_ID, 'text': text},
                    timeout=15
                )
        except Exception as e:
            print(f"[TELEGRAM] {e}")

    threading.Thread(target=_worker, daemon=True).start()


def telegram_alert(key, text, photo_bgr=None):
    """Send alert with cooldown."""
    if not TELEGRAM_ENABLED: return
    now = time.time()
    with _tg_lock:
        if now - _tg_last_alert.get(key, 0) < ALERT_COOLDOWN:
            return
        _tg_last_alert[key] = now
    _tg_send(text, photo_bgr)


def send_startup_message():
    if not TELEGRAM_ENABLED: return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    _tg_send("Face Recognition started\n"
             f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# ================= JSON + NPY DB =================
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    db = []
    for item in raw:
        templates = []
        if "templates" in item:
            for t in item["templates"]:
                p = t["file"]
                if os.path.exists(p):
                    templates.append({
                        "encoding": np.load(p),
                        "file": p,
                        "quality": float(t.get("quality", 0.5)),
                    })
        elif "encoding_file" in item:
            p = item["encoding_file"]
            if os.path.exists(p):
                templates.append({
                    "encoding": np.load(p),
                    "file": p,
                    "quality": float(item.get("quality", 0.5)),
                })
        if templates:
            db.append({
                "name": item["name"],
                "age":  item.get("age", ""),
                "templates": templates,
            })
    return db


def save_db(db):
    readable = []
    for person in db:
        entry = {"name": person["name"], "age": person["age"], "templates": []}
        for t in person["templates"]:
            np.save(t["file"], t["encoding"])
            entry["templates"].append({
                "file":    t["file"],
                "quality": float(t["quality"]),
            })
        readable.append(entry)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(readable, f, indent=2, ensure_ascii=False)


def make_enc_filename(name, idx):
    safe = "".join(c if c.isalnum() else "_" for c in name)
    return os.path.join(ENC_DIR, f"{safe}_{idx}_{int(time.time()*1000)}.npy")


# ================= Tk root =================
root = tk.Tk()
root.withdraw()


# ================= Camera Source Menu =================
def ask_camera_source():
    result = {"type": None, "value": None}
    dlg = tk.Toplevel(root)
    dlg.title("Camera Source")
    dlg.geometry("440x280")
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(dlg, text="Choose Camera Source",
             font=("Arial", 13, "bold")).pack(pady=(14, 6))
    tk.Label(dlg, text="Mobile: open 'IP Webcam' app, tap Start Server,\n"
                       "then type the IP shown on the phone.",
             font=("Arial", 9), fg="gray", justify="center").pack(pady=(0, 12))

    def choose_pc():
        result["type"] = "pc"; result["value"] = 0; dlg.destroy()

    tk.Button(dlg, text="PC Camera", width=36, height=1,
              font=("Arial", 11), command=choose_pc).pack(pady=6)

    tk.Label(dlg, text="-------  or  -------",
             fg="gray", font=("Arial", 9)).pack(pady=6)

    frame = tk.Frame(dlg); frame.pack(pady=4)
    tk.Label(frame, text="Mobile IP:").grid(row=0, column=0, padx=4)
    ip_var = tk.StringVar(value="192.168.0.109:8080")
    tk.Entry(frame, textvariable=ip_var, width=22,
             font=("Courier", 11)).grid(row=0, column=1, padx=4)

    def choose_mobile():
        ip = ip_var.get().strip()
        if not ip: return
        for pfx in ("http://", "https://"):
            if ip.startswith(pfx): ip = ip[len(pfx):]
        if "/" in ip: ip = ip.split("/")[0]
        result["type"] = "mobile"
        result["value"] = f"http://{ip}/video"
        dlg.destroy()

    tk.Button(frame, text="Connect", width=10,
              command=choose_mobile).grid(row=0, column=2, padx=6)

    tk.Label(dlg, text="Example: 192.168.0.109:8080",
             font=("Arial", 8), fg="gray").pack(pady=6)
    tk.Button(dlg, text="Cancel", width=10,
              command=dlg.destroy).pack(pady=6)
    dlg.focus_force()
    root.wait_window(dlg)
    return result["type"], result["value"]


# ================= Add Details Form =================
def ask_details(face_no=None):
    res = {"name": None, "age": None}
    dlg = tk.Toplevel(root)
    dlg.title(f"Add - Unknown #{face_no}" if face_no else "Add Details")
    dlg.geometry("340x220")
    dlg.resizable(False, False)
    dlg.grab_set()

    if face_no:
        tk.Label(dlg, text=f"Adding details for Unknown #{face_no}",
                 font=("Arial", 10, "bold"), fg="blue").grid(
                 row=0, column=0, columnspan=2, pady=(10, 4))

    tk.Label(dlg, text="Name:").grid(row=1, column=0, padx=12, pady=10, sticky="e")
    ne = tk.Entry(dlg, width=22); ne.grid(row=1, column=1, padx=12, pady=10)
    tk.Label(dlg, text="Age:").grid(row=2, column=0, padx=12, pady=10, sticky="e")
    ae = tk.Entry(dlg, width=22); ae.grid(row=2, column=1, padx=12, pady=10)

    err = tk.Label(dlg, text="", fg="red"); err.grid(row=3, column=0, columnspan=2)

    def on_save():
        n, a = ne.get().strip(), ae.get().strip()
        if not n: err.config(text="Name is required!"); return
        if a and not a.isdigit(): err.config(text="Age must be numeric!"); return
        res["name"], res["age"] = n, a
        dlg.destroy()

    tk.Button(dlg, text="Save", width=10, command=on_save).grid(row=4, column=0, pady=10)
    tk.Button(dlg, text="Cancel", width=10, command=dlg.destroy).grid(row=4, column=1, pady=10)
    ne.focus_set()
    root.wait_window(dlg)
    return res["name"], res["age"]


# ================= Enhancement =================
def enhance_face(face_bgr):
    try:
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return face_bgr


def get_embedding_fast(face_crop):
    h, w = face_crop.shape[:2]
    if w < 112 or h < 112:
        scale = 112 / max(w, h)
        face_crop = cv2.resize(face_crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
    if ENHANCE_SMALL:
        face_crop = enhance_face(face_crop)
    try:
        faces = app.get(face_crop)
        if len(faces) == 0: return None
        return faces[0].embedding
    except Exception:
        return None


def face_quality_score(face_crop):
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        h, w = face_crop.shape[:2]
        size = min(w, h)
        bright = float(np.mean(gray))
        bright_score = 1.0 - abs(bright - 128) / 128.0
        blur_n = min(blur / 500.0, 1.0)
        size_n = min(size / 200.0, 1.0)
        return float(0.4 * blur_n + 0.4 * size_n + 0.2 * bright_score)
    except Exception:
        return 0.0


# ================= Label Smoother =================
class LabelSmoother:
    def __init__(self, history=5, min_vote=3, forget=8):
        self.history = history
        self.min_vote = min_vote
        self.forget = forget
        self.buffer = []
        self.last_known = None
        self.uncertain_streak = 0
        self.high_conf_count = 0
        self.best_new_enc    = None
        self.best_new_qual   = 0.0

    def update(self, current_label, is_known):
        self.buffer.append((current_label, is_known))
        if len(self.buffer) > self.history:
            self.buffer.pop(0)

        if is_known:
            self.uncertain_streak = 0
            self.last_known = current_label
            return current_label, "known"

        self.high_conf_count = 0

        votes = {}
        for lbl, k in self.buffer:
            if k: votes[lbl] = votes.get(lbl, 0) + 1

        if votes:
            best_label = max(votes, key=votes.get)
            if votes[best_label] >= self.min_vote:
                self.uncertain_streak += 1
                if self.uncertain_streak < self.forget:
                    return best_label, "uncertain"
                else:
                    self.last_known = None
                    return "Unknown", "unknown"

        self.uncertain_streak += 1
        if self.uncertain_streak < self.forget and self.last_known:
            return self.last_known, "uncertain"

        self.last_known = None
        return "Unknown", "unknown"


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter + 1e-10)


def recognize(enc, db, is_small):
    thr = TOL_SMALL if is_small else TOLERANCE
    best_idx = -1; best_dist = 2.0
    for p_idx, person in enumerate(db):
        for t in person["templates"]:
            d = cosine_dist(enc, t["encoding"])
            if d < best_dist:
                best_dist = d; best_idx = p_idx

    if best_idx >= 0 and best_dist <= thr:
        p = db[best_idx]
        label = p["name"]
        if p["age"]: label += f" ({p['age']})"
        return best_idx, best_dist, label, True
    return -1, best_dist, "Unknown", False


def try_auto_update(db, person_idx, new_enc, new_quality):
    if person_idx < 0 or person_idx >= len(db): return False
    person = db[person_idx]
    templates = person["templates"]

    best_q = max(t["quality"] for t in templates)
    if new_quality < best_q * UPDATE_QUALITY_MARGIN:
        return False

    for p_idx, other in enumerate(db):
        if p_idx == person_idx: continue
        for t in other["templates"]:
            if cosine_dist(new_enc, t["encoding"]) < CROSS_CHECK_DIST:
                return False

    idx = len(templates)
    fname = make_enc_filename(person["name"], idx)
    templates.append({
        "encoding": new_enc.copy(), "file": fname,
        "quality": float(new_quality),
    })

    if len(templates) > MAX_TEMPLATES:
        anchor = templates[0]
        rest = templates[1:]
        rest.sort(key=lambda x: x["quality"], reverse=True)
        keep = rest[:MAX_TEMPLATES - 1]
        for t in rest[MAX_TEMPLATES - 1:]:
            try: os.remove(t["file"])
            except Exception: pass
        person["templates"] = [anchor] + keep

    save_db(db)
    print(f"[AUTO-ADD] {person['name']} template #{len(person['templates'])} "
          f"(q: {new_quality:.2f})")
    return True


# ================= Drawing =================
def draw_box(frame, top, right, bottom, left, color, label, extra="", scale=1.0):
    thickness = max(2, int(2 * scale))
    font_scale = max(0.55, 0.6 * scale)
    pad = int(30 * scale)

    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    if top - pad > 0:
        cv2.rectangle(frame, (left, top - pad), (right, top), color, cv2.FILLED)
        ty = top - int(8 * scale)
    else:
        cv2.rectangle(frame, (left, bottom), (right, bottom + pad), color, cv2.FILLED)
        ty = bottom + int(22 * scale)

    text_color = (255, 255, 255) if color != COLOR_UNCERTAIN else (0, 0, 0)
    cv2.putText(frame, label, (left + int(6 * scale), ty),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color,
                max(1, thickness - 1))
    if extra:
        cv2.putText(frame, extra, (left + int(6 * scale), ty + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, text_color, 1)


def draw_header(frame, total, known, uncertain, unknown, src_label="", tg_on=False):
    h, w = frame.shape[:2]
    bar_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    fs = max(0.45, min(0.7, w / 1400))
    pad = int(20 * (fs / 0.7))
    font = cv2.FONT_HERSHEY_DUPLEX
    x = pad
    y = int(bar_h * 0.68)

    def put(txt, color, thick=1):
        nonlocal x
        cv2.putText(frame, txt, (x, y), font, fs, color, thick)
        (tw, _), _ = cv2.getTextSize(txt, font, fs, thick)
        x += tw + pad * 2

    put(f"Total: {total}", (255, 255, 255), 1)
    put(f"Known: {known}", COLOR_KNOWN, 2)
    if uncertain > 0:
        put(f"Uncertain: {uncertain}", COLOR_UNCERTAIN, 2)
    put(f"Unknown: {unknown}", COLOR_UNKNOWN, 2)

    right_txt = f"[{src_label}]"
    if tg_on: right_txt += "  TG"
    (tw, _), _ = cv2.getTextSize(right_txt, font, fs * 0.8, 1)
    cv2.putText(frame, right_txt, (w - tw - 15, y),
                font, fs * 0.8, (200, 200, 200), 1)


def draw_footer(frame, unknown_count):
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if unknown_count > 0:
        hint = (f"Press 1-{unknown_count} = add   |   S = save all   |   "
                f"F = fullscreen   |   Q = quit")
    else:
        hint = "F = fullscreen   |   Q = quit"

    fs = max(0.4, min(0.55, w / 1800))
    cv2.putText(frame, hint, (14, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 220, 220), 1)


# ================= Camera Thread =================
class CameraThread:
    def __init__(self, source, is_network=False):
        self.source = source
        self.is_network = is_network
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = True
        self._open()
        for _ in range(50):
            if self.ret and self.frame is not None: break
            time.sleep(0.1)
        threading.Thread(target=self._update, daemon=True).start()

    def _open(self):
        try:
            if self.is_network:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap is not None and self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                self.ret = False
        except Exception:
            self.ret = False

    def _update(self):
        fails = 0
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.is_network:
                        time.sleep(1.0); self._open()
                    continue
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret, self.frame = ret, frame
                    fails = 0
                else:
                    fails += 1
                    if self.is_network and fails > 30:
                        try: self.cap.release()
                        except: pass
                        time.sleep(0.5); self._open(); fails = 0
                    else:
                        time.sleep(0.01)
            except Exception:
                time.sleep(0.1)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        try:
            if self.cap is not None: self.cap.release()
        except: pass


# ================= Helper =================
def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a, b))


# ================= Main =================
def main():
    src_type, src_value = ask_camera_source()
    if src_type is None:
        print("[INFO] Cancelled.")
        return

    if src_type == "pc":
        src_label = "PC"
        cam = CameraThread(0, is_network=False)
    else:
        src_label = "Mobile"
        print(f"[INFO] Connecting to: {src_value}")
        cam = CameraThread(src_value, is_network=True)
        time.sleep(1)

    if not cam.ret:
        print("[ERROR] Could not open camera/stream!")
        return

    db = load_db()
    total_t = sum(len(p["templates"]) for p in db)
    print(f"[INFO] DB: {len(db)} persons, {total_t} templates")
    print(f"[INFO] Emotion: {'ON' if EMOTION_AVAILABLE else 'OFF (deepface not installed)'}")
    print(f"[INFO] Telegram: {'ON' if (TELEGRAM_ENABLED and TELEGRAM_TOKEN) else 'OFF'}")

    if TELEGRAM_ENABLED and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_startup_message()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, INIT_W, INIT_H)
    cv2.moveWindow(WIN_NAME, 80, 40)

    fullscreen   = False
    frame_count  = 0
    last_results = []
    trackers     = []
    last_emotion_time = 0.0
    frame_w = 640

    while True:
        ok, raw = cam.read()
        if not ok or raw is None:
            time.sleep(0.01)
            continue

        frame = raw.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_count += 1

        do_emotion = (EMOTION_AVAILABLE and
                      time.time() - last_emotion_time > EMOTION_INTERVAL)
        if do_emotion:
            last_emotion_time = time.time()

        # ---------- Detection ----------
        if frame_count % DETECT_EVERY == 0:
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            faces = app.get(small)

            last_results = []
            new_trackers = []

            for f in faces:
                x1, y1, x2, y2 = [int(v / SCALE) for v in f.bbox]
                w_px, h_px = x2 - x1, y2 - y1
                if w_px < MIN_FACE_SIZE or h_px < MIN_FACE_SIZE:
                    continue

                is_small = max(w_px, h_px) < SMALL_FACE_LIMIT
                enc = f.embedding

                face_crop = None
                if is_small:
                    pad = int(0.15 * max(w_px, h_px))
                    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                    cx2 = min(frame.shape[1], x2 + pad)
                    cy2 = min(frame.shape[0], y2 + pad)
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    enc2 = get_embedding_fast(face_crop)
                    if enc2 is not None:
                        enc = enc2

                # Recognition
                person_idx, best_dist, raw_label, raw_known = recognize(
                    enc, db, is_small)

                # Tracker matching
                box = (x1, y1, x2, y2)
                matched = None
                best_iou = 0.3
                for t in trackers:
                    v = iou(box, t["box"])
                    if v > best_iou:
                        best_iou = v; matched = t

                if matched is None:
                    smoother = LabelSmoother(SMOOTH_HISTORY,
                                             SMOOTH_MIN_VOTE, SMOOTH_FORGET)
                    track = {"box": box, "smoother": smoother,
                             "person_idx": -1, "person_locked": False,
                             "emotion": None, "emotion_conf": 0.0,
                             "is_new": True}
                else:
                    track = matched
                    track["box"] = box
                    track["is_new"] = False

                new_trackers.append(track)

                # Emotion
                if do_emotion:
                    if face_crop is None:
                        pad = int(0.15 * max(w_px, h_px))
                        qx1 = max(0, x1 - pad); qy1 = max(0, y1 - pad)
                        qx2 = min(frame.shape[1], x2 + pad)
                        qy2 = min(frame.shape[0], y2 + pad)
                        face_crop = frame[qy1:qy2, qx1:qx2]
                    if face_crop is not None and face_crop.size > 0:
                        emo, conf = detect_emotion(face_crop)
                        if emo:
                            track["emotion"] = emo
                            track["emotion_conf"] = conf

                # Distance
                distance_m = estimate_distance(w_px, frame_w)

                # Auto-update
                if raw_known and person_idx >= 0:
                    if not track.get("person_locked"):
                        track["person_idx"] = person_idx
                        track["person_locked"] = True

                    if track["person_idx"] == person_idx \
                            and best_dist <= UPDATE_MIN_DIST:
                        if face_crop is None:
                            pad = int(0.15 * max(w_px, h_px))
                            qx1 = max(0, x1 - pad); qy1 = max(0, y1 - pad)
                            qx2 = min(frame.shape[1], x2 + pad)
                            qy2 = min(frame.shape[0], y2 + pad)
                            face_crop = frame[qy1:qy2, qx1:qx2]
                        cur_q = face_quality_score(face_crop)

                        s = track["smoother"]
                        s.high_conf_count += 1
                        if cur_q > s.best_new_qual:
                            s.best_new_qual = cur_q
                            s.best_new_enc = enc.copy()

                        if s.high_conf_count >= UPDATE_MIN_VOTES:
                            try_auto_update(db, person_idx,
                                            s.best_new_enc, s.best_new_qual)
                            s.high_conf_count = 0
                            s.best_new_enc = None
                            s.best_new_qual = 0.0
                    else:
                        track["smoother"].high_conf_count = 0
                else:
                    track["person_locked"] = False
                    track["person_idx"] = -1

                # Smoother update
                label, state = track["smoother"].update(raw_label, raw_known)
                if state == "known":
                    color = COLOR_KNOWN; is_unknown_final = False
                elif state == "uncertain":
                    color = COLOR_UNCERTAIN; is_unknown_final = False
                else:
                    color = COLOR_UNKNOWN; is_unknown_final = True

                if state == "uncertain" and label != "Unknown":
                    label = label + "  (?)"

                # Alert for new faces only
                if track.get("is_new") and state != "uncertain":
                    photo_for_tg = frame[max(0, y1-20):min(frame_h, y2+20),
                                         max(0, x1-20):min(frame_w, x2+20)].copy()
                    if state == "unknown" and UNKNOWN_ALERT:
                        txt = (f"Unknown person detected!\n"
                               f"Time: {time.strftime('%H:%M:%S')}\n"
                               f"Distance: ~{distance_m:.1f} m")
                        if track["emotion"]:
                            txt += f"\nEmotion: {track['emotion']} " \
                                   f"({track['emotion_conf']:.0f}%)"
                        telegram_alert(f"unknown_{int(time.time())}",
                                       txt, photo_for_tg)
                    elif state == "known" and KNOWN_ALERT:
                        name = raw_label
                        txt = (f"{name} arrived\n"
                               f"Time: {time.strftime('%H:%M:%S')}\n"
                               f"Distance: ~{distance_m:.1f} m")
                        telegram_alert(f"known_{name}", txt, photo_for_tg)
                    track["is_new"] = False

                # Extra info
                parts = []
                if distance_m > 0:
                    parts.append(f"~{distance_m:.1f}m")
                if track["emotion"]:
                    parts.append(f"{track['emotion']} {track['emotion_conf']:.0f}%")
                age_est = int(f.age) if hasattr(f, "age") and f.age is not None else ""
                if age_est:
                    parts.append(f"Est.Age {age_est}")
                extra = "  |  ".join(parts)

                last_results.append(
                    (x1, y1, x2, y2, label, color, enc, extra,
                     is_unknown_final, is_small, state)
                )

            trackers = new_trackers

        # ---------- Draw ----------
        unknown_faces = []
        counter = 0
        known_count = 0
        uncertain_count = 0

        for (x1, y1, x2, y2, label, color, enc, extra,
             is_unknown, is_small, state) in last_results:
            if state == "unknown":
                counter += 1
                label = f"Unknown (press {counter})"
                unknown_faces.append((enc, (x1, y1, x2, y2)))
                draw_box(frame, y1, x2, y2, x1, color, label, extra, 1.0)
            elif state == "uncertain":
                uncertain_count += 1
                draw_box(frame, y1, x2, y2, x1, color, label, extra, 1.0)
            else:
                known_count += 1
                draw_box(frame, y1, x2, y2, x1, color, label, extra, 1.0)

        draw_header(frame, len(last_results), known_count,
                    uncertain_count, counter, src_label,
                    tg_on=(TELEGRAM_ENABLED and bool(TELEGRAM_TOKEN)))
        draw_footer(frame, counter)

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
            if win_w > 0 and win_h > 0:
                fh, fw = frame.shape[:2]
                if (win_w, win_h) != (fw, fh):
                    frame = cv2.resize(frame, (win_w, win_h),
                                       interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('s') and unknown_faces:
            for i, (enc, _) in enumerate(unknown_faces, 1):
                n, a = ask_details(face_no=i)
                if n:
                    add_new_person(db, n, a, enc)
            save_db(db)
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(unknown_faces):
                enc, box = unknown_faces[idx]
                n, a = ask_details(face_no=idx + 1)
                if n:
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2]
                    q = face_quality_score(crop) if crop.size > 0 else 0.5
                    add_new_person(db, n, a, enc, quality=q)
                    save_db(db)

    cam.release()
    cv2.destroyAllWindows()


def add_new_person(db, name, age, encoding, quality=0.7):
    fname = make_enc_filename(name, 0)
    db.append({
        "name": name, "age": age,
        "templates": [{
            "encoding": encoding.copy(), "file": fname,
            "quality": float(quality),
        }],
    })
    print(f"[ADDED] {name} ({age})")


if __name__ == "__main__":
    main()