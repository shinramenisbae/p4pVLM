import base64
import io
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emotion_detector import EmotionDetector

# Load environment variables from a .env file, if present (search upwards)
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# --- Passive biosignal imports ---
import os
import sys
import torch
import requests
from datetime import datetime, timedelta
from scipy.signal import resample

# Ensure we can import the passive model
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PASSIVE_NET_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "passive", "model", "network")
)
if _PASSIVE_NET_DIR not in sys.path:
    sys.path.append(_PASSIVE_NET_DIR)
try:
    from CNN import EmotionCNN  # type: ignore
except Exception as e:  # pragma: no cover
    EmotionCNN = None  # fallback


class ImagePayload(BaseModel):
    image_base64: str


# Fusion input models
class VisualInput(BaseModel):
    valence: float
    arousal: float
    confidence: float


class PassiveInput(BaseModel):
    valence: float
    arousal: float


class FusionPayload(BaseModel):
    visual: Optional[VisualInput] = None
    passive: Optional[PassiveInput] = None
    videoId: Optional[str] = None
    videoTimeSec: Optional[int] = None


app = FastAPI(title="Active Emotion API", version="0.1.0")

# Allow local webapp (Vite dev) and any localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://127.0.0.1",
    ],
    # Allow any localhost/127.0.0.1 with any port, and common LAN dev hosts
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a single detector instance for reuse
detector = EmotionDetector()

# --- Passive model and research API setup ---
PASSIVE_BASE_URL = os.getenv("PASSIVE_BASE_URL", "http://130.216.217.53:8000")
PASSIVE_EMAIL = os.getenv("PASSIVE_EMAIL")
PASSIVE_PASSWORD = os.getenv("PASSIVE_PASSWORD")
PASSIVE_USER_ID = os.getenv("PASSIVE_USER_ID")
# Optional simulation start (ISO format). If not provided, start from now-12s.
PASSIVE_SIMULATE_FROM = os.getenv("PASSIVE_SIMULATE_FROM") or os.getenv("passive_simulate_from")
PASSIVE_DEBUG = str(os.getenv("PASSIVE_DEBUG", "")).lower() in ("1", "true", "yes", "y")

if PASSIVE_DEBUG:
    try:
        print(
            f"[passive] Debug: BASE_URL={PASSIVE_BASE_URL} email_set={bool(PASSIVE_EMAIL)} user_id_set={bool(PASSIVE_USER_ID)} simulate_from={PASSIVE_SIMULATE_FROM}"
        )
    except Exception:
        pass

# Global state for research API session and last timestamp window
_passive_token: Optional[str] = None
_passive_last_ts: Optional[datetime] = None

# Load EmotionCNN weights
_passive_model: Optional[EmotionCNN] = None
_passive_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if EmotionCNN is not None:
        _passive_model = EmotionCNN()
        print("test1")
        weights_path = os.path.join(_PASSIVE_NET_DIR, "emotion_cnn.pth")
        print("test2")
        print(f"[passive] Loading model from {weights_path}")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=_passive_model_device)
            _passive_model.load_state_dict(state)
            _passive_model.eval()
            _passive_model.to(_passive_model_device)
        else:
            print(f"[passive] Warning: weights not found at {weights_path}")
    else:
        print(
            "[passive] Warning: EmotionCNN import failed; passive endpoint will simulate."
        )
except Exception as e:  # pragma: no cover
    print(f"[passive] Error loading model: {e}")
    _passive_model = None
############################
# Active image save config #
############################
# Save incoming webcam frames under active/output/<Pid>
_SAVE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "output"))



def _fmt_time(dt: datetime) -> str:
    # Match the format used in Run.py (12-hour clock)
    return dt.strftime("%Y-%m-%dT%I:%M:%S")


def _passive_login() -> Optional[str]:
    global _passive_token
    if not PASSIVE_EMAIL or not PASSIVE_PASSWORD:
        return None
    try:
        resp = requests.post(
            f"{PASSIVE_BASE_URL}/login-researcher",
            data={"username": PASSIVE_EMAIL, "password": PASSIVE_PASSWORD},
            timeout=10,
        )
        if resp.status_code == 200:
            _passive_token = resp.json().get("access_token")
            return _passive_token
        else:
            return None
    except Exception:
        return None


def _clean_ppg_values(rows):
    cleaned = []
    for row in rows or []:
        raw_value = row.get("ppg_gr")
        if raw_value is None:
            continue
        values = []
        if isinstance(raw_value, list):
            values = raw_value
        elif isinstance(raw_value, str):
            try:
                parsed = eval(
                    raw_value, {"__builtins__": None}, {}
                )  # safe enough for numbers/lists
                if isinstance(parsed, list):
                    values = parsed
            except Exception:
                pass
        filtered = [v for v in values if isinstance(v, (int, float)) and v >= 0]
        cleaned.extend(filtered)
    return cleaned


def _predict_valence_arousal_from_ppg(ppg_list):
    """Run the passive CNN over the 12s chunk; return (valence_class, arousal_class)."""
    if not _passive_model or not ppg_list:
        # Simulated neutral
        return 1, 1  # treat as positive/active for demo
    # Upsample to 64 Hz as in Run.py
    x_64hz = resample(
        np.array(ppg_list, dtype=np.float32), int(len(ppg_list) * (64 / 25))
    )
    # Normalise to [0, 1000]
    personal_min = float(np.min(x_64hz)) if len(x_64hz) > 0 else 0.0
    personal_max = float(np.max(x_64hz)) if len(x_64hz) > 0 else 1.0
    den = (personal_max - personal_min) or 1.0
    x_norm = (x_64hz - personal_min) / den * 1000.0
    # Segment into pulses of 140
    pulse_len = 140
    segments = []
    for start in range(0, max(0, len(x_norm) - pulse_len + 1), pulse_len):
        segments.append(x_norm[start : start + pulse_len])
    if not segments:
        return 1, 1
    tensor = torch.tensor(
        np.array(segments), dtype=torch.float32, device=_passive_model_device
    )
    with torch.no_grad():
        valence_logits, arousal_logits = _passive_model(tensor)
        valence_pred = torch.argmax(valence_logits, dim=1).cpu().numpy()
        arousal_pred = torch.argmax(arousal_logits, dim=1).cpu().numpy()
    # Aggregate by majority vote
    from collections import Counter

    v = Counter(valence_pred).most_common(1)[0][0]
    a = Counter(arousal_pred).most_common(1)[0][0]
    return int(v), int(a)


@app.post("/predict")
def predict(
    payload: ImagePayload | None = None,
    image_file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    pid: str | None = Form(default=None),
    video_id: str | None = Form(default=None),
    video_time_sec: int | None = Form(default=None),
):
    try:
        # Accept multipart upload or JSON/base64
        img_bytes = None
        if image_file is not None:
            img_bytes = image_file.file.read()
        else:
            data = image_base64 or (payload.image_base64 if payload else None)
            if not data:
                raise HTTPException(status_code=400, detail="Missing image payload")
            if "," in data:
                data = data.split(",", 1)[1]
            img_bytes = base64.b64decode(data)
        # Prepare save path under output/<Pid> for annotated image
        saved_path = None
        fpath = None
        try:
            safe_pid = str(pid or "unknown").strip()
            safe_pid = "".join(ch for ch in safe_pid if ch.isalnum() or ch in ("-", "_")) or "unknown"
            out_dir = os.path.join(_SAVE_ROOT, safe_pid)
            os.makedirs(out_dir, exist_ok=True)
            # Build filename: "VideoID-VideoTime.jpg"
            raw_label = str(video_id or "video")
            base_label = os.path.basename(raw_label)
            base_label, _ = os.path.splitext(base_label)
            safe_label = "".join(ch for ch in base_label if ch.isalnum() or ch in ("-", "_")) or "video"
            try:
                t_int = int(video_time_sec) if video_time_sec is not None else 0
            except Exception:
                t_int = 0
            fname = f"{safe_label}-{t_int}.jpg"
            fpath = os.path.join(out_dir, fname)
        except Exception as e:
            try:
                print(f"[active] Error preparing save path: {e}")
            except Exception:
                pass
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        processed_img, results = detector.detect_emotions(img)

        # Save the detector's annotated frame directly (consistent with emotion_detector.py)
        try:
            if fpath:
                ok = cv2.imwrite(fpath, processed_img)
                if ok and os.path.exists(fpath):
                    saved_path = fpath
                else:
                    print(f"[active] Warning: cv2.imwrite failed or file missing at {fpath}")
        except Exception as e:
            print(f"[active] Error saving annotated frame: {e}")

        # Select the most confident face (if any) as primary
        primary = None
        if results:
            primary = max(results, key=lambda r: r.get("confidence", 0.0))

        # Convert any numpy types in results/primary to native Python for JSON serialization
        def to_python(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_python(v) for v in obj]
            return obj

        sanitized_results = to_python(results)
        sanitized_primary = to_python(primary) if primary is not None else None

        return {
            "success": True,
            "faces": sanitized_results,
            "primary": sanitized_primary,
            "savedPath": saved_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/passive/predict")
def passive_predict():
    """Fetch 12s biosignal window from research API, run CNN, return valence/arousal.
    Falls back to simulation if credentials/model unavailable.
    If PASSIVE_SIMULATE_FROM is set, the response will mark simulated=True (for clarity)
    even if real data was fetched, while still using real data where available.
    """
    global _passive_token, _passive_last_ts
    try:
        # Init last ts
        if _passive_last_ts is None:
            if PASSIVE_SIMULATE_FROM:
                simulated = True
                try:
                    _passive_last_ts = datetime.fromisoformat(PASSIVE_SIMULATE_FROM)
                except Exception:
                    _passive_last_ts = datetime.now() - timedelta(seconds=12)
            else:
                _passive_last_ts = datetime.now() - timedelta(seconds=12)

        # Try login if needed
        if _passive_token is None and PASSIVE_EMAIL and PASSIVE_PASSWORD:
            _passive_token = _passive_login()

        # If no token or user id, simulate
        simulated = False
        headers = {}
        try:
            if _passive_token and PASSIVE_USER_ID:
                headers = {"Authorization": f"Bearer {_passive_token}"}
                start_dt = _passive_last_ts
                end_dt = _passive_last_ts + timedelta(seconds=12)
                params = {
                    "columns": ["timestamp", "ppg_gr"],
                    "user_id": PASSIVE_USER_ID,
                    "start_date": _fmt_time(start_dt),
                    "end_date": _fmt_time(end_dt),
                }
                resp = requests.get(
                    f"{PASSIVE_BASE_URL}/research/sensor-data",
                    headers=headers,
                    params=params,
                    timeout=10,
                )
                if resp.status_code == 200:
                    rows = resp.json()
                    ppg = _clean_ppg_values(rows)
                    v_class, a_class = _predict_valence_arousal_from_ppg(ppg)
                else:
                    if PASSIVE_DEBUG:
                        print(f"[passive] Debug: research API non-200: {resp.status_code}")
                    simulated = True
                    v_class, a_class = 1, 1
            else:
                if PASSIVE_DEBUG:
                    print(
                        f"[passive] Debug: simulating due to missing token or user id (token_set={bool(_passive_token)}, user_id_set={bool(PASSIVE_USER_ID)})"
                    )
                simulated = True
                v_class, a_class = 1, 1
        except Exception:
            # Treat network/HTTP errors as simulated fallback
            if PASSIVE_DEBUG:
                print("[passive] Debug: exception when fetching research data; simulating")
            simulated = True
            v_class, a_class = 1, 1

        # Advance window regardless to keep sync with 12s cadence
        _passive_last_ts = _passive_last_ts + timedelta(seconds=12)

        # Map classes to [-1, +1] for frontend friendliness
        def to_score(c: int) -> float:
            return 1.0 if int(c) == 1 else -1.0

        # If a simulate-from start time is configured, force the response flag to simulated
        simulated_flag = True if PASSIVE_SIMULATE_FROM else simulated
        return {
            "success": True,
            "simulated": simulated_flag,
            "valence_class": int(v_class),
            "arousal_class": int(a_class),
            "valence": to_score(v_class),
            "arousal": to_score(a_class),
            "window": {
                "start": _fmt_time(_passive_last_ts - timedelta(seconds=12)),
                "end": _fmt_time(_passive_last_ts),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Fusion endpoint ---


def _map_va_to_emotion(valence: float, arousal: float) -> str:
    # Simple mapping aligned with LateFusionModule
    v_bin = 1 if valence > 0 else 0
    a_bin = 1 if arousal > 0 else 0
    if abs(valence) < 0.2 and abs(arousal) < 0.2:
        return "neutral"
    if a_bin == 1 and abs(arousal) > 0.7:
        if abs(valence) > 0.5:
            return "happy" if v_bin == 1 else "angry"
        else:
            return "surprise" if arousal > 0 else "fear"
    if a_bin == 0 and abs(arousal) < 0.3:
        if abs(valence) > 0.3:
            return "content" if v_bin == 1 else "sad"
        else:
            return "calm"
    mapping = {
        (0, 0): "sad",
        (0, 1): "angry",
        (1, 0): "calm",
        (1, 1): "happy",
    }
    return mapping.get((v_bin, a_bin), "neutral")


@app.post("/fusion/predict")
def fusion_predict(payload: FusionPayload):
    try:
        visual = payload.visual
        passive = payload.passive
        if visual is None and passive is None:
            raise HTTPException(
                status_code=400, detail="Missing visual and passive inputs"
            )

        # Missing modality handling
        if passive is None and visual is not None:
            return {
                "success": True,
                "strategy": "visual_only",
                "valence": float(visual.valence),
                "arousal": float(visual.arousal),
                "discrete_emotion": _map_va_to_emotion(visual.valence, visual.arousal),
                "fusion_confidence": float(max(0.0, min(1.0, visual.confidence))),
                "contributing_modalities": ["visual"],
            }
        if visual is None and passive is not None:
            return {
                "success": True,
                "strategy": "biosignal_only",
                "valence": float(passive.valence),
                "arousal": float(passive.arousal),
                "discrete_emotion": _map_va_to_emotion(
                    passive.valence, passive.arousal
                ),
                "fusion_confidence": 0.5,
                "contributing_modalities": ["biosignal"],
            }

        # Both available → rule-based with visual confidence modulation
        vis_conf = max(0.0, min(1.0, visual.confidence))
        # Default visual weight 0.6, lowered proportionally by low confidence
        # Simple mapping: w_vis = 0.6 * vis_conf (0..0.6), w_bio = 1 - w_vis (0.4..1)
        w_vis = 0.6 * vis_conf
        w_bio = 1.0 - w_vis

        # Disagreement checks
        val_disagree = abs((passive.valence or 0.0) - (visual.valence or 0.0))
        aro_disagree = abs((passive.arousal or 0.0) - (visual.arousal or 0.0))

        # If visual is fairly confident and disagreement is strong, split: visual for valence, biosignal for arousal
        if vis_conf > 0.7 and (val_disagree > 1.0 or aro_disagree > 1.0):
            fused_v = float(visual.valence)
            fused_a = float(passive.arousal)
            strategy = "rule_based_split_decision"
            fusion_conf = vis_conf
        else:
            # Weighted average
            fused_v = w_bio * float(passive.valence) + w_vis * float(visual.valence)
            fused_a = w_bio * float(passive.arousal) + w_vis * float(visual.arousal)
            strategy = "rule_based_weighted"
            fusion_conf = vis_conf  # no passive confidence provided

        return {
            "success": True,
            "strategy": strategy,
            "valence": fused_v,
            "arousal": fused_a,
            "discrete_emotion": _map_va_to_emotion(fused_v, fused_a),
            "fusion_confidence": float(fusion_conf),
            "contributing_modalities": ["biosignal", "visual"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)
