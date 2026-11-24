import cv2
import socket
import struct
import json
import os
import numpy as np
import time
import datetime
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9999"))
MODEL_PPE_PATH = os.getenv("MODEL_PPE_PATH", "best.pt")
MODEL_PERSON_PATH = os.getenv("MODEL_PERSON_PATH", "yolov8n.pt")
FIREBASE_CONFIG_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-adminsdk.json")

firebase_initialized = False
db = None

if os.path.exists(FIREBASE_CONFIG_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_CONFIG_PATH)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_initialized = True
        print("[SUCCESS] Mobil Sunucusu Firebase Firestore bağlantısı başarıyla sağlandı.")
    except Exception as e:
        print(f"[WARNING] Firebase başlatılamadı (Sistem lokal modda çalışmaya devam edecek): {e}")
else:
    print(f"[WARNING] '{FIREBASE_CONFIG_PATH}' bulunamadı. Sistem lokal modda çalışmaya devam edecek.")

print("[INFO] Mobil yapay zeka modelleri yükleniyor...")
model_ppe = YOLO(MODEL_PPE_PATH)
model_person = YOLO(MODEL_PERSON_PATH)

# ==========================================
# 2. CRITICAL CONFIGURATION: THRESHOLDS & COOLDOWN
# ==========================================
# GÜNCELLENDİ: Modelin eğitime katılırken aldığı gerçek etiket isimleri ve güven eşikleri
# Rule flag sent by the client -> exact class names of the model that satisfy it
RULE_CLASSES = {
    "check_mask": {"mask"},
    "check_helmet": {"helmet"},
    "check_vest": {"vest"},
    "check_glasses": {"glasses", "goggles"},
    "check_belt": {"belt"},
    "check_suit": {"protective_suit"},
    "check_gloves": {"safety_gloves"},
    "check_toolbox": {"toolbox"},
    "check_welding": {"welding_helmet"},
    "check_ear": {"ear_protection"},
}

# (rule flag, accepted class names, banner text, Firestore record text)
ALERTS = [
    ("check_mask", RULE_CLASSES["check_mask"], "UYARI: Maske Yok!", "Maske Yok!"),
    ("check_helmet", RULE_CLASSES["check_helmet"], "UYARI: Baret Yok!", "Baret Yok!"),
    ("check_vest", RULE_CLASSES["check_vest"], "UYARI: Yelek Yok!", "Yelek Yok!"),
    ("check_glasses", RULE_CLASSES["check_glasses"], "UYARI: Gozluk Yok!", "Gözlük Yok!"),
    ("check_belt", RULE_CLASSES["check_belt"], "UYARI: Kemer Yok!", "Kemer Yok!"),
    ("check_suit", RULE_CLASSES["check_suit"], "UYARI: Koruyucu Elbise Yok!", "Koruyucu Elbise Yok!"),
    ("check_gloves", RULE_CLASSES["check_gloves"], "UYARI: Eldiven Yok!", "Eldiven Yok!"),
    ("check_toolbox", RULE_CLASSES["check_toolbox"], "UYARI: Alet Cantasi Yok!", "Alet Çantası Yok!"),
    ("check_welding", RULE_CLASSES["check_welding"], "UYARI: Kaynak Maskesi Yok!", "Kaynak Maskesi Yok!"),
    ("check_ear", RULE_CLASSES["check_ear"], "UYARI: Kulak Koruyucu Yok!", "Kulak Koruyucu Yok!"),
]

THRESHOLDS = {
    "mask": 0.60,
    "helmet": 0.15,
    "vest": 0.30,
    "glasses": 0.30,
    "goggles": 0.30,
    "belt": 0.25,
    "protective_suit": 0.35,   # yellow chemical protective suit -> protective_suit
    "safety_gloves": 0.35,     # safety gloves -> safety_gloves
    "toolbox": 0.30,           # toolbox
    "welding_helmet": 0.40,    # welding helmet -> welding_helmet
    "ear_protection": 0.35     # ear protection -> ear_protection
}

# Mobil bildirimlerin veri akışını şişirmemesi için zaman takibi mekanizması
last_push_times = {}
PUSH_COOLDOWN = 10  # Aynı ihlal türü maksimum 10 saniyede bir Firebase'e raporlanır

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def send_to_firebase(ihlal_turu):
    """Belirlenen kurumsal ihlalleri asenkron takiple bulut veritabanına kaydeder."""
    if not firebase_initialized or db is None:
        return
    now = time.time()
    if now - last_push_times.get(ihlal_turu, 0) > PUSH_COOLDOWN:
        try:
            db.collection("Ihlaller").add({
                "tur": ihlal_turu,
                "tarih": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "durum": "Kritik"
            })
            last_push_times[ihlal_turu] = now
            print(f"[FIREBASE LOG] Mobil İhlal Bildirimi: {ihlal_turu}")
        except Exception as e:
            print(f"[ERROR] Firebase Firestore servis hatası: {e}")

def draw_alert(frame, text, y):
    """Canlı video akışı üzerine görsel uyarı katmanı çizer."""
    cv2.rectangle(frame, (5, y - 25), (400, y + 5), (0, 0, 255), -1)
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def recv_exactly(sock, n):
    """Soket tamponundan tam olarak hedeflenen bayt boyutu gelene kadar bekler."""
    data = b''
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data += packet
        except socket.error:
            return None
    return data

# ==========================================
# 4. SERVER SOCKET SETUP
# ==========================================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"[INFO] Python Mobil Socket Sunucusu başlatıldı. Port: {PORT}")

cap = cv2.VideoCapture(0)

# ==========================================
# 5. MAIN PROCESSING LOOP
# ==========================================
while True:
    print("[INFO] Android mobil uygulamasının (Java İstemcisi) bağlanması bekleniyor...")
    conn, addr = server_socket.accept()
    print(f"[CONNECTED] Android Cihaz Segmenti Bağlandı: {addr}")

    try:
        while True:
            # 5.1. Android Mobil İstemciden Kuralların Alınması
            raw_msg_len = recv_exactly(conn, 4)
            if not raw_msg_len:
                print("[INFO] İletişim Android tarafınca sonlandırıldı.")
                break

            msg_len = struct.unpack('>I', raw_msg_len)[0]
            data = recv_exactly(conn, msg_len)
            if not data: break

            rules = json.loads(data.decode('utf-8'))

            # 5.2. Kamera Çerçeve Yakalama ve Matris Optimizasyonu
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Video yakalama aygıtı (Kamera) hatası!")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            results_ppe = model_ppe.predict(frame_resized, conf=0.15, verbose=False)
            annotated_frame = frame_resized.copy()

            detected_classes = []

            # 5.3. Bounding Box Analizi ve Dinamik Eşleşme Kontrolleri
            if len(results_ppe[0].boxes) > 0:
                for box in results_ppe[0].boxes:
                    cls_id = int(box.cls[0])
                    raw_cls_name = model_ppe.names[cls_id]
                    cls_name = raw_cls_name.lower()
                    confidence = float(box.conf[0])

                    required_conf = THRESHOLDS.get(cls_name, 0.30)
                    if confidence < required_conf:
                        continue

                    detected_classes.append(cls_name)

                    # Exact class-name matching (the version used in the paper's laboratory tests
                    # matched substrings, so "welding_helmet" also satisfied the helmet rule).
                    should_draw = any(rules.get(rule) and cls_name in names
                                      for rule, names in RULE_CLASSES.items())

                    if should_draw:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{raw_cls_name} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5.4. Koşullu İhlal Değerlendirmesi ve Bulut (Firebase) Senkronizasyonu
            y_offset = 40
            for rule, names, alert_text, record_text in ALERTS:
                if rules.get(rule) and not any(c in names for c in detected_classes):
                    draw_alert(annotated_frame, alert_text, y_offset)
                    send_to_firebase(record_text)
                    y_offset += 40

            # 5.5. Görüntünün Mobil Ağ Uyumluluğu İçin Sıkıştırılması ve Akış Transferi
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, img_encoded = cv2.imencode('.jpg', annotated_frame, encode_param)

            data_to_send = img_encoded.tobytes()
            size = len(data_to_send)

            conn.sendall(struct.pack(">I", size) + data_to_send)
            time.sleep(0.06)

    except Exception as e:
        print(f"[ERROR] Mobil veri transfer döngüsünde kritik hata: {e}")

    finally:
        conn.close()
        print("[INFO] Mobil bağlantı havuzu temizlendi. Yeni soket oturumu bekleniyor...\n")