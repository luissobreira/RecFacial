import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
import tempfile
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# Cache global do dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

_mtcnn = None
_resnet = None

def get_models():
    """Retorna instâncias únicas de MTCNN e InceptionResnetV1 em cache."""
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=30,
            thresholds=[0.7, 0.8, 0.85],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=device
        )
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return _mtcnn, _resnet

def get_image_blur_score(crop_pil):
    """Calcula a pontuação de nitidez/desfoque da imagem usando a variância do Laplaciano."""
    arr = np.array(crop_pil.convert('L'))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())

def draw_futuristic_hud(img_pil, box, lmk, label="SUSPECT", is_match=True):
    """Desenha traços biométricos futuristas, malha de triangulação e HUD cibernético no rosto."""
    img = np.array(img_pil.convert('RGB'))
    h, w, c = img.shape
    
    clean_label = label.replace("Í", "I").replace("ú", "u").replace("Ú", "U")
    
    if is_match:
        primary_color = (0, 255, 102)     # #00FF66 Neon Green
        secondary_color = (255, 240, 0)   # #00F0FF Cyber Cyan
    else:
        primary_color = (80, 60, 255)     # Crimson Red
        secondary_color = (0, 200, 255)   # Amber / Cyan
        
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    
    corner_len = max(12, int(min(bw, bh) * 0.22))
    thick = 2
    
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), primary_color, thick)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), primary_color, thick)
    
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), primary_color, thick)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), primary_color, thick)
    
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), primary_color, thick)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), primary_color, thick)
    
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), primary_color, thick)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), primary_color, thick)
    
    left_eye, right_eye, nose, left_mouth, right_mouth = lmk.astype(int)
    
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    chin = (nose[0], min(h, y2 - int(bh * 0.05)))
    forehead = (eye_center[0], max(0, y1 + int(bh * 0.12)))
    left_cheek = (max(0, x1 + int(bw * 0.12)), nose[1])
    right_cheek = (min(w, x2 - int(bw * 0.12)), nose[1])
    
    nodes = [left_eye, right_eye, nose, left_mouth, right_mouth, forehead, chin, left_cheek, right_cheek]
    
    overlay = img.copy()
    mesh_col = secondary_color
    
    cv2.line(overlay, tuple(left_eye), tuple(right_eye), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(left_eye), tuple(nose), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(right_eye), tuple(nose), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(nose), tuple(left_mouth), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(nose), tuple(right_mouth), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(left_mouth), tuple(right_mouth), mesh_col, 1, cv2.LINE_AA)
    
    cv2.line(overlay, tuple(forehead), tuple(left_eye), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(forehead), tuple(right_eye), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(left_eye), tuple(left_cheek), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(right_eye), tuple(right_cheek), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(left_cheek), tuple(left_mouth), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(right_cheek), tuple(right_mouth), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(left_mouth), tuple(chin), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(right_mouth), tuple(chin), mesh_col, 1, cv2.LINE_AA)
    cv2.line(overlay, tuple(nose), tuple(chin), mesh_col, 1, cv2.LINE_AA)
    
    img = cv2.addWeighted(overlay, 0.70, img, 0.30, 0)
    
    for p in nodes:
        pt = tuple(p)
        cv2.circle(img, pt, 3, primary_color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)
        
    cv2.circle(img, tuple(left_eye), 10, primary_color, 1, cv2.LINE_AA)
    cv2.circle(img, tuple(right_eye), 10, primary_color, 1, cv2.LINE_AA)
    
    cv2.putText(img, f"ID :: {clean_label}", (x1, max(20, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, primary_color, 1, cv2.LINE_AA)
                
    return Image.fromarray(img)

def get_enhanced_face_embedding(crop_pil):
    """
    Calcula embedding de alta precisão com amostragem multi-crop e espelhamento (Flip Invariance).
    Minimiza erros causados por variações de pose, iluminação e ângulo.
    """
    _, resnet = get_models()
    
    # 1. Imagem Direta
    img_orig = crop_pil.resize((160, 160))
    arr_orig = np.array(img_orig).astype(np.float32) / 255.0
    arr_orig = (arr_orig - 0.5) / 0.5
    tensor_orig = torch.tensor(arr_orig).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Imagem Espelhada Horizontais (Horizontal Flip para pose invariance)
    img_flip = ImageOps.mirror(img_orig)
    arr_flip = np.array(img_flip).astype(np.float32) / 255.0
    arr_flip = (arr_flip - 0.5) / 0.5
    tensor_flip = torch.tensor(arr_flip).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb_orig = resnet(tensor_orig).squeeze().cpu().numpy()
        emb_flip = resnet(tensor_flip).squeeze().cpu().numpy()
        
    # Média dos vetores original e espelhado
    combined_emb = (emb_orig + emb_flip) / 2.0
    
    norm = np.linalg.norm(combined_emb)
    if norm > 0:
        combined_emb = combined_emb / norm
    return combined_emb, tensor_orig.squeeze(0)

def process_face_image(img_pil, label="SUBJECT", is_match=True):
    """Detecta rosto, extrai landmarks e alinha uma imagem estática."""
    mtcnn, _ = get_models()
    
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        
    boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
    
    if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.75:
        return None, None, None, None, None, "Nenhum rosto detectado com confiança suficiente."
        
    box = boxes[0].astype(int)
    lmk = landmarks[0]
    
    width, height = img_pil.size
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
    
    w_box, h_box = x2 - x1, y2 - y1
    pad_x, pad_y = int(w_box * 0.25), int(h_box * 0.25)
    crop_box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))
    
    crop_pil = img_pil.crop(crop_box).resize((160, 160))
    
    crop_base = img_pil.crop(crop_box).resize((350, 350))
    scale_x = 350.0 / (crop_box[2] - crop_box[0])
    scale_y = 350.0 / (crop_box[3] - crop_box[1])
    
    box_rel = np.array([
        (box[0] - crop_box[0]) * scale_x,
        (box[1] - crop_box[1]) * scale_y,
        (box[2] - crop_box[0]) * scale_x,
        (box[3] - crop_box[1]) * scale_y
    ])
    lmk_rel = np.zeros_like(lmk)
    for i in range(len(lmk)):
        lmk_rel[i][0] = (lmk[i][0] - crop_box[0]) * scale_x
        lmk_rel[i][1] = (lmk[i][1] - crop_box[1]) * scale_y
        
    face_crop_hud = draw_futuristic_hud(crop_base, box_rel, lmk_rel, label=label, is_match=is_match)
    landmarks_img = draw_futuristic_hud(img_pil.copy(), box, lmk, label=label, is_match=is_match)
    
    # Gerar embedding aprimorado com multi-crop flip-invariance
    emb, tensor = get_enhanced_face_embedding(crop_pil)
    
    return tensor, face_crop_hud, landmarks_img, box, lmk, None, emb

def compare_embeddings(emb1, emb2, threshold_mode='padrao'):
    """Compara dois vetores de embedding faciais com modelo estatístico calibrado."""
    cosine_sim = float(np.dot(emb1, emb2))
    euclidean_dist = float(np.linalg.norm(emb1 - emb2))
    
    def calculate_certainty(sim):
        if sim >= 0.75:
            pct = 95.0 + (sim - 0.75) * 20.0
        elif sim >= 0.60:
            pct = 75.0 + (sim - 0.60) * 133.33
        elif sim >= 0.45:
            pct = 50.0 + (sim - 0.45) * 166.67
        elif sim >= 0.30:
            pct = 20.0 + (sim - 0.30) * 200.0
        else:
            pct = max(0.0, sim * 66.67)
        return float(np.clip(pct, 0.0, 99.9))

    certainty_pct = calculate_certainty(cosine_sim)
    
    thresholds = {
        'rigoroso': {'match_sim': 0.68, 'inconclusive_sim': 0.55},
        'padrao':   {'match_sim': 0.60, 'inconclusive_sim': 0.48},
        'permissivo': {'match_sim': 0.52, 'inconclusive_sim': 0.42}
    }
    
    t = thresholds.get(threshold_mode, thresholds['padrao'])
    
    if cosine_sim >= t['match_sim']:
        classification = "COMPATÍVEL (MESMA PESSOA)"
        status_code = "MATCH"
        color = "#28a745"
        reason = f"Similaridade ({cosine_sim:.3f}) excede o limiar de convergência pericial ({t['match_sim']:.2f})."
    elif cosine_sim >= t['inconclusive_sim']:
        classification = "INCONCLUSIVO (REQUER ANÁLISE COMPLEMENTAR)"
        status_code = "INCONCLUSIVE"
        color = "#ffc107"
        reason = f"Similaridade ({cosine_sim:.3f}) em zona intermediária entre {t['inconclusive_sim']:.2f} e {t['match_sim']:.2f}."
    else:
        classification = "INCOMPATÍVEL (PESSOAS DIFERENTES)"
        status_code = "NO_MATCH"
        color = "#dc3545"
        reason = f"Similaridade ({cosine_sim:.3f}) abaixo do limiar mínimo ({t['inconclusive_sim']:.2f})."

    return {
        'cosine_sim': cosine_sim,
        'euclidean_dist': euclidean_dist,
        'certainty_pct': certainty_pct,
        'classification': classification,
        'status_code': status_code,
        'color': color,
        'reason': reason
    }

def process_video_file(video_bytes_or_path, emb_ref, sample_fps_step=6, threshold_mode='padrao', min_blur_score=35.0, progress_callback=None):
    """Processa arquivo de vídeo com extração de alta precisão."""
    mtcnn, _ = get_models()
    
    if isinstance(video_bytes_or_path, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(video_bytes_or_path)
        tmp.close()
        video_path = tmp.name
    else:
        video_path = video_bytes_or_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Não foi possível abrir o arquivo de vídeo."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 300

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_count = 0
    raw_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if progress_callback and (frame_count % 3 == 0 or frame_count == total_frames):
            pct = min(1.0, float(frame_count) / float(total_frames))
            progress_callback(pct)
            
        if frame_count % sample_fps_step != 0:
            continue
            
        timestamp_sec = frame_count / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
        if boxes is None or len(boxes) == 0:
            continue
            
        for i, box in enumerate(boxes):
            if probs[i] is None or probs[i] < 0.85:
                continue
                
            box_int = box.astype(int)
            lmk_int = landmarks[i]
            
            left_eye, right_eye = lmk_int[0], lmk_int[1]
            interpupillary_dist = np.linalg.norm(left_eye - right_eye)
            if interpupillary_dist < 14:
                continue
                
            w_img, h_img = img_pil.size
            x1, y1, x2, y2 = max(0, box_int[0]), max(0, box_int[1]), min(w_img, box_int[2]), min(h_img, box_int[3])
            bw, bh = x2 - x1, y2 - y1
            if bw < 25 or bh < 25:
                continue
                
            pad_x, pad_y = int(bw * 0.25), int(bh * 0.25)
            crop_box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w_img, x2 + pad_x), min(h_img, y2 + pad_y))
            crop_pil = img_pil.crop(crop_box)
            
            blur_score = get_image_blur_score(crop_pil)
            if blur_score < min_blur_score:
                continue
                
            emb, _ = get_enhanced_face_embedding(crop_pil)
            metrics = compare_embeddings(emb_ref, emb, threshold_mode=threshold_mode)
            
            raw_detections.append({
                'timestamp_sec': timestamp_sec,
                'frame_img': img_pil,
                'crop_box': crop_box,
                'box': box_int,
                'lmk': lmk_int,
                'prob': probs[i],
                'blur_score': blur_score,
                'embedding': emb,
                'metrics': metrics
            })

    cap.release()
    if isinstance(video_bytes_or_path, (bytes, bytearray)) and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

    if progress_callback:
        progress_callback(1.0)

    if len(raw_detections) == 0:
        return [], "Nenhum rosto com qualidade suficiente (nítido e visível) foi detectado no vídeo."

    clusters = []
    for det in raw_detections:
        matched_cluster = None
        for cl in clusters:
            sim = np.dot(det['embedding'], cl['representative_emb'])
            if sim >= 0.70:
                matched_cluster = cl
                break
                
        if matched_cluster is not None:
            matched_cluster['detections'].append(det)
            if (det['metrics']['cosine_sim'] + det['blur_score']/1000.0) > (matched_cluster['best_det']['metrics']['cosine_sim'] + matched_cluster['best_det']['blur_score']/1000.0):
                matched_cluster['best_det'] = det
                matched_cluster['representative_emb'] = det['embedding']
        else:
            clusters.append({
                'representative_emb': det['embedding'],
                'best_det': det,
                'detections': [det]
            })

    unique_persons = []
    for idx, cl in enumerate(clusters, 1):
        best = cl['best_det']
        is_match = (best['metrics']['status_code'] == 'MATCH')
        
        img_pil = best['frame_img']
        crop_box = best['crop_box']
        box = best['box']
        lmk = best['lmk']
        
        crop_base = img_pil.crop(crop_box).resize((350, 350))
        scale_x = 350.0 / (crop_box[2] - crop_box[0])
        scale_y = 350.0 / (crop_box[3] - crop_box[1])
        
        box_rel = np.array([
            (box[0] - crop_box[0]) * scale_x,
            (box[1] - crop_box[1]) * scale_y,
            (box[2] - crop_box[0]) * scale_x,
            (box[3] - crop_box[1]) * scale_y
        ])
        lmk_rel = np.zeros_like(lmk)
        for i in range(len(lmk)):
            lmk_rel[i][0] = (lmk[i][0] - crop_box[0]) * scale_x
            lmk_rel[i][1] = (lmk[i][1] - crop_box[1]) * scale_y
            
        crop_hud = draw_futuristic_hud(crop_base, box_rel, lmk_rel, label=f"INDIVIDUO_{idx}", is_match=is_match)
        
        ts_sec = best['timestamp_sec']
        mins = int(ts_sec // 60)
        secs = int(ts_sec % 60)
        time_str = f"{mins:02d}:{secs:02d}s"
        
        unique_persons.append({
            'person_id': f"Indivíduo #{idx}",
            'label': f"ROSTO #{idx}",
            'crop_hud': crop_hud,
            'timestamp_str': time_str,
            'blur_score': best['blur_score'],
            'detections_count': len(cl['detections']),
            'metrics': best['metrics'],
            'tensor': None
        })

    unique_persons.sort(key=lambda p: p['metrics']['cosine_sim'], reverse=True)
    
    return unique_persons, None
