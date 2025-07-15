import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchvision import transforms
from torchvision.models import resnet50

model = YOLO("best.pt")
tracker = DeepSort(max_age=50)
embedder = resnet50(pretrained=True)
embedder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
])

def get_embedding(crop):
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    with torch.no_grad():
        img = transform(crop).unsqueeze(0)
        emb = embedder(img).squeeze()
        return emb.numpy()

def run_reid_on_video(video_path, label_prefix, output_path):
    cap = cv2.VideoCapture(video_path)
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            crop = frame[y1:y2, x1:x2]
            emb = get_embedding(crop)
            if emb is None:
                continue
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "0", emb))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_prefix}_Player {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        out.write(frame)

    cap.release()
    out.release()

run_reid_on_video("tacticam.mp4", "T", "tacticam_IDs.mp4")
run_reid_on_video("broadcast.mp4", "B", "broadcast_IDs.mp4")