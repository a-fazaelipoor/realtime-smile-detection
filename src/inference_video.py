import argparse
import time
import cv2
import torch
from model import SmileClassifier, FaceAligner


def main():
    parser = argparse.ArgumentParser(description="Real-time Smile Detection on Webcam or Video File")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--weights", type=str, default="models/best_smile_model.pth", help="Path to trained .pth weights")
    parser.add_argument("--output", type=str, default="", help="Optional path to save annotated output video (e.g. output.mp4)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for smile classification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # Load trained model
    model = SmileClassifier().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Initialize face aligner
    aligner = FaceAligner(asset_path="models/face_landmarker.task", max_faces=5)

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    prev_time = time.time()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = aligner.extract_aligned_faces(frame)
        if faces:
            bboxes = [f[0] for f in faces]
            tensors = torch.stack([f[1] for f in faces]).to(device)

            with torch.no_grad():
                logits = model(tensors)
                probs = torch.sigmoid(logits).cpu().numpy()

            for (x1, y1, x2, y2), prob in zip(bboxes, probs):
                is_smile = prob >= args.threshold
                label = f"Smile: {prob*100:.1f}%" if is_smile else f"Non-Smile: {(1-prob)*100:.1f}%"
                color = (0, 200, 0) if is_smile else (0, 0, 220)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 6, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        curr_time = time.time()
        fps_text = f"FPS: {1.0 / max(1e-5, curr_time - prev_time):.1f}"
        prev_time = curr_time
        cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("GENKI-4K Smile Detection (Press Q to Exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()