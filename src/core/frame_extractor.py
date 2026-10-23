from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval_s: float = 2.0,
) -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frames = []
    timestamp = 0.0

    while timestamp < duration:
        frame_path = frames_dir / f"frame_{timestamp:08.1f}.jpg"

        if not frame_path.is_file():
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000,
            )

            success, frame = capture.read()

            if success:
                cv2.imwrite(str(frame_path), frame)
                print(f"Captured frame on {timestamp:08.1f} s...")

        if frame_path.is_file():
            frames.append({
                "timestamp": timestamp,
                "path": frame_path,
            })

        timestamp += interval_s

    capture.release()

    return frames