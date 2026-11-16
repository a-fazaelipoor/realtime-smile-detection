import os
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.conv2d(out)
        return self.relu(out * self.scale + x)


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out)
        return self.relu(out * self.scale + x)


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out) * self.scale + x
        return out if self.noReLU else self.relu(out)


class Mixed_6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)


class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1)


class InceptionResNetV1Backbone(nn.Module):
    def __init__(self, dropout_prob=0.6):
        super().__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.10) for _ in range(10)])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.20) for _ in range(5)])
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        return x


class SmileClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResNetV1Backbone()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(1)


class FaceAligner:
    """Real-time face detector and eye-based geometric aligner using MediaPipe FaceLandmarker."""
    def __init__(self, asset_path="models/face_landmarker.task", max_faces=5):
        if not os.path.exists(asset_path):
            os.makedirs(os.path.dirname(asset_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, asset_path)

        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_aligned_faces(self, frame_bgr, output_size=160, margin_ratio=0.20):
        """Returns a list of (bbox_xyxy, aligned_tensor) for all detected faces in frame."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        faces = []
        if not result.face_landmarks:
            return faces

        for landmarks in result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

            # Original bounding box for display overlay
            orig_x1 = max(0, int(np.min(pts[:, 0]) - 10))
            orig_y1 = max(0, int(np.min(pts[:, 1]) - 15))
            orig_x2 = min(w, int(np.max(pts[:, 0]) + 10))
            orig_y2 = min(h, int(np.max(pts[:, 1]) + 10))

            # Eye-based affine alignment
            right_eye = np.mean([pts[33], pts[133]], axis=0)
            left_eye = np.mean([pts[362], pts[263]], axis=0)
            dy = left_eye[1] - right_eye[1]
            dx = left_eye[0] - right_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            eyes_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)

            rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
            aligned_img = cv2.warpAffine(
                frame_rgb, rot_mat, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            aligned_pts = np.hstack([pts, ones]) @ rot_mat.T

            x_min, y_min = np.min(aligned_pts, axis=0)
            x_max, y_max = np.max(aligned_pts, axis=0)
            pad_x = (x_max - x_min) * margin_ratio
            pad_y = (y_max - y_min) * margin_ratio

            x1 = max(0, int(x_min - pad_x))
            y1 = max(0, int(y_min - pad_y))
            x2 = min(w, int(x_max + pad_x))
            y2 = min(h, int(y_max + pad_y))

            cropped = aligned_img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            resized = cv2.resize(cropped, (output_size, output_size))
            tensor = self.transform(resized)
            faces.append(((orig_x1, orig_y1, orig_x2, orig_y2), tensor))

        return faces