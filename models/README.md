# Trained weights

The weights are not stored in the Git history (`*.pt` is ignored). They are attached to the GitHub release of this repository:

| File | Model | Classes | Training run |
|---|---|---|---|
| `yolov8m_ppe_5class_v1.pt` | YOLOv8m | 5 (mask, helmet, belt, vest, glasses) | 3 Jan 2026, `training/v1_five_classes` (MD5 `149ac83abbb0f70d060c9c65eb1a0bd0`) |
| `yolov8m_ppe_10class_v2.pt` | YOLOv8m | 10 | 29 May 2026, `training/v2_ten_classes` (MD5 `d3830f02ce7818eaa7e1d2eb684cd2ee`) |

Copy the file you want to use to `ai_server/best.pt` (or set `MODEL_PPE_PATH` in `ai_server/.env`). The detection servers expect the ten-class model; class index 6 must be `ear_protection`.
