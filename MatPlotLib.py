"""
    D435 카메라 + MediaPipe Holistic: 팔 + 손가락(21개) 3D 실시간 시각화
    - Pose  : 전신 스켈레톤 + 팔 관절 3D (뎁스 카메라 기반)
    - Hands : 양손 손가락 21개 랜드마크 3D (뎁스 카메라 기반)
    - matplotlib 3D 플롯 : 어깨→팔꿈치→손목 + 손가락 실시간 도식화

    [Latency 측정]
    - Frame Latency  : wait_for_frames() 반환 직후 ~ 첫 이미지 배열 준비 완료
    - Total Latency  : wait_for_frames() 반환 직후 ~ matplotlib 플롯 업데이트 완료
    두 값 모두 CV 창 오버레이 + 터미널에 실시간 출력
"""
import sys
sys.path.insert(0, "/home/ohheemin/.local/lib/python3.10/site-packages")

import time                          # ← Latency 측정용
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import deque

COLOR_W, COLOR_H, FPS = 848, 480, 30
DEPTH_W, DEPTH_H      = 848, 480

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_pinky","right_pinky",
    "left_index","right_index",
    "left_thumb","right_thumb",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

HAND_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # 엄지
    (0,5),(5,6),(6,7),(7,8),           # 검지
    (0,9),(9,10),(10,11),(11,12),      # 중지
    (0,13),(13,14),(14,15),(15,16),    # 약지
    (0,17),(17,18),(18,19),(19,20),    # 소지
    (5,9),(9,13),(13,17),              # 손바닥 가로
]

ARM_LEFT  = [11, 13, 15]   # shoulder → elbow → wrist
ARM_RIGHT = [12, 14, 16]
ARM_ALL   = list(set(ARM_LEFT + ARM_RIGHT))

DISPLAY_INDICES = [0, 11,12, 13,14, 15,16, 23,24, 25,26, 27,28]

COLOR_JOINT   = (0,   255, 100)
COLOR_TEXT    = (255, 255,   0)
COLOR_TEXT_BG = (0,   0,     0)
COLOR_SKEL    = (100, 200, 255)
COLOR_LH_CV   = (255, 180,  50)   # 왼손 BGR
COLOR_RH_CV   = (50,  180, 255)   # 오른손 BGR

TRAIL_LEN  = 40
PLOT_EVERY = 3

mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def draw_text_with_bg(img, text, pos, font_scale=0.38, thickness=1,
                       text_color=COLOR_TEXT, bg_color=COLOR_TEXT_BG, alpha=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = img.copy()
    cv2.rectangle(overlay,(x-2,y-th-2),(x+tw+2,y+baseline+2), bg_color,-1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text,(x,y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def get_3d_point(depth_frame, intrinsics, px, py):
    """픽셀 → 카메라 좌표계 3D 점 [X, Y, Z] (미터)"""
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if px < 0 or py < 0 or px >= dw or py >= dh:
        return None
    d = depth_frame.get_distance(int(px), int(py))
    if d <= 0.1 or d > 6.0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d)


def collect_hand_3d(hand_lms, depth_frame, intrinsics, img_w, img_h):
    pts = []
    for lm in hand_lms.landmark:
        px = int(lm.x * img_w)
        py = int(lm.y * img_h)
        pts.append(get_3d_point(depth_frame, intrinsics, px, py))
    return pts


def draw_hand_on_frame(disp, hand_lms, img_w, img_h, cv_color, depth_frame, intrinsics):
    pts_px = [(int(lm.x * img_w), int(lm.y * img_h))
              for lm in hand_lms.landmark]

    for a, b in HAND_CONNECTIONS:
        cv2.line(disp, pts_px[a], pts_px[b], cv_color, 1, cv2.LINE_AA)

    label_indices = {0,4,8,12,16,20}
    for i, (px, py) in enumerate(pts_px):
        pt3d = get_3d_point(depth_frame, intrinsics, px, py)
        r = 4 if i in label_indices else 2
        cv2.circle(disp, (px,py), r, cv_color, -1, cv2.LINE_AA)
        if i in label_indices and pt3d is not None:
            x3,y3,z3 = pt3d
            draw_text_with_bg(disp,
                f"{HAND_NAMES[i]}({x3:+.2f},{z3:.2f})",
                (px+4, py+4), font_scale=0.30,
                text_color=(255,255,255))


# ── matplotlib 3D 초기화 ─────────────────────────────────────────────────────
def init_3d_plot():
    plt.ion()
    fig = plt.figure("팔+손 3D 실시간", figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    _setup_ax(ax)
    fig.tight_layout()
    return fig, ax


def _setup_ax(ax):
    ax.set_xlabel("X (m)  ←→")
    ax.set_ylabel("Z (m)  깊이→")
    ax.set_zlabel("높이 (m)")
    ax.set_title("팔 + 손가락 3D\n왼손: 파랑  /  오른손: 빨강", fontsize=10)
    ax.set_xlim(-0.8,  0.8)
    ax.set_ylim( 0.3,  3.0)
    ax.set_zlim(-0.9,  0.5)
    ax.view_init(elev=20, azim=-65)


def _mpl_xyz(pt):
    return pt[0], pt[2], -pt[1]


def draw_arm_mpl(ax, coords, indices, color, label):
    pts   = [coords.get(i) for i in indices]
    valid = [p is not None for p in pts]
    if not any(valid):
        return
    jlabels = ["어깨","팔꿈치","손목"]
    for p, v, jl in zip(pts, valid, jlabels):
        if v:
            mx, my, mz = _mpl_xyz(p)
            ax.scatter(mx, my, mz, color=color, s=70, zorder=5)
            ax.text(mx, my, mz+0.04, jl, color=color, fontsize=7, ha="center")
    for k in range(len(pts)-1):
        if valid[k] and valid[k+1]:
            a,b = pts[k], pts[k+1]
            ax.plot([a[0],b[0]], [a[2],b[2]], [-a[1],-b[1]],
                    color=color, linewidth=3,
                    label=label if k==0 else "")


def draw_hand_mpl(ax, hand_pts, color, dot_size=20):
    if not hand_pts:
        return
    for i, p in enumerate(hand_pts):
        if p is None:
            continue
        mx, my, mz = _mpl_xyz(p)
        s = dot_size * 2 if i in {0,4,8,12,16,20} else dot_size
        ax.scatter(mx, my, mz, color=color, s=s, zorder=6, alpha=0.85)
    for a, b in HAND_CONNECTIONS:
        pa, pb = hand_pts[a], hand_pts[b]
        if pa is None or pb is None:
            continue
        ax.plot([pa[0],pb[0]], [pa[2],pb[2]], [-pa[1],-pb[1]],
                color=color, linewidth=1.5, alpha=0.8)


def draw_trail_mpl(ax, trail, color):
    if len(trail) < 2:
        return
    arr = np.array(trail)
    alphas = np.linspace(0.05, 0.45, len(arr))
    for i in range(len(arr)-1):
        ax.plot([arr[i,0], arr[i+1,0]],
                [arr[i,2], arr[i+1,2]],
                [-arr[i,1], -arr[i+1,1]],
                color=color, alpha=float(alphas[i]), linewidth=1.0)


def update_3d_plot(ax, fig, arm_coords,
                   lhand_pts, rhand_pts,
                   trail_l, trail_r):
    ax.cla()
    _setup_ax(ax)

    draw_arm_mpl(ax, arm_coords, ARM_LEFT,  "blue", "왼팔")
    draw_arm_mpl(ax, arm_coords, ARM_RIGHT, "red",  "오른팔")
    draw_hand_mpl(ax, lhand_pts, "cornflowerblue")
    draw_hand_mpl(ax, rhand_pts, "tomato")
    draw_trail_mpl(ax, trail_l, "blue")
    draw_trail_mpl(ax, trail_r, "red")

    handles = [
        plt.Line2D([0],[0], color="blue",            linewidth=3, label="왼팔/손"),
        plt.Line2D([0],[0], color="red",             linewidth=3, label="오른팔/손"),
        plt.Line2D([0],[0], color="gray", alpha=0.4, linewidth=1, label="손목 궤적"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # RealSense
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  FPS)

    print("리얼센스 연결 중...")
    profile    = pipeline.start(cfg)
    align      = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile().get_intrinsics())
    print(f"해상도 {COLOR_W}x{COLOR_H} @ {FPS}fps  |  "
          f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fig3d, ax3d = init_3d_plot()

    arm_coords = {}
    lhand_pts  = []
    rhand_pts  = []
    trail_l    = deque(maxlen=TRAIL_LEN)
    trail_r    = deque(maxlen=TRAIL_LEN)

    # ── Latency 추적용 변수 ───────────────────────────────────────────────────
    frame_lat_buf = deque(maxlen=30)   # Frame Latency  이동 평균 (30프레임)
    total_lat_buf = deque(maxlen=30)   # Total Latency  이동 평균
    lat_frame_ms  = 0.0
    lat_total_ms  = 0.0
    lat_frame_avg = 0.0
    lat_total_avg = 0.0
    # ─────────────────────────────────────────────────────────────────────────

    show_depth = False
    frame_idx  = 0
    print("실행 중 — Q/ESC:종료  S:저장  D:깊이맵")

    try:
        while True:

            # ━━━ ① 기준 시각: wait_for_frames() 반환 직후 ━━━━━━━━━━━━━━━━━━━
            frames      = pipeline.wait_for_frames()
            t0          = time.perf_counter()          # 프레임 수신 완료 시각

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # ━━━ ② 이미지 배열 준비 완료 → Frame Latency 확정 ━━━━━━━━━━━━━━
            color_img    = np.asanyarray(color_frame.get_data())
            t_image_ready = time.perf_counter()
            lat_frame_ms  = (t_image_ready - t0) * 1000.0   # ms
            frame_lat_buf.append(lat_frame_ms)
            lat_frame_avg = sum(frame_lat_buf) / len(frame_lat_buf)

            depth_arr = np.asanyarray(depth_frame.get_data())
            h, w      = color_img.shape[:2]
            disp      = color_img.copy()

            rgb    = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            cur_arm = {}
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                for ca, cb in POSE_CONNECTIONS:
                    a, b = lms[ca], lms[cb]
                    if a.visibility > 0.3 and b.visibility > 0.3:
                        cv2.line(disp,
                                 (int(a.x*w), int(a.y*h)),
                                 (int(b.x*w), int(b.y*h)),
                                 COLOR_SKEL, 2, cv2.LINE_AA)

                for idx in DISPLAY_INDICES:
                    lm = lms[idx]
                    if lm.visibility < 0.3:
                        continue
                    px, py = int(lm.x*w), int(lm.y*h)
                    pt3d   = get_3d_point(depth_frame, intrinsics, px, py)

                    if idx in ARM_ALL and pt3d is not None:
                        cur_arm[idx] = pt3d

                    cv2.circle(disp, (px,py), 5, COLOR_JOINT, -1, cv2.LINE_AA)
                    cv2.circle(disp, (px,py), 7, (255,255,255),  1, cv2.LINE_AA)

                    name = LANDMARK_NAMES[idx]
                    if pt3d:
                        x3,y3,z3 = pt3d
                        draw_text_with_bg(disp,
                            f"{name}({x3:+.2f},{y3:+.2f},{z3:.2f})m",
                            (px+8, py+4))
                    else:
                        draw_text_with_bg(disp,
                            f"{name} MP({lm.x:.2f},{lm.y:.2f})",
                            (px+8, py+4))

                for idx2, lm2 in enumerate(lms):
                    if idx2 in DISPLAY_INDICES or lm2.visibility < 0.4:
                        continue
                    cv2.circle(disp,(int(lm2.x*w),int(lm2.y*h)),
                               3,(180,180,180),-1,cv2.LINE_AA)

            for idx in ARM_ALL:
                if idx in cur_arm:
                    arm_coords[idx] = cur_arm[idx]

            if result.left_hand_landmarks:
                lhand_pts = collect_hand_3d(
                    result.left_hand_landmarks, depth_frame, intrinsics, w, h)
                draw_hand_on_frame(disp, result.left_hand_landmarks,
                                   w, h, COLOR_LH_CV, depth_frame, intrinsics)
                if lhand_pts[0] is not None:
                    trail_l.append(lhand_pts[0])

            if result.right_hand_landmarks:
                rhand_pts = collect_hand_3d(
                    result.right_hand_landmarks, depth_frame, intrinsics, w, h)
                draw_hand_on_frame(disp, result.right_hand_landmarks,
                                   w, h, COLOR_RH_CV, depth_frame, intrinsics)
                if rhand_pts[0] is not None:
                    trail_r.append(rhand_pts[0])

            if frame_idx % PLOT_EVERY == 0:
                update_3d_plot(ax3d, fig3d,
                               arm_coords, lhand_pts, rhand_pts,
                               trail_l, trail_r)

                t_plot_done  = time.perf_counter()
                lat_total_ms = (t_plot_done - t0) * 1000.0
                total_lat_buf.append(lat_total_ms)
                lat_total_avg = sum(total_lat_buf) / len(total_lat_buf)

                print(
                    f"[Frame {frame_idx:05d}] "
                    f"Frame Latency: {lat_frame_ms:6.2f} ms (avg {lat_frame_avg:6.2f} ms) | "
                    f"Total Latency: {lat_total_ms:7.2f} ms (avg {lat_total_avg:7.2f} ms)"
                )

            draw_text_with_bg(
                disp,
                f"Frame Lat: {lat_frame_ms:6.2f} ms  avg {lat_frame_avg:6.2f} ms",
                (10, h - 110),
                font_scale=0.42,
                text_color=(100, 255, 180),
            )

            draw_text_with_bg(
                disp,
                f"Total Lat: {lat_total_ms:6.2f} ms  avg {lat_total_avg:6.2f} ms",
                (10, h - 90),
                font_scale=0.42,
                text_color=(255, 200, 80),
            )

            draw_text_with_bg(disp, f"Frame:{frame_idx}", (10,22),
                              font_scale=0.45, text_color=(200,255,200))
            draw_text_with_bg(disp, "Q/ESC:Quit  S:Save  D:Depth", (10,44),
                              font_scale=0.40, text_color=(200,200,200))

            detected  = "Detected" if result.pose_landmarks else "No Person"
            dot_color = (0,255,0) if result.pose_landmarks else (0,0,255)
            cv2.circle(disp, (w-20,20), 8, dot_color, -1)
            draw_text_with_bg(disp, detected, (w-110,24),
                              font_scale=0.40, text_color=(255,255,255))

            lh_status = "왼손 O" if result.left_hand_landmarks  else "왼손 X"
            rh_status = "오른손 O" if result.right_hand_landmarks else "오른손 X"
            draw_text_with_bg(disp, lh_status, (w-120, 44),
                              font_scale=0.38,
                              text_color=(200,200,255) if result.left_hand_landmarks
                                         else (100,100,100))
            draw_text_with_bg(disp, rh_status, (w-120, 62),
                              font_scale=0.38,
                              text_color=(255,200,200) if result.right_hand_landmarks
                                         else (100,100,100))

            y_off = h - 130  
            for arm_name, indices, col in [
                    ("LEFT ARM",  ARM_LEFT,  (200,200,255)),
                    ("RIGHT ARM", ARM_RIGHT, (255,200,200))]:
                draw_text_with_bg(disp, arm_name, (10,y_off),
                                  font_scale=0.38, text_color=col)
                y_off -= 16
                for idx in reversed(indices):
                    pt  = arm_coords.get(idx)
                    jn  = LANDMARK_NAMES[idx].split("_")[-1]
                    txt = (f"  {jn}:({pt[0]:+.2f},{pt[1]:+.2f},{pt[2]:.2f})m"
                           if pt else f"  {jn}: --")
                    draw_text_with_bg(disp, txt, (10,y_off),
                                      font_scale=0.32, text_color=col)
                    y_off -= 13
                y_off -= 5

            cv2.imshow("D435 + Holistic | 팔+손가락 3D", disp)

            if show_depth:
                dc = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_arr, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Depth Map", dc)
            else:
                try: cv2.destroyWindow("Depth Map")
                except: pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fn = f"pose_capture_{frame_idx:04d}.png"
                cv2.imwrite(fn, disp)
                fig3d.savefig(f"pose_3d_{frame_idx:04d}.png", dpi=120)
                print(f"[SAVE] {fn}  +  pose_3d_{frame_idx:04d}.png")
            elif key == ord('d'):
                show_depth = not show_depth

            frame_idx += 1

    finally:
        pipeline.stop()
        holistic.close()
        cv2.destroyAllWindows()
        plt.close("all")
        print("종료 완료.")


if __name__ == "__main__":
    main()