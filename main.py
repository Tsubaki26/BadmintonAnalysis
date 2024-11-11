import cv2
import numpy as np
import torch
import time
import ffmpeg  # type: ignore
import math
import matplotlib.pyplot as plt


# VIDEO_PATH = "./videos/cutvirsion.mp4"
# VIDEO_PATH = "./videos/cutversion_nopeople.mp4"
VIDEO_PATH = "./videos/cutversion2.mp4"
COURT_PATH = "./img//court_vertical.jpg"
t = time.time()
OUTPUT_VIDEO_PATH = f"./outputs/out{t}.mp4"

X_LIMIT = [311, 1588]
Y_LIMIT = [372, 1080]
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
IMG_MIDDLE = 617

COURT_WIDTH = 183 * 2
COURT_HEIGHT = 402 * 2

# 左上から右回り
COURT_XY_BEFORE_WARP = [
    [645, 397],
    [1288, 401],
    [1518, 997],
    [393, 988],
]

DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

SAVE_VIDEO = False

POINT_SIZE = 4 * 2
OUTPUT_POINT_SIZE = 2 * 2
TRACK_FRAMES = 20

OUTPUT_INTERVAL = 15

if __name__ == "__main__":
    count = 0  # フレームのカウンター

    """
    動画、コート画像の読み込み
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FRAMES: ", num_frame)
    court_img = cv2.imread(COURT_PATH)
    court_img = cv2.resize(court_img, (COURT_WIDTH, COURT_HEIGHT))
    court_for_output = court_img

    """
    モデル定義
    """
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    if SAVE_VIDEO:
        """
        動画保存の準備
        """
        probe = ffmpeg.probe(VIDEO_PATH)
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]
        fps = int(eval(video_streams[0]["r_frame_rate"]))

        writer = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(IMG_WIDTH, IMG_HEIGHT),
                r=fps,
            )
            .output(OUTPUT_VIDEO_PATH, pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    """
    player ポインターの追跡用配列
    """
    pos_track = []
    track_size = []
    # 配列の前から順に描画されるので、サイズは徐々に大きくなる
    for i in range(TRACK_FRAMES):
        size = int(POINT_SIZE / TRACK_FRAMES * (TRACK_FRAMES - i - 1) + 1)
        track_size.append(size)
    track_size.reverse()
    print(track_size)

    """
    総移動距離のグラフ用配列
    """
    total_movement = [0, 0]
    total_movement_track = [[0, 0]]

    while cap.isOpened():
        count += 1

        # 進捗状況の表示
        if count % 100 == 0:
            print(f"NOW: {(count/num_frame*100):.2f}%")

        ret, frame = cap.read()
        if not ret:
            # 動画の最終フレームを過ぎたらループから抜ける
            print("============ FINISHED ============")
            break

        court_for_display = court_img.copy()
        img_for_display = frame.copy()

        """
        playerの検出
        """
        results = model(frame)
        pos_xy = []
        # player1, player2の順で検出される
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            label = model.names[int(cls)]
            # コート外を除く
            if (
                conf > 0.6
                and label == "person"
                and (int(x1) > X_LIMIT[0])
                and (int(x2) < X_LIMIT[1])
                and (int(y2) > Y_LIMIT[0])
                and (int(y2) < Y_LIMIT[1])
            ):
                # playerの足元の中心座標を格納
                pos_xy.append([int(x1) + (int(x2) - int(x1)) // 2, int(y2), 1])

                """
                player の足元に表示する楕円の作成
                ・動画と同じサイズの黒い画像を作成し、そこに楕円を描画。
                ・動画のフレームに加算して半透明を再現。
                """
                ellipse_add = np.uint8(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))
                cv2.ellipse(
                    ellipse_add,
                    (
                        (int(x1) + (int(x2) - int(x1)) // 2, int(y2)),
                        (int(x2) - int(x1), 30),  # 横直径、縦直径
                        0,
                    ),
                    color=(50, 20, 20),
                    thickness=-1,
                )
                img_for_display = cv2.add(img_for_display, ellipse_add)

                """
                playerの名前の表示
                ・とりあえず下を player1 としている。
                """
                padding = [30, 15, 40, 40]  # top, bottom, left, right
                if int(y2) < IMG_MIDDLE:
                    player_name = "player2"
                    # playerのクリップ
                    # player2_clip = frame[
                    #     int(y1) - padding[0] : int(y2) + padding[1],
                    #     int(x1) - padding[2] : int(x2) + padding[3],
                    # ]
                    # player2_clip = cv2.resize(
                    #     player2_clip,
                    #     (player2_clip.shape[1] * 2, player2_clip.shape[0] * 2),
                    # )
                    # print(player2_clip.shape)
                else:
                    player_name = "player1"
                    # playerのクリップ
                    # player1_clip = frame[
                    #     int(y1) - padding[0] : int(y2) + padding[1],
                    #     int(x1) - padding[2] : int(x2) + padding[3],
                    # ]
                    # player1_clip = cv2.resize(
                    #     player1_clip,
                    #     (player1_clip.shape[1] * 2, player1_clip.shape[0] * 2),
                    # )
                cv2.putText(
                    img_for_display,
                    f"{player_name} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 125, 255),
                    2,
                )

        """
        射影変換
        ・まだ変換前の座標設定は手動
        """
        pts1 = np.float32(
            [
                COURT_XY_BEFORE_WARP[0],
                COURT_XY_BEFORE_WARP[1],
                COURT_XY_BEFORE_WARP[2],
                COURT_XY_BEFORE_WARP[3],
            ]
        )
        pts2 = np.float32(
            [[0, 0], [COURT_WIDTH, 0], [COURT_WIDTH, COURT_HEIGHT], [0, COURT_HEIGHT]]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # image = cv2.warpPerspective(frame, M, (COURT_WIDTH, COURT_HEIGHT))

        """
        コート画像上のplayerの位置を計算
        """
        pos_xy_warp = []
        print(len(pos_xy))
        # 両プレイヤーが検出できている時のみ記録。サイズが合わないから。いい方法が見つかれば変更する。
        if len(pos_xy) >= 2:
            for pos in pos_xy:
                # playerのポジション座標を射影変換
                poslist_np = np.array(pos).reshape(-1, 1)
                transformed_point = np.dot(M, poslist_np)
                transformed_point = transformed_point / transformed_point[2]
                pos_xy_warp.append(transformed_point)

            pos_track.append(pos_xy_warp)
            # TRACK_FRAMESまでしか追跡しない
            if len(pos_track) > TRACK_FRAMES:
                pos_track.pop(0)

        """
        コート画像にプレイヤーの位置をプロット
        """
        for i, pos_xy_warp_ in enumerate(pos_track):
            for pos in pos_xy_warp_:
                x, y, _ = pos
                x, y = int(x[0]), int(y[0])
                # 過去に行くほど明るくなる
                change_color = (TRACK_FRAMES - i - 1) * 5
                cv2.circle(
                    court_for_display,
                    (x, y),
                    track_size[i],
                    color=(0 + change_color, 125 + change_color, 255 + change_color),
                    thickness=-1,
                )

        # 動画とコート画像の合成（左上）
        img_for_display[50 : 50 + COURT_HEIGHT, 50 : 50 + COURT_WIDTH] = (
            court_for_display
        )

        if SAVE_VIDEO:
            img_for_display = cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB)
            writer.stdin.write(img_for_display.tobytes())
        else:
            cv2.imshow("match", img_for_display)
            # cv2.imshow("player1", player1_clip)
            # cv2.imshow("player2", player2_clip)
            cv2.waitKey()

        """
        出力
        ・総移動距離の加算
        ・一定間隔でプレイヤーの位置をプロット
        """
        if count % OUTPUT_INTERVAL == 0 and count >= TRACK_FRAMES:
            # 総移動距離の加算
            move_amount_p1 = math.sqrt(
                (pos_track[0][0][0] - pos_track[TRACK_FRAMES - 1][0][0]) ** 2
                + (pos_track[0][0][1] - pos_track[TRACK_FRAMES - 1][0][1]) ** 2
            )
            move_amount_p2 = math.sqrt(
                (pos_track[0][1][0] - pos_track[TRACK_FRAMES - 1][1][0]) ** 2
                + (pos_track[0][1][1] - pos_track[TRACK_FRAMES - 1][1][1]) ** 2
            )
            total_movement[0] += move_amount_p1
            total_movement[1] += move_amount_p2
            total_movement_track.append(total_movement.copy())

            # 一定間隔でプレイヤーの位置をプロット
            circle_add = np.uint8(np.zeros((COURT_HEIGHT, COURT_WIDTH, 3)))
            for pos in pos_xy_warp:
                x, y, _ = pos
                x, y = int(x[0]), int(y[0])
                cv2.circle(
                    circle_add,
                    (x, y),
                    OUTPUT_POINT_SIZE,
                    color=(50, 50, 50),
                    thickness=-1,
                )
                court_for_output = cv2.add(court_for_output, circle_add)

    if SAVE_VIDEO:
        writer.stdin.close()
        writer.wait()

    # 総移動距離のグラフ
    y1 = []
    y2 = []
    for ys in total_movement_track:
        y1.append(ys[0])
        y2.append(ys[1])
    print(len(y1))
    x = list(range(len(y1)))
    plt.plot(x, y1, label="player1")
    plt.plot(x, y2, label="player2")
    plt.legend()
    plt.xlabel("frame (1/15)")
    plt.ylabel("total movement")
    plt.savefig("./outputs/total_movement.jpg")
    plt.show()
    plt.close()

    # cv2.imshow("court", court_for_output)
    # cv2.waitKey()

    cv2.imwrite(f"./outputs/output{t}.jpg", court_for_output)
