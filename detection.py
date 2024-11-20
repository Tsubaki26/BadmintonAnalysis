import warnings

import cv2
import numpy as np
import torch


class Detection:
    def __init__(self):
        # モデルの初期設定
        self.model_for_detecting_player = torch.hub.load(
            "ultralytics/yolov5", "yolov5s"
        )
        # 修正できないので注意文を無視する
        warnings.filterwarnings("ignore", category=FutureWarning)

    def detect_player(
        self, img, video_size, img_for_display, xlimit, ylimit, court_middle_points
    ):
        """
        playerの検出

        Returns:
        ・選手達の足元の座標リスト
        ・選手の足元に楕円、頭上に名前が追加された画像
        """
        with torch.amp.autocast("cuda"):
            results = self.model_for_detecting_player(img)
        player_foot_pos_xy_list = []
        # player1, player2の順で検出される(下にいる選手がplayer1)
        # このやりかたはよくないと思うから変更するかも
        y2_0 = 0
        for i, det in enumerate(results.xyxy[0].cpu().numpy()):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = self.model_for_detecting_player.names[int(cls)]
            # コート外を除く
            if (
                conf > 0.6
                and label == "person"
                and (x1 > xlimit[0])
                and (x2 < xlimit[1])
                and (y2 > ylimit[0])
                and (y2 < ylimit[1])
            ):
                # playerの足元の中心座標を格納
                if i == 0:
                    y2_0 = y2
                    player_foot_pos_xy_list.append([x1 + (x2 - x1) // 2, y2, 1])
                elif i == 1:
                    if y2 < y2_0:
                        player_foot_pos_xy_list.append([x1 + (x2 - x1) // 2, y2, 1])
                    else:
                        player_foot_pos_xy_list.insert(0, [x1 + (x2 - x1) // 2, y2, 1])
                """
                player の足元に表示する楕円の作成
                ・動画と同じサイズの黒い画像を作成し、そこに楕円を描画。
                ・動画のフレームに加算して半透明を再現。
                """
                ellipse_add = np.uint8(np.zeros((video_size[1], video_size[0], 3)))
                x_center = x1 + (x2 - x1) // 2
                cv2.ellipse(
                    ellipse_add,
                    (
                        (x_center, y2),
                        (x2 - x1, 30),  # 横直径、縦直径
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
                # コートの中心線の係数
                a = (
                    -1
                    * (court_middle_points[1][1] - court_middle_points[0][1])
                    / (court_middle_points[1][0] - court_middle_points[0][0])
                )
                # コートの中心線の切片
                b = court_middle_points[1][1] - a * court_middle_points[1][0]
                if y2 < a * x_center + b:
                    player_name = "player2"
                else:
                    player_name = "player1"
                cv2.putText(
                    img_for_display,
                    f"{player_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 125, 255),
                    2,
                )
        return player_foot_pos_xy_list, img_for_display
