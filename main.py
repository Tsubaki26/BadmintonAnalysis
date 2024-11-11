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
COURT_IMG_PATH = "./img//court_vertical.jpg"
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

point_radius = 4 * 2
output_point_radius = 2 * 2
TRACK_FRAMES = 20

OUTPUT_INTERVAL = 15


def load_video(self, video_path):
    """
    動画を読み込み、フレーム情報を取得
    """
    self.cap = cv2.VideoCapture(video_path)
    self.num_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(
        f"#THE NUMBER OF FRAMES: {self.num_frame} | FRAME WIDTH: {self.cap_width} | FRAME HEIGHT: {self.cap_height}"
    )


class MatchAnalyzer:
    def __init__(
        self, court_img_path=COURT_IMG_PATH, court_size_wh=(COURT_WIDTH, COURT_HEIGHT)
    ):
        # コート画像の初期設定
        self.court_img_path = court_img_path
        self.court_size_wh = court_size_wh
        self.court_img = cv2.imread(self.court_img_path)
        self.court_img = cv2.resize(self.court_img, self.court_size_wh)
        self.court_img_for_output = self.court_img

        # 動画保存の初期設定
        self.is_save_video = False
        self.writer = None

        # player の位置ポインターの初期設定
        self.tracking_frames = 20
        self.point_radius = 8
        self.output_point_radius = 4

        # モデルの初期設定
        self.model_for_detecting_player = torch.hub.load(
            "ultralytics/yolov5", "yolov5s"
        )

        # 動画中のコートの中心の高さ
        self.img_middle = 617

        # 解析結果の出力用変数の初期設定
        self.output_interval = self.tracking_frames
        self.total_movement = [0, 0]
        self.total_movement_track = [[0, 0]]

    def setup_video_writer(self, video_path, video_size, output_path):
        """
        動画保存用 writer の設定をする
        """
        img_width, img_height = video_size
        probe = ffmpeg.probe(video_path)
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]
        fps = int(eval(video_streams[0]["r_frame_rate"]))

        self.writer = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(img_width, img_height),
                r=fps,
            )
            .output(output_path, pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, img):
        """
        Arguments:
        ・BGR画像
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.writer.stdin.write(img.tobytes())

    def finish_saving_video(self):
        self.writer.stdin.close()
        self.writer.wait()

    def make_point_size_list(self):
        """
        player ポインターの追跡用配列の作成
        配列の前から順に描画されるので、サイズは徐々に大きくなる
        """
        track_size_list = []
        # 配列の前から順に描画されるので、サイズは徐々に大きくなる
        for i in range(self.tracking_frames):
            size = int(
                self.point_radius
                / self.tracking_frames
                * (self.tracking_frames - i - 1)
                + 1
            )
            track_size_list.append(size)
        track_size_list.reverse()
        print(track_size_list)

        return track_size_list

    def set_tracking_point_parameters(
        self, tracking_frames, point_radius, output_point_radius
    ):
        self.tracking_frames = tracking_frames
        self.point_radius = point_radius
        self.output_point_radius = output_point_radius

    def detect_player(self, img, img_for_display):
        """
        playerの検出

        Returns:
        ・選手達の足元の座標リスト
        ・選手の足元に楕円、頭上に名前が追加された画像
        """
        results = self.model_for_detecting_player(img)
        player_foot_pos_xy_list = []
        # player1, player2の順で検出される(下にいる選手がplayer1)
        # このやりかたはよくないと思うから変更するかも
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = self.model_for_detecting_player.names[int(cls)]
            # コート外を除く
            if (
                conf > 0.6
                and label == "person"
                and (x1 > X_LIMIT[0])
                and (x2 < X_LIMIT[1])
                and (y2 > Y_LIMIT[0])
                and (y2 < Y_LIMIT[1])
            ):
                # playerの足元の中心座標を格納
                player_foot_pos_xy_list.append([x1 + (x2 - x1) // 2, y2, 1])

                """
                player の足元に表示する楕円の作成
                ・動画と同じサイズの黒い画像を作成し、そこに楕円を描画。
                ・動画のフレームに加算して半透明を再現。
                """
                ellipse_add = np.uint8(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))
                cv2.ellipse(
                    ellipse_add,
                    (
                        (x1 + (x2 - x1) // 2, y2),
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
                padding = [30, 15, 40, 40]  # top, bottom, left, right
                if y2 < self.img_middle:
                    player_name = "player2"
                    # playerのクリップ
                    # player2_clip = frame[
                    #     y1 - padding[0] : y2 + padding[1],
                    #     x1 - padding[2] : x2 + padding[3],
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
                    #     y1 - padding[0] : y2 + padding[1],
                    #     x1 - padding[2] : x2 + padding[3],
                    # ]
                    # player1_clip = cv2.resize(
                    #     player1_clip,
                    #     (player1_clip.shape[1] * 2, player1_clip.shape[0] * 2),
                    # )
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

    def get_perspective_trans_matrix(self, poslist1):
        """
        射影変換行列を作成する。
        Arguments:
        ・変換前の4点座標（左上から右回り）
        ・変換後の4点座標（左上から右回り）
        Returns:
        ・変換行列
        """
        poslist2 = [
            [0, 0],
            [COURT_WIDTH, 0],
            [COURT_WIDTH, COURT_HEIGHT],
            [0, COURT_HEIGHT],
        ]
        pts1 = np.float32(poslist1)
        pts2 = np.float32(poslist2)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return M

    def perspective_transform_position(self, player_foot_pos_xy_list, M):
        pos_xy_perspective_transformed = []
        for pos in player_foot_pos_xy_list:
            # playerのポジション座標を射影変換
            poslist_np = np.array(pos).reshape(-1, 1)
            transformed_point = np.dot(M, poslist_np)
            transformed_point = transformed_point / transformed_point[2]
            pos_xy_perspective_transformed.append(transformed_point)

    def plot_player_pos(self, court_img, pos_track, track_size_list):
        for i, pos_xy_ in enumerate(pos_track):
            for pos in pos_xy_:
                x, y, _ = pos
                x, y = int(x[0]), int(y[0])
                # 過去に行くほど明るくなる
                change_color = (self.tracking_frames - i - 1) * 5
                cv2.circle(
                    court_img,
                    (x, y),
                    track_size_list[i],
                    color=(
                        0 + change_color,
                        125 + change_color,
                        255 + change_color,
                    ),
                    thickness=-1,
                )
        return court_img

    def add_total_movement(self, pos_track):
        # 総移動距離の加算
        move_amount_p1 = math.sqrt(
            (pos_track[0][0][0] - pos_track[self.tracking_frames - 1][0][0]) ** 2
            + (pos_track[0][0][1] - pos_track[self.tracking_frames - 1][0][1]) ** 2
        )
        move_amount_p2 = math.sqrt(
            (pos_track[0][1][0] - pos_track[self.tracking_frames - 1][1][0]) ** 2
            + (pos_track[0][1][1] - pos_track[self.tracking_frames - 1][1][1]) ** 2
        )
        self.total_movement[0] += move_amount_p1
        self.total_movement[1] += move_amount_p2
        self.total_movement_track.append(self.total_movement.copy())

    def add_player_position(self, pos_xy_perspective_transformed):
        circle_add = np.uint8(
            np.zeros((self.court_size_wh[0], self.court_size_wh[1], 3))
        )
        for pos in pos_xy_perspective_transformed:
            x, y, _ = pos
            x, y = int(x[0]), int(y[0])
            cv2.circle(
                circle_add,
                (x, y),
                self.output_point_radius,
                color=(50, 50, 50),
                thickness=-1,
            )
            self.court_img_for_output = cv2.add(self.court_img_for_output, circle_add)

    def make_total_movement_graph(self):
        y1 = []
        y2 = []
        for ys in self.total_movement_track:
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

    def make_player_position_plot(self):
        # cv2.imshow("court", self.court_img_for_output)
        # cv2.waitKey()
        cv2.imwrite(f"./outputs/output{t}.jpg", self.court_img_for_output)

    def analyze_match(
        self,
        match_video_path,
        is_save_video=False,
        output_path="./out.mp4",
    ):
        """
        試合動画の解析
        """
        self.is_save_video = is_save_video
        count = 0  # フレームのカウンター

        # 動画、コート画像の読み込み
        load_video(match_video_path)

        if self.is_save_video:
            self.setup_video_writer(
                match_video_path, (self.cap_width, self.cap_height), output_path
            )

        # player ポインターの追跡用配列
        pos_track = []
        track_size_list = self.make_point_size_list()

        while self.cap.isOpened():
            count += 1

            # 進捗状況の表示
            if count % 100 == 0:
                print(f"NOW: {(count/self.num_frame*100):.2f}%")

            ret, frame = self.cap.read()
            if not ret:
                # 動画の最終フレームを過ぎたらループから抜ける
                print("============ FINISHED ============")
                break

            # 動画用に画像をコピー
            court_for_display = self.court_img.copy()
            img_for_display = frame.copy()

            #  player の検出
            player_foot_pos_xy_list, img_for_display = self.detect_player(
                frame, img_for_display
            )

            """
            player座標を射影変換し、コート画像上のplayerの位置を取得
            """
            # 射影変換前の座標設定はいまのところ手動
            poslist1 = [
                COURT_XY_BEFORE_WARP[0],
                COURT_XY_BEFORE_WARP[1],
                COURT_XY_BEFORE_WARP[2],
                COURT_XY_BEFORE_WARP[3],
            ]
            M = self.get_perspective_trans_matrix(poslist1)
            # image = cv2.warpPerspective(frame, M, (COURT_WIDTH, COURT_HEIGHT))
            # 両プレイヤーが検出できている時のみ記録。サイズが合わないから。いい方法が見つかれば変更する。
            if len(player_foot_pos_xy_list) >= 2:
                pos_xy_perspective_transformed = self.perspective_transform_position(
                    player_foot_pos_xy_list, M
                )
                pos_track.append(pos_xy_perspective_transformed)
                # TRACK_FRAMESまでしか追跡しない
                if len(pos_track) > self.tracking_frames:
                    pos_track.pop(0)

            """
            コート画像に player の位置をプロット
            """
            court_for_display = self.plot_player_pos(
                court_for_display, pos_track, track_size_list
            )

            """
            動画とコート画像の合成（左上）
            """
            img_for_display[50 : 50 + COURT_HEIGHT, 50 : 50 + COURT_WIDTH] = (
                court_for_display
            )

            if self.is_save_video:
                self.write_frame(img_for_display)
            else:
                """
                フレームの表示
                """
                cv2.imshow("match", img_for_display)
                # cv2.imshow("player1", player1_clip)
                # cv2.imshow("player2", player2_clip)
                cv2.waitKey()

            """
            解析
            """
            if count % self.output_interval == 0 and count >= self.tracking_frames:
                # 総移動距離の加算
                self.add_total_movement(pos_track)

                # player の位置を記録
                self.add_player_position(pos_xy_perspective_transformed)

        if self.is_save_video:
            self.finish_saving_video()

        # 総移動距離のグラフ
        self.make_total_movement_graph()
        # player の位置プロット
        self.make_player_position_plot()


if __name__ == "__main__":
    analyzer = MatchAnalyzer()
    analyzer.analyze_match(VIDEO_PATH)
