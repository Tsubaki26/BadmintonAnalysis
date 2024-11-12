import time
import math

import cv2
import numpy as np
import torch
import ffmpeg  # type: ignore
import matplotlib.pyplot as plt

import analysis
from detection import Detection

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

OUTPUT_INTERVAL = 15


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
        # self.tracking_frames = 20
        self.tracking_frames = 60  # 2seconds
        self.point_radius = 8
        self.output_point_radius = 4

        # 検出プログラム
        self.detection = Detection()
        self.xlimit = [311, 1588]
        self.ylimit = [372, 1080]

        # 動画中のコートの中心の高さ
        self.court_middle = 617

        # 解析結果の出力用変数の初期設定
        self.output_interval = self.tracking_frames
        self.total_movement = [0, 0]
        self.total_movement_track = [[0, 0]]

    def load_video(self, video_path):
        """
        動画を読み込み、フレーム情報を取得
        """
        self.cap = cv2.VideoCapture(video_path)
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"\n#THE NUMBER OF FRAMES: {self.num_frame} | FRAME WIDTH: {self.cap_width} | FRAME HEIGHT: {self.cap_height}\n"
        )

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
        print(f"TRACK SIZE LIST: {track_size_list}")

        return track_size_list

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

        return pos_xy_perspective_transformed

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

    def add_move_gradient(self, court_img, pos_track, tracking_frames):
        """
        移動の傾きを計算
        """
        now_index = tracking_frames - 1
        reflect_frames = 20
        for i in range(2):
            dx = (
                pos_track[now_index][i][0] - pos_track[now_index - reflect_frames][i][0]
            )
            #! playerの検出される順番が変わることがあるから、おかしくなる。辞書型に変えるか、順番が固定されるようにすべき
            dy = -1 * (
                pos_track[now_index][i][1] - pos_track[now_index - reflect_frames][i][1]
            )
            print(f"dx: {dx}  | dy: {dy}")
            print("length", math.sqrt(dx**2 + dy**2))
            if (math.sqrt(dx**2 + dy**2)) > 50:
                if dx < 0:
                    if dy < 0:
                        gradient_rad = -math.pi - math.atan(
                            float(dy) / (float(-dx) + 0.0001)
                        )
                    else:
                        gradient_rad = math.pi - math.atan(
                            float(dy) / (float(-dx) + 0.0001)
                        )
                    gradient_deg = gradient_rad * 180 / math.pi
                else:
                    gradient_rad = math.atan(float(dy) / (float(dx) + 0.0001))
                    gradient_deg = gradient_rad * 180 / math.pi
                print("gradient_deg", gradient_deg)
                now_x, now_y = int(pos_track[now_index][i][0]), int(
                    pos_track[now_index][i][1] - 10
                )
                cv2.putText(
                    court_img,
                    f"{gradient_deg:.1f}deg",
                    (now_x - 10, now_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )
        return court_img

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
        self.load_video(match_video_path)

        if self.is_save_video:
            print("prepare to save video")
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
                print(f"NOW: {(count/self.num_frame*100):.2f}%\n")

            ret, frame = self.cap.read()
            if not ret:
                # 動画の最終フレームを過ぎたらループから抜ける
                print("============ FINISHED ============")
                break

            # 動画用に画像をコピー
            court_for_display = self.court_img.copy()
            img_for_display = frame.copy()

            #  player の検出
            player_foot_pos_xy_list, img_for_display = self.detection.detect_player(
                frame,
                (self.cap_width, self.cap_height),
                img_for_display,
                self.xlimit,
                self.ylimit,
                self.court_middle,
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

            # コート画像に player の位置をプロット
            court_for_display = self.plot_player_pos(
                court_for_display, pos_track, track_size_list
            )
            # コート画像に移動の傾きを表示
            if count >= self.tracking_frames:
                court_for_display = self.add_move_gradient(
                    court_for_display, pos_track, self.tracking_frames
                )

            # 動画とコート画像の合成（左上）
            img_for_display[
                50 : 50 + self.court_size_wh[1], 50 : 50 + self.court_size_wh[0]
            ] = court_for_display

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
                self.total_movement, self.total_movement_track = (
                    analysis.add_total_movement(
                        self.total_movement,
                        self.total_movement_track,
                        pos_track,
                        self.tracking_frames,
                    )
                )

                # player の位置を記録
                self.court_img_for_output = analysis.add_player_position(
                    self.court_img_for_output,
                    self.court_size_wh,
                    pos_xy_perspective_transformed,
                    self.output_point_radius,
                )

        if self.is_save_video:
            self.finish_saving_video()

        # 総移動距離のグラフ
        analysis.make_total_movement_graph(
            self.total_movement_track, output_path=f"./outputs/total_movement{t}.jpg"
        )
        # player の位置プロット
        analysis.make_player_position_plot(
            self.court_img_for_output, output_path=f"./outputs/output{t}.jpg"
        )


if __name__ == "__main__":
    analyzer = MatchAnalyzer()
    analyzer.analyze_match(
        VIDEO_PATH, is_save_video=False, output_path=OUTPUT_VIDEO_PATH
    )
