import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_total_movement(
    total_movement, total_movement_track, pos_track, tracking_frames
):
    # 総移動距離の加算
    move_amount_p1 = math.sqrt(
        (pos_track[0][0][0] - pos_track[tracking_frames - 1][0][0]) ** 2
        + (pos_track[0][0][1] - pos_track[tracking_frames - 1][0][1]) ** 2
    )
    move_amount_p2 = math.sqrt(
        (pos_track[0][1][0] - pos_track[tracking_frames - 1][1][0]) ** 2
        + (pos_track[0][1][1] - pos_track[tracking_frames - 1][1][1]) ** 2
    )
    total_movement[0] += move_amount_p1
    total_movement[1] += move_amount_p2
    total_movement_track.append(total_movement.copy())

    return total_movement, total_movement_track


def add_player_position(
    court_img_for_output,
    court_size_wh,
    pos_xy_perspective_transformed,
    output_point_radius,
):
    circle_add = np.uint8(np.zeros((court_size_wh[1], court_size_wh[0], 3)))
    for pos in pos_xy_perspective_transformed:
        x, y, _ = pos
        x, y = int(x[0]), int(y[0])
        cv2.circle(
            circle_add,
            (x, y),
            output_point_radius,
            color=(50, 50, 50),
            thickness=-1,
        )
        court_img_for_output = cv2.add(court_img_for_output, circle_add)
    return court_img_for_output


def make_total_movement_graph(
    total_movement_track, output_path="./total_movement_output.jpg"
):
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
    plt.savefig(output_path)
    plt.show()
    plt.close()


def make_player_position_plot(court_img_for_output, output_path="./court_output.jpg"):
    # cv2.imshow("court", court_img_for_output)
    # cv2.waitKey()
    cv2.imwrite(output_path, court_img_for_output)
