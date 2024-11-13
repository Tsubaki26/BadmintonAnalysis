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
    # plt.show()
    # plt.close()


def make_speed_graph(time_track, speed_track, output_path="./speed_output.jpg"):
    y1 = []
    y2 = []
    for ys in speed_track:
        y1.append(ys[0])
        y2.append(ys[1])

    plt.figure(figsize=(20, 5))
    plt.plot(time_track, y1, label="player1")
    plt.plot(time_track, y2, label="player2")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("speed")
    plt.savefig(output_path)
    # plt.show()
    # plt.close()


def make_speed_bar(speed_range, output_path="./speed_bar_output.jpg"):
    speed_classes = []
    speed_values1 = []
    speed_values2 = []

    for i in range(len(speed_range)):
        if i == len(speed_range) - 1:
            speed_str = f"{i}-"
        else:
            speed_str = f"{i}-{i+1}"
        speed_classes.append(speed_str)

    x = np.arange(len(speed_range))
    for key, value in speed_range.items():
        speed_values1.append(value[0])
        speed_values2.append(value[1])

    fig, ax = plt.subplots()
    width = 0.3
    p1 = ax.bar(
        x,
        speed_values1,
        width,
        color="r",
        label="player1",
    )
    p2 = ax.bar(
        x + width,
        speed_values2,
        width,
        color="b",
        label="player2",
    )
    ax.bar_label(p1)
    ax.bar_label(p2)
    ax.legend()
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(tuple(speed_classes))
    ax.set_xlabel("Speed range [m/s]")
    ax.set_ylabel("Counts")
    plt.savefig(output_path)
    # plt.show()
    # plt.close()


def make_parameters_text(parameters, output_path="./parameters.txt"):
    """
    Arguments:
        ・辞書型のパラメータリスト
    """
    text_file = open(output_path, "w")
    for param_name, value in parameters.items():
        text_file.write(f"{param_name}: {value}")
    text_file.close()


def make_player_position_plot(court_img_for_output, output_path="./court_output.jpg"):
    # cv2.imshow("court", court_img_for_output)
    # cv2.waitKey()
    cv2.imwrite(output_path, court_img_for_output)
