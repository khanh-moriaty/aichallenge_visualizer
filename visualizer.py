import cv2
import os
import json
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from math import sqrt
from multiprocessing import Pool
from cfg.config import *

########################################################################
# Chú thích về video_info:
#
# video_info[...][0] = ROI (Region-of-Interest): list of coordinate pairs
# video_info[...][1] = MOI (Motion-of-Interest): list of (list of coordinate pairs)
# video_info[...][2] = Object cv2.VideoWriter để ghi video output
# video_info[...][3] = Số lượng frame của video.
# Bằng "-1" nếu chưa load content của video.
# video_info[...][4] = Object cv2.VideoCapture
# video_info[...][5][i][j] = List chứa các (class_id, x, y) được đếm tại frame thứ i
# đi theo moi thứ j. (x == y == -1) nếu input không hỗ trợ.
#
#########################################################################


def loadVideoInfo(info_dir):
    video_info = {}
    dir = os.listdir(info_dir)
    dir = [fi for fi in dir if os.path.splitext(fi)[1] == '.json']
    dir.sort()
    for fi in dir:
        bfn, _ = os.path.splitext(fi)

        video_info[bfn] = [None, [], None, -1, [], []]

        info_path = os.path.join(info_dir, fi)
        info_js = json.load(open(info_path, 'r'))
        shapes = info_js['shapes']

        video_info[bfn][1] = [[] for _ in range(len(shapes))]
        for shape in shapes:
            pts = shape['points']
            pts = np.array([[int(x) for x in y] for y in pts])
            # Khởi tạo ROI:
            if shape['label'] == "zone":
                video_info[bfn][0] = pts

            # Thêm các MOI:
            if shape['label'][:-2] == "direction":
                moi_id = int(shape['label'][-2:])
                video_info[bfn][1][moi_id] = pts
    return video_info


def loadVideoContent(video_info, video_dir, output_dir, video_name):
    # Check if video_info exists
    if not video_name in video_info:
        print("MOI and ROI for video", video_name, "does not exist!")
        return

    # Check if content already loaded:
    if video_info[video_name][3] > -1:
        return

    video_path = os.path.join(video_dir, video_name+'.mp4')
    output_path = os.path.join(output_dir, video_name+'.mp4')
    vid = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, codec, vid_fps, (vid_width, vid_height))
    frame_count = 0
    # video_info[video_name][4] = np.ndarray(shape=(1200, vid_height, vid_width, 3), dtype=np.uint8)
    while True:
        _, img = vid.read()
        if img is None:
            break
        # print(img.shape)
        # print(video_info[video_name][4].shape)
        # video_info[video_name][4][frame_count] = img
        frame_count += 1

    vid.release()
    video_info[video_name][2] = out
    video_info[video_name][3] = frame_count
    # video_info[video_name][4] = np.array(video_info[video_name][4])
    video_info[video_name][4] = cv2.VideoCapture(video_path)
    moi_count = len(video_info[video_name][1])
    video_info[video_name][5] = [[[] for _ in range(moi_count)]
                                 for __ in range(frame_count)]


def calc_median(arr):
    return np.median(arr, axis=0)


def findOptimalProcess(n):
    a = [x for x in range(1, int(sqrt(n))) if n % x == 0]
    return a[-1]


def visualize(sub_file_path, video_dir, info_dir, output_dir, testing=False):
    t = time.time()
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sub_file = open(sub_file_path, 'r')
    video_info = loadVideoInfo(info_dir)
    for line in sub_file.read().splitlines():
        line_content = line.split(' ')
        video_name = line_content[0]
        loadVideoContent(video_info, video_dir, output_dir, video_name)
        frame_id = int(line_content[1]) - 1
        moi_id = int(line_content[2])
        class_id = int(line_content[3])
        x = y = -1
        if len(line_content) > 4:  # If testing data
            x = int(line_content[4])
            y = int(line_content[5])

        video_info[video_name][5][frame_id][moi_id].append((class_id, (x, y)))

    for video_name in video_info:
        video = video_info[video_name]
        frame_count = video[3]
        if frame_count == -1:
            continue
        print('frame count:', frame_count)

        moi_count = len(video[1])
        track_list = []

        print(time.time() - t)
        t = time.time()

        vid = video[4]
        vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
        print('fps:', vid_fps)
        width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # img_mask = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        # cv2.fillPoly(img_mask, [video[0]], ROI_COLOR)
        # cv2.polylines(img, [video[0]], isClosed=True, color=ROI_COLOR_BGR, thickness=2)

        # for moi_id, moi in enumerate(video[1]):
        #     if moi_id == 0:
        #         continue
        #     cv2.polylines(img_mask, [moi[:-1]], isClosed=False,
        #                   color=getColorMOI_BGRA(moi_id), thickness=3)
        #     cv2.arrowedLine(img_mask, tuple(
        #         moi[-2]), tuple(moi[-1]), color=getColorMOI_BGRA(moi_id), thickness=3)

        # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGBA2BGRA)

        # Normalize alpha channel to [0; 1]
        # alpha_channel = img_mask[:, :, 3].astype(float) / 255.0
        # alpha_channel = np.repeat(alpha_channel[:, :, np.newaxis], 3, axis=2)

        curr_count = np.zeros(shape=(moi_count, 5), dtype=int)

        segment_start = list(range(0, frame_count, FRAME_PER_SEGMENT))
        segment_end = segment_start[1:].append(frame_count)
        num_segment = len(segment_start)
        flash_list = []
        for i in tqdm(range(num_segment)):
            img = np.ndarray(shape=(FRAME_PER_SEGMENT, height, width, 3),
                             dtype=np.uint8)
            for j in range(FRAME_PER_SEGMENT):
                _, frame = vid.read()
                img[j] = frame
            # img = img * (1-alpha_channel) + img_mask[:, :, :3] * alpha_channel
            # img = img.astype(np.uint8)

            for j in range(FRAME_PER_SEGMENT):
                cv2.polylines(img[j], [video[0]], isClosed=True, color=ROI_COLOR_BGR, thickness=4)

                for moi_id, moi in enumerate(video[1]):
                    if moi_id == 0:
                        continue
                    cv2.polylines(img[j], [moi[:-1]], isClosed=False,
                                color=getColorMOI_BGR(moi_id), thickness=3)
                    cv2.arrowedLine(img[j], tuple(
                        moi[-2]), tuple(moi[-1]), color=getColorMOI_BGR(moi_id), thickness=3)
                    
                for moi_id in range(1, moi_count):
                    obj_list = video[5][i*FRAME_PER_SEGMENT + j][moi_id]
                    for obj in obj_list:
                        curr_count[moi_id][obj[0]] += 1
                        if obj[1][0] > -1:
                            flash_list.append((i*FRAME_PER_SEGMENT+j, obj))

                    count_str = ' '.join([str(x) for x in curr_count[moi_id][1:]])
                    moi = video[1][moi_id]
                    cv2.putText(img[j], count_str, (moi[-1][0] + 50, moi[-1][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=getColorMOI_BGR(moi_id), thickness=2)

                # Remove frames older than 0.25s
                flash_list = [flash for flash in flash_list if (
                    i*FRAME_PER_SEGMENT+j-flash[0] < (vid_fps * 0.25))]
                for flash in flash_list:
                    radius = (15 * flash[1][1][1] // height)
                    cv2.circle(img[j], flash[1][1], radius=radius,
                               color=(0, 0, 255), thickness=-1)
                    cv2.putText(img[j], str(flash[1][0]), (flash[1][1][0]-radius, flash[1][1][1]-radius),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=flash[1][1][1] / height, color=(0, 255, 255), thickness=2)
                frame_str = "frame_id: " + str(i*FRAME_PER_SEGMENT + j + 1)
                cv2.putText(img[j], frame_str, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)

            [video[2].write(frame) for frame in img]

        # img = video[4][0]
        # video[4][:][:,:,:] = video[4][:][:,:,:] * (1-alpha) + moi_mask[:,:,:3] * alpha
        # video[4] = video[4].astype(np.uint8)

        # for i in range(frame_count):
            # img = video[4][i]
            # height, width = img.shape[:2]

            # alpha = color_fill[3] / 255.0
            # img = img.astype(float)
            # img[:,:,:] = img[:,:,:] * (1-alpha) + moi_mask[:,:,:3] * alpha
            # img = img.astype(np.uint8)

            # for j in range(moi_count):
            #     for obj in video_info[video_name][5][i][j]:
            #         track_list.append((i+1,j,obj))

        print(time.time() - t)
        t = time.time()
        # for i in range(frame_count):
        #     video[2].write(video[4][i])

        video[2].release()

        # frame_stack = frame_count // 1
        # CPU_PROCESS = findOptimalProcess(frame_stack)
        # print('CPU_PROCESS', CPU_PROCESS)
        # print(time.time() - t)
        # t = time.time()
        # pool = Pool(CPU_PROCESS)
        # segments = video[4][:frame_stack]
        # segments = np.split(segments, CPU_PROCESS)
        # img = pool.map(calc_median, segments)

        # # img = video[4][::findOptimalProcess(frame_count)]
        # img = np.median(img, axis=0)
        # frame = video[4][100]
        # sub = np.clip(frame.astype(int) - img.astype(int), 0, 255).astype(np.uint8)
        # out_img_path = os.path.join(output_dir, 'test.jpg')
        # out_frame_path = os.path.join(output_dir, 'frame.jpg')
        # out_sub_path = os.path.join(output_dir, 'sub.jpg')
        # cv2.imwrite(out_img_path, img)
        # cv2.imwrite(out_frame_path, frame)
        # cv2.imwrite(out_sub_path, img-frame)

    print(time.time() - t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script supports visualizing submission file on videos')
    parser.add_argument('--sub_file_path', type=str, default="submission/submission.txt",
                        help='path to submission file.')
    parser.add_argument('--video_dir', type=str, default="videos/",
                        help='path to videos directory.')
    parser.add_argument('--info_dir', type=str, default="zones-movement_paths/",
                        help='path to videos info (ROI, MOI) *.json directory.')
    parser.add_argument('--output_dir', type=str, default="output/",
                        help='path to output videos directory.')
    parser.add_argument('--testing', type=bool, default=False,
                        help='if this is for testing purpose.')
    args = parser.parse_args()

    visualize(args.sub_file_path, args.video_dir,
              args.info_dir, args.output_dir, args.testing)
