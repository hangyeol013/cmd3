import subprocess
import argparse
from pathlib import Path
import re
from decimal import Decimal
import numpy as np
import os
from joblib import Parallel, delayed




def clip_process(video_file_path):

    dst_clip_path = Path(str(video_file_path).replace('video', 'video_clips_096').replace('.mkv', ''))
    dst_clip_path.mkdir(exist_ok=True)

    ffmpeg_clips = ['ffmpeg', '-i', str(video_file_path), '-codec:v', 'libx264', '-map', '0', '-force_key_frames', 'expr:gte(t,n_forced*1)', '-segment_time', '1', '-f', 'segment', '-reset_timestamps', '1', '{}/clip_%03d.mkv'.format(dst_clip_path)]
    subprocess.run(ffmpeg_clips)

    # ffprobe_cmd_duration = ('ffprobe -v error -of default=noprint_wrappers=1:nokey=1 -show_entries format=duration').split()
    # ffprobe_cmd_duration.append(str(video_file_path))

    # p2 = subprocess.run(ffprobe_cmd_duration, capture_output=True)
    # duration = p2.stdout.decode('utf-8').splitlines()[0]

    # duration = float(duration)
    # for t in range(int(duration/4)):
        # ffmpeg_clips = ['ffmpeg', '-i', str(video_file_path), '-ss', '{}ms'.format(t*960*8), '-t', '{}ms'.format(960*8), '-c', 'copy', '{}/clip_{:03}.mkv'.format(dst_clip_path, t)]
    # subprocess.run(ffmpeg_clips)


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    print(res)
    if len(res) < 3:
        return
    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]

    ffprobe_cmd_duration = ('ffprobe -v error -of default=noprint_wrappers=1:nokey=1 -show_entries format=duration').split()
    ffprobe_cmd_duration.append(str(video_file_path))

    p2 = subprocess.run(ffprobe_cmd_duration, capture_output=True)
    duration = p2.stdout.decode('utf-8').splitlines()[0]

    duration = float(duration)
    n_frames = int(frame_rate * duration)

    name = str(video_file_path).split('201')[1].replace('.mkv', '')[2:]
    dst_dir_path = dst_root_path / name
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    print('n_exist_frames: ', n_exist_frames)
    print('n_frames: ', n_frames)
    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)
    
    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%03d.jpg'.format(dst_dir_path)]
    subprocess.run(ffmpeg_cmd)
    print('\n')


def class_process(class_dir_path, dst_root_path, clip_root_path, mode, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    dst_clip_path = clip_root_path / class_dir_path.name
    dst_clip_path.mkdir(exist_ok=True)

    if mode == "clip":
        for video_file_path in sorted(class_dir_path.iterdir()):
            clip_process(video_file_path)
    else:
        for video_file_path in sorted(class_dir_path.iterdir()):
            for clip_path in sorted(video_file_path.iterdir()):
                video_process(clip_path, dst_class_path, ext, fps, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir_path', default='./dataset/video', type=Path, help='Directory path of videos')
    parser.add_argument(
        '--clip_path', default='./dataset/video_clips_096', type=Path, help='Directory path of videos clips')
    parser.add_argument(
        '--dst_path',
        default='./dataset/video_jpg_096',
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    ext = '.mkv'

    class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
    mode = 'clip'
    status_list = Parallel(
        n_jobs=args.n_jobs,
        backend='threading')(delayed(class_process)(
            class_dir_path, args.dst_path, args.clip_path, mode, ext, args.fps, args.size)
                                for class_dir_path in class_dir_paths)
    clip_dir_paths = [x for x in sorted(args.clip_path.iterdir())]
    mode = 'jpg'
    status_list = Parallel(
        n_jobs=args.n_jobs,
        backend='threading')(delayed(class_process)(
            clip_dir_path, args.dst_path, args.clip_path, mode, ext, args.fps, args.size)
                                for clip_dir_path in clip_dir_paths)