import os
import cv2
import numpy as np
import re, subprocess, tqdm, time, datetime

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


def video_to_images(vid_file, img_folder=None, verbose=False):
    """
    :param vid_file: str, path to video file (video format tested : .mp4)
    :param img_folder: str, will save images in that folder

    :return: return str path output
    """
    if img_folder is None:
        img_folder = os.path.join('/tmp', os.path.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               '-qscale:v', '2']

    command.append(f'{img_folder}/%06d.jpg')

    if verbose: print(f'Running \"{" ".join(command)}\"')
    x = subprocess.run(command, capture_output=True)
    handle_ffmpeg_errors(x, vid_file)

    if verbose: print(f'Images saved to \"{img_folder}\"')

    return img_folder


def handle_ffmpeg_errors(x, vid_file):
    """
    Raise exceptions if errors found when running ffmpeg

    inputs :
    - output of subprocess.run(command, capture_output=True)
    - video filename
    """
    if x.returncode == 1:
        if x.stderr.decode("utf-8").find("Invalid data found when processing input") != -1:
            raise Exception("Invalid data found when processing input : {}".format(vid_file))
        elif x.stderr.decode("utf-8").find("av_interleaved_write_frame()") != -1:
            raise Exception("input file is truncated/corrupted or no disk left : {}".format(vid_file))
        elif x.stderr.decode("utf-8").find("codec not currently supported in container") != -1:
            raise Exception("codec not currently supported : {}".format(vid_file))
        else:
            print(x.stderr)
            raise Exception("video extraction failed for unknown reason : {}".format(vid_file))


def images_to_video_ffmpeg(img_folder, output_vid_file, name_image="%06d.jpg", fps=30, verbose=True):
    """
    Params :
      - img_folder: str, path to the input image folder
      - output_vid_file: str, path to the video that's gonna be generated
      - name_images : how the images in the folder are named. For exemple, if there are named : 0000001.jpg, it's gonna be
          name_images = "%06d.jpg". If there are named : image_00001.jpg : image_%05d
      - fps (float) : output video fps

    Returns : None
    """

    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-framerate', str(fps), '-i', f'{img_folder}/{name_image}', '-r', str(fps),
        '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    if verbose: print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def get_fps(filename, verbose=False):
    """
    Params :
      - filename: str
      - verbose: bool, if True, print fps

    Returns :
      - Frames Per Second
    """
    def ffprobe_func(filename):
        if verbose: print("get_fps")

        out = subprocess.check_output(
            ["ffprobe", filename, "-v", "0", "-select_streams", "v", "-print_format", "flat", "-show_entries",
             "stream=r_frame_rate"])

        if verbose: print("out : ", out)
        # out may be in the format :
        # streams.stream.0.r_frame_rate="30/1"\n
        # streams.stream.0.r_frame_rate="30000/1001"\nstreams.stream.1.r_frame_rate="90000/1"\n

        parsed_out = re.findall(b'"(\d+(\/\d+)?)"', out)

        if parsed_out:
            possible_fps = np.array([eval(frac[0]) for frac in parsed_out])

            index_close_30 = np.argmin(np.abs(30-possible_fps))
            if verbose: print("fps : ", possible_fps[index_close_30])

            fps_fbx_compatible = EMODE_DICT_KEYS[np.argmin(np.abs(EMODE_DICT_KEYS-possible_fps[index_close_30]))]
            if verbose: print("fps (fbx compatible): ", possible_fps[index_close_30])

            return fps_fbx_compatible
        else:
            raise Exception("Could not read fps in video file : {}".format(filename))

    return wrapper_ffprobe(ffprobe_func, filename)
