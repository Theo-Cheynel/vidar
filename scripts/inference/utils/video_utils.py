import os
import cv2
import numpy as np
import re, subprocess, tqdm, time, datetime

from . import text_utils, os_utils
from ..kinetix_fbx.fbx_utils import EMODE_DICT_KEYS

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


def images_to_video_cv2(img_folder, output_vid_file, image_format=".jpg", fps=30):
    """
    Params :
      - img_folder: str, path to the input image folder
      - output_vid_file: str, path to the video that's gonna be generated
      - image_format: only used for cv2 method, format of image
      - fps: only used for cv2 method, output video fps

    Returns : None
    """

    images = text_utils.natural_sort_list([img for img in os.listdir(img_folder) if img.endswith(image_format)])
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_vid_file, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    os.makedirs(img_folder, exist_ok=True)


def trim_crop_video(input_path, output_path, trim=None, crop=None, output_images=True, verbose=False):
    """
    Args:
      - input_path(str): path of the input video
      - trim:            if not None, is a list of 2 elements : trim_start (seconds) and trim_end (seconds, excluded)
      - crop:            if not None, is a list of 4 elements : x, y, width, height
      - output_path:     path of the output video or folder of images, cannot be the same as input_path
      - output_images (bool): if True, will output a folder of images, else a video
      - verbose(bool):

    Returns: w, h
    """
    assert output_path != input_path, "output_path cannot be the same as input_path"
    w, h = get_width_and_height(input_path)
    video_length = get_length(input_path)

    if output_images:
        command = ['ffmpeg', '-i', input_path, '-f', 'image2', '-v', 'error', '-qscale:v', '2']
        os.makedirs(output_path, exist_ok=True)
    else:
        command = ['ffmpeg', '-i', input_path]

    if trim is not None:
        trim[0] = float(trim[0])
        trim[1] = float(trim[1])
        if trim[0] >= 0 and trim[1] <= video_length:
            # objective here is to have a command like : -ss 00:01:00 -i input.mp4 -to 00:02:35.25
            # with -ss the time to start in the video, and -to the end time (HH:MM:SS.CC)

            trim_start = datetime.datetime.fromtimestamp(trim[0]).strftime('%H:%M:%S.%f')
            trim_end   = datetime.datetime.fromtimestamp(trim[1]).strftime('%H:%M:%S.%f')

            command.append('-ss')
            command.append(trim_start)
            command.append('-to')
            command.append(trim_end)
        else:
            trim = None
            print("WARNING: Silencing error in trim!")

    if crop is not None:
        crop[2:] = [min(crop[2], w-crop[0]), min(crop[3], h-crop[1])]

        if crop[2] > 0 and crop[3] > 0:
            command.append('-filter:v')
            command.append('crop=' + str(crop[2]) + ':' + str(crop[3]) + ':' + str(crop[0]) + ':' + str(crop[1]) + ':exact=1')
        else:
            crop = None
            print("WARNING: Silencing error in crop!")

    if output_images:
        command.append(f'{output_path}/%06d.jpg')
    else:
        command.append(output_path)
        command.append("-y")

    if verbose: print(f'Running \"{" ".join(command)}\"')
    x = subprocess.run(command, capture_output=True)
    handle_ffmpeg_errors(x, input_path)

    if crop is not None:
        return crop[2:]
    else:
        return w, h


def reencode_video(filename, new_filename):
    """
    Copy video to reencode it (and use it with ffprobe)

    Params :
      - filename    : video path
      - new_filename: new video path
    """
    subprocess.run(["ffmpeg", "-i", filename,
                             "-vcodec", "copy",
                             "-acodec", "copy",
                             new_filename],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)


def wrapper_ffprobe(ffprobe_func, filename, is_except=False):
    """
    Params:
      - ffprobe_func: function containing ffprobe
      - filename    : file path
      - is_except   : (bool) has tried to reencode the video

    Returns:
      - return of the ffprobe_func function
    """
    try:
        return ffprobe_func(filename)

    except:
        assert is_except==False, "ERROR: Something is went wrong with the codec and ffprobe!"

        filename_wo_ext, ext = os.path.splitext(filename)
        new_filename = filename_wo_ext+"_temp"+ext

        reencode_video(filename, new_filename)
        os.replace(new_filename, filename)

        return wrapper_ffprobe(ffprobe_func, filename, is_except=True)


def get_length(filename):
    """
    :param filename:
    :return: int time of a video in sec
    """
    def ffprobe_func(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        return float(result.stdout)

    return wrapper_ffprobe(ffprobe_func, filename)


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


def get_width_and_height(filename, verbose=False):
    """
    :param filename:
    :return: width, height in pixels
    """
    def ffprobe_func(filename):
        result = subprocess.run(["ffprobe", "-v", "error",
                                 "-select_streams", "v:0",
                                 "-show_entries", "stream=width,height:side_data=rotation",
                                 "-of", "default=nw=1:nk=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        if verbose: print("result.stdout", result.stdout.strip())
        tab = result.stdout.strip().split(b"\n")
        if verbose: print("tab", tab)

        # We check if the video has a rotation argument in metadata to retrieve correct width and height.
        if len(tab) == 3:
            rotation = int(tab[2]) % 180
            if rotation == 90:
                # ffmpeg swaps width and height.
                if verbose: print(f"width={int(tab[1])}", f"height={int(tab[0])}")
                return int(tab[1]), int(tab[0])

            elif rotation == 0:
                if verbose: print(f"width={int(tab[0])}", f"height={int(tab[1])}")
                return int(tab[0]), int(tab[1])

            else:
                raise Exception("Warning : ffmpeg have an unknown rotation for this video")

        if verbose: print(f"width={int(tab[0])}", f"height={int(tab[1])}")
        return int(tab[0]), int(tab[1])

    return wrapper_ffprobe(ffprobe_func, filename)


def get_n_frames(filename):
    """
    :param filename:
    :return: number of frames
    """
    def ffprobe_func(filename):
        result = subprocess.run(["ffprobe", "-v", "error",
                                 "-select_streams", "v:0",
                                 "-show_entries", "stream=nb_frames",
                                 "-of", "default=nokey=1:noprint_wrappers=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        return int(result.stdout)

    return wrapper_ffprobe(ffprobe_func, filename)


def mov2mp4(path_mov, path_mp4):
    """
    ffmpeg -i movie.mov -vcodec copy -acodec copy out.mp4
    :param path_mov:
    :param path_mp4:
    :return:
    """

    command = "ffmpeg -i" + path_mov + "-vcodec copy -acodec copy" + path_mp4
    subprocess.call(command)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    """
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates (s, tx, ty)
    :param bbox (ndarray, shape=(4,)): bbox coordinates (cx, cy, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return: weak perspective camera in original img coordinates
    """
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    hw, hh = img_width / 2, img_height / 2

    sx = cam[:, 0] * h / img_width
    sy = cam[:, 0] * h / img_height

    tx = (cx / hw - 1) / sx + cam[:, 1]
    ty = (cy / hh - 1) / sy + cam[:, 2]

    return np.stack([sx, sy, tx, ty]).T


def trim_video_frames(input_path, output_path, frame_begin, frame_end):
    """
    trim the video according to frames.
    (keep in mind that n_frames_output = frame_end - frame_begin + 1)

    Params :
      - input_path  : (str) path to the video to be trimmed
      - output_path : (str) path to save the output video. Cannot be the same as input_path
      - frame_begin : (int) first frame to be in the output trimmed video
      - frame_end   : (int) last frame to be in the output trimmed video

    Returns : None
    """
    assert frame_end > frame_begin
    assert frame_end <= get_n_frames(filename=input_path)

    fps = get_fps(filename=input_path)

    t0 = frame_begin / fps
    tf = frame_end / fps

    trim_crop_video(input_path=input_path, output_path=output_path, trim=[t0, tf], crop=None, output_images=False, verbose=False)
