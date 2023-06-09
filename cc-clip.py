import cv2
import numpy as np
import threading
import concurrent.futures
import asyncio
from tqdm import tqdm
import os
import argparse
import shutil
from natsort import natsorted
from PIL import ImageFont, ImageDraw, Image

parser = argparse.ArgumentParser(description='Create clips of events using recordings from the hit game CrossCode') # noqa
parser.add_argument('-t', '--type',         type=str,    required=True, choices=["ko", "death"],    help='Type of event to search for') # noqa
parser.add_argument('files',                type=argparse.FileType('r'), nargs='+',                 help='The files to process') # noqa
parser.add_argument('-o', '--output',       type=str, required=True,                                help='Output file destination path') # noqa
parser.add_argument('-A', '--after',        type=int, default=150,                                  help='Number of threads to use when searching for deaths (default: 4)') # noqa
parser.add_argument('-B', '--before',       type=int, default=500,                                  help='Clip time before event in ms (default: 150)') # noqa
parser.add_argument('--grid',               type=int, default=1,                                    help='Grid size') # noqa
parser.add_argument('--threads',            type=int, default=4,                                    help='Clip time after event in ms (default: 500)') # noqa
parser.add_argument('--no-delete-clips',    action='store_false', default=True,                     help='Dont delete the clips at the end') # noqa
parser.add_argument('--no-generate-clips',  action='store_false', default=True,                     help='Skip clip generation (use already existing ones)') # noqa
parser.add_argument('--no-tile',            action='store_false', default=True,                     help='Skip tile video generation (use already exisiting ones)') # noqa
parser.add_argument('--intro-duration',     type=float, default=2,                                  help='Duration of the intro. 0 to disable (default: 2)') # noqa

args = parser.parse_args()
search_type = args.type
files = args.files
thread_count = args.threads
out_filepath = args.output
cleanup_clips = args.no_delete_clips
do_generate_clips = args.no_generate_clips
gen_video_before = args.before
gen_video_after = args.after
do_tile = args.no_tile
grid_size = args.grid
intro_duration = args.intro_duration


if not out_filepath.endswith(".mkv"):
    print("Output file has to end with .mkv")
    exit()
else:
    out_filename = os.path.basename(out_filepath)
    out_basename = out_filename[:-4]


for i in range(len(files)):
    files[i] = files[i].name
    if not files[i].endswith(".mkv"):
        print("Only .mkv files are accepted.")

if search_type == "ko":
    _template = cv2.imread("ko.png", cv2.IMREAD_GRAYSCALE)
    template_cuts = [
            [910, 484],
            ]
    threshold = 0.95
elif search_type == "death":
    template = cv2.imread("death.png", cv2.IMREAD_GRAYSCALE)
    template_cuts = [
            [131, 12],
            [297, 90],
            ]
    threshold = 0.95

template_w, template_h = template.shape
for x in range(len(template_cuts)):
    template_cuts[x].append(template_cuts[x][0] + template_h + 1)
    template_cuts[x].append(template_cuts[x][1] + template_w + 1)

frame_match_array_lock = threading.Lock()
frame_count_lock = threading.Lock()
frame_lock = threading.Lock()

work_dir = os.getcwd()
clips_dir = f"{work_dir}/clips"
if not os.path.exists(clips_dir):
    os.makedirs(clips_dir)

if not os.path.exists(f"{clips_dir}/grid"):
    os.makedirs(f"{clips_dir}/grid")

temp_dir = f"{work_dir}/tmp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


max_frame_jump = 100

total_videos_length = 0
total_clips = 0


def process_file(video):
    print(f"Processing file {video}")
    global cap, video_frame_count, video_fps, video_width, video_height, start_frame, end_frame, total_videos_length # noqa
    # Load the video file
    cap = cv2.VideoCapture(video)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_videos_length = total_videos_length + video_frame_count / video_fps

    start_frame = 0
    end_frame = -1

    if video == "/mnt/nvme/2023-05-10_15-31-44.mkv":
        beg_frames = [4921, 6100, 10669, 15979, 20048, 21119, 22439, 23485, 24557, 26353, 26987, 30177, 33496, 35333, 38442, 40647, 44236, 44852, 48533, 51639, 53523, 57545, 62014, 65637, 69571, 73528, 76983, 81729, 85428, 89044, 93895, 94901, 102191, 105385, 108995, 119393] # noqa
    elif video == "/mnt/nvme/2023-05-10_16-25-09.mkv":
        beg_frames = [4521, 8152, 11638, 14533, 20316, 23720, 27468, 32350, 39232, 42638, 46765, 47409, 49709, 56499, 69253, 73954, 81490, 92110, 93241, 100922, 102548, 103224, 106516, 117062, 122547, 126149, 136252, 141998, 150454, 159326, 162859, 166348, 170123] # noqa
    elif video == "/mnt/nvme/2023-05-10_17-24-38.mkv":
        beg_frames = [2402, 6723, 14474, 18377, 22555, 25873, 29549, 33152, 41947, 50993, 54552, 58006] # noqa
    elif video == "/mnt/nvme/2023-05-10_23-40-18.mkv":
        beg_frames = [3907, 9031, 14044, 20933, 23999, 32276, 38655, 46938, 49310, 56988, 59211, 61611, 64720, 67706, 79421, 81277, 93254, 97083, 110863, 122856, 135613] # noqa
    else:
        search_for_frames()
        beg_frames = get_unique_moments(frames_with_matches)

    # for value in frames_with_matches:
    #     if value > 20000 and value < 23000:
    #         print(value, end=' ')
    # print()

    # for frame in range(57500, 62020):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #     ret, img = cap.read()
    #     print(frame)
    #     show_image(img, 1)

    # for frame in beg_frames:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #     ret, img = cap.read()
    #     print(frame)
    #     show_image(img, 0)
    if do_generate_clips:
        generate_clips(beg_frames, video)


def show_image(img, delay):
    cv2.imshow("Image", img)
    if delay != 0:
        cv2.waitKey(delay)
    else:
        while True:
            key = cv2.waitKey(0)
            if key & 0xFF == ord('n'):
                break
            if key & 0xFF == ord('q'):
                exit()


async def check_frame(frame_number, frame):
    for cut_index in range(len(template_cuts)):
        if check_frame_with_template(frame_number, frame, cut_index):
            break
    pbar.update()


def check_frame_with_template(frame_number, frame, cut_index):
    cut_frame = frame[template_cuts[cut_index][1]:template_cuts[cut_index][3],
                      template_cuts[cut_index][0]:template_cuts[cut_index][2]]
    # show_image(cut_frame, 0)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)

    # Search for the template in the frame
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    binary = (result > threshold).astype(np.uint8) * 255

    # Find the locations of matches above the threshold
    locations = cv2.findNonZero(binary)

    # If there are any matches, add this frame to the list
    if locations is not None:
        add_frame_to_array(frame_number, frame, locations)
        return True

    return False


def add_frame_to_array(frame_number, frame, locations):
    with frame_match_array_lock:
        frames_with_matches.append(frame_number)


def process_frame(thread_idx):
    global frame_number
    while True:
        with frame_lock:
            ret, frame = cap.read()
            frame_number = frame_number + 1

        if not ret or frame_number > end_frame:
            break

        asyncio.run(check_frame(frame_number, frame))


def search_for_frames():
    print("Searching for matching frames")

    global end_frame, frames_with_matches, pbar, frame_number

    frames_with_matches = []
    if end_frame != -1 and video_frame_count > end_frame:
        frame_total = end_frame
    else:
        end_frame = video_frame_count
        frame_total = video_frame_count

    frame_total = frame_total - start_frame

    pbar = tqdm(total=frame_total, unit=' frames')

    frame_number = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    threads = []
    for i in range(thread_count):
        t = threading.Thread(target=process_frame, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    pbar.close()

    frames_with_matches.sort()
    print(frames_with_matches)
    exit()

    print(f"\nFound {len(frames_with_matches)} frames with matches\n")
    # print(frames_with_matches)
    # print("\n")


def get_unique_moments(frames):
    print("Searching for unique moments")

    beg_frames = []
    last_frame = -100000

    for frame in tqdm(frames, unit=' frames'):
        if last_frame + max_frame_jump < frame:
            beg_frames.append(frame)

        last_frame = frame

    print(f"\nFound {len(beg_frames)} frames: {beg_frames}\n")
    return beg_frames


def ffmpeg_combine(arr, filename, copy=True):
    list_file_path = f"{temp_dir}/list.txt"
    list_file = open(list_file_path, 'w')
    for f in arr:
        list_file.write(f"file {f}\n")
    list_file.close()

    copy_flag = ""
    if copy:
        copy_flag = "-c copy"
    cmd = f"ffmpeg -f concat -safe 0 -i {list_file_path} {copy_flag} -loglevel error -fflags +genpts -y {filename}" # noqa
    os.system(cmd)
    os.remove(list_file_path)


def generate_clips(beg_frames, video):
    print("Generating clips")

    global pbar
    pbar = tqdm(total=len(beg_frames), unit='clip')
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor: # noqa
        futures = [executor.submit(generate_video_of_frame, video, frame_number) for frame_number in beg_frames] # noqa

        concurrent.futures.wait(futures)
        pbar.close()

    print("\n")


def generate_video_of_frame(video, frame_number):
    filename = f"{file_index}_{frame_number}.mkv"

    video_file = f"{temp_dir}/video-{filename}"
    audio_file = f"{temp_dir}/audio-{filename}"

    start_frame = frame_number - int(gen_video_before*video_fps/1000)
    end_frame = frame_number + int(gen_video_after*video_fps/1000)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, video_fps, (video_width, video_height), True) # noqa

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

    # Extract audio
    start_time = str(max(0, frame_number / video_fps - gen_video_before/1000)) # noqa
    end_time = str((gen_video_before+gen_video_after)/1000)
    cmd = f"ffmpeg -i {video} -ss {start_time} -t {end_time} -loglevel error -vn -acodec copy {audio_file}" # noqa
    os.system(cmd)

    # Combine video and audio
    cmd = f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -map 0:v:0 -map 1:a:0 -loglevel error -y clips/{filename}" # noqa
    os.system(cmd)

    # Clean up
    os.remove(video_file)
    os.remove(audio_file)

    with frame_lock:
        pbar.update()

    pass


def combine_clips():
    clips = natsorted([f for f in os.listdir(clips_dir) if f.endswith(".mkv")]) # noqa
    global total_clips
    global recode_video
    total_clips = len(clips)

    if grid_size < 2:
        print("Combining clips")
        ffmpeg_combine([f"{clips_dir}/{f}" for f in clips], f"{out_filepath}", copy=True) # noqa
    else:
        print("Tiling clips")
        clips_per_screen = grid_size * grid_size
        tiled_clips = []

        pbar_len = int(len(clips)/clips_per_screen)*clips_per_screen
        if pbar_len == 0:
            pbar_len = len(clips)
            clips_per_screen = pbar_len

        pbar = tqdm(total=pbar_len, unit=' frames') # noqa

        while len(clips) >= clips_per_screen:
            selected_clips = []
            for i in range(clips_per_screen):
                selected_clips.append(f"{clips_dir}/{clips.pop(0)}")

            temp_file_out = f"{clips_dir}/grid/{len(tiled_clips)}_{out_filename}" # noqa

            # epic guy from https://stackoverflow.com/questions/63993922/how-to-merge-several-videos-in-a-grid-in-one-video # noqa
            input_videos = ""
            input_setpts = "nullsrc=size={}x{} [base];".format(video_width, video_height) # noqa
            input_overlays = "[base][video0] overlay=shortest=1 [tmp0];"
            grid_width = grid_size
            grid_height = grid_size
            audio_merge = ""
            audio_volume = ""

            for index, path_video in enumerate(selected_clips):
                input_videos += f" -i {path_video}"
                input_setpts += f"[{index}:v] setpts=PTS-STARTPTS, scale={video_width//grid_width}x{video_height//grid_height} [video{index}];" # noqa
                if index > 0 and index < clips_per_screen - 1:
                    input_overlays += f"[tmp{index-1}][video{index}] overlay=shortest=1:x={video_width//grid_width * (index%grid_width)}:y={video_height//grid_height * (index//grid_height)} [tmp{index}];" # noqa
                if index == clips_per_screen - 1:
                    input_overlays += f"[tmp{index-1}][video{index}] overlay=shortest=1:x={video_width//grid_width * (index%grid_width)}:y={video_height//grid_height * (index//grid_width)}" # noqa
                audio_volume += f"[{index}:a]volume={grid_size}[a{index}];"
                audio_merge += f"[a{index}]"

            cmd = f"ffmpeg{input_videos} -filter_complex \"{input_setpts}{input_overlays},premultiply=inplace=1;{audio_volume}{audio_merge}amix=inputs={clips_per_screen}[audio_out]\" -map '[audio_out]'  -loglevel error -fflags +genpts -y {temp_file_out}" # noqaa

            if do_tile:
                os.system(cmd)
            pbar.update(clips_per_screen)

            tiled_clips.append(temp_file_out)

        pbar.close()

        if do_tile:
            print("Combining tiled clips")
            ffmpeg_combine(tiled_clips, f"{clips_dir}/grid/combine.mkv")

            if len(clips) > 0:
                print("Re-coding regular clips to .mkv")
                pbar = tqdm(total=len(clips), unit=' frames')
                for f in clips:
                    new_file = f"{clips_dir}/grid/{f}.mkv"
                    cmd = f"ffmpeg -i {clips_dir}/{f} -loglevel error -y {new_file}" # noqa
                    os.system(cmd)
                    tiled_clips.append(new_file)
                    pbar.update()

                pbar.close()

            print("Combining tiled clips with regular clips")
            ffmpeg_combine(tiled_clips, out_filepath)

    if cleanup_clips:
        print("Removing temporary clips")
        shutil.rmtree(clips_dir)

    print("Re-encoding the video")
    temp_file = f"{temp_dir}/{out_filename}"
    shutil.copy(out_filepath, temp_file)
    cmd = f"ffmpeg -i {temp_file} -fflags +genpts -y -loglevel error {out_filepath}" # noqa
    os.system(cmd)
    os.remove(temp_file)


def add_info_clip():
    minutes, seconds = divmod(total_videos_length, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    font_path = "DejaVuSans.ttf"

    img = Image.new('RGB', (video_width, video_height), color='black')

    text = [
            ['Master Magmoth', 70, 100],
            ['Location: Faj\'ro dungeon', 30, 100],
            ['Number of deaths: {event_count}', 30, 40],
            ['Total time spent: {hours}h, {minutes}m', 30, 40]
            ]
    y = 0
    for i, arr in enumerate(text):
        y += arr[2]
        font = ImageFont.truetype(font_path, arr[1])
        txt = arr[0].format(event_count=total_clips,
                            hours=hours, minutes=minutes, seconds=seconds)
        draw = ImageDraw.Draw(img)
        fontColor = (255, 255, 255)

        text_size = draw.textbbox((0, 0), txt, font)

        x = (video_width - text_size[2]) / 2
        # y = (img.height - text_size[3]) / 2

        draw.text((x, y), txt, font=font, fill=fontColor)

    # cv2.imshow('frame', np.array(img))
    # cv2.waitKey(2000)

    intro_image = f"{temp_dir}/{out_basename}.png"
    img.save(intro_image)

    print("Combining intro with clips")
    temp_file = f"{temp_dir}/{out_filename}"
    shutil.copy(out_filename, temp_file)
    cmd = f"ffmpeg -loop 1 -t {intro_duration} -i {intro_image} -i {temp_file} -f lavfi -t 0.1 -i anullsrc -filter_complex '[0][2][1:v][1:a]concat=n=2:v=1:a=1[v][a]' -map '[v]' -map '[a]' -fflags +genpts -loglevel error -y {out_filepath}" # noqa
    os.system(cmd)

    os.remove(temp_file)
    os.remove(intro_image)


file_index = 0

for f in files:
    process_file(f)
    file_index = file_index + 1


if intro_duration > 0:
    combine_clips()

add_info_clip()


# Release the video capture object
cap.release()


print()
print("All done")
print(f"Output file: {out_filepath}")
