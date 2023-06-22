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
import math
import matplotlib.pyplot as plt
from termcolor import colored
import json


parser = argparse.ArgumentParser(description='Create clips of events using recordings from the hit game CrossCode') # noqa
parser.add_argument('-t', '--type',         type=str, required=True, choices=["ko", "death"], help='Type of event to search for') # noqa
parser.add_argument('files',                type=argparse.FileType('r'), nargs='+',           help='The files to process') # noqa
parser.add_argument('-o', '--output',       type=str, required=True,                          help='Output file destination path') # noqa
parser.add_argument('-A', '--after',        type=int,             default=150,                help='Number of threads to use when searching for deaths (default: 4)') # noqa
parser.add_argument('-B', '--before',       type=int,             default=500,                help='Clip time before event in ms (default: 150)') # noqa
parser.add_argument('--grid',               type=int,             default=1,                  help='Grid size') # noqa
parser.add_argument('--threads',            type=int,             default=4,                  help='Clip time after event in ms (default: 500)') # noqa
parser.add_argument('--intro-duration',     type=float,           default=2,                  help='Duration of the intro. 0 to disable (default: 2)') # noqa
parser.add_argument('--fight-name',         type=str,                                         help='Fight name to put in intro') # noqa
parser.add_argument('--fight-location',     type=str,                                         help='Fight location to put in intro') # noqa
parser.add_argument('--progress-graph',      action='store_true',  default=False,             help='Generate boss progress graph (bosses only)') # noqa
parser.add_argument('--last-count',         type=int,             default=0,                  help='Number of clips at the end to apply custom -B and -A, and also keep from being tiled (default: 0)') # noqa
parser.add_argument('-lA', '--last-after',  type=int,             default=600,                help='Clip time before event in ms, only applies to last clips (default: 400)') # noqa
parser.add_argument('-lB', '--last-before', type=int,             default=4000,               help='Clip time after event in ms, only applies to last clips (default: 1500)') # noqa
parser.add_argument('-sef', '--store-event-frames', action='store_true',  default=False,      help='Store event frames after video is processed') # noqa
parser.add_argument('-ref', '--restore-event-frames', action='store_true',  default=False,    help='Restore event frames from event-frames.json') # noqa
parser.add_argument('--no-delete-clips',    action='store_false', default=True,               help='Dont delete the clips at the end') # noqa
parser.add_argument('--no-generate-clips',  action='store_false', default=True,               help='Skip clip generation (use already existing ones)') # noqa
parser.add_argument('--no-tile',            action='store_false', default=True,               help='Skip tile video generation (use already exisiting ones)') # noqa

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
store_event_frames = args.store_event_frames
restore_event_frames = args.restore_event_frames
intro_duration = args.intro_duration
fight_location = args.fight_location
fight_name = args.fight_name
progress_graph = args.progress_graph

last_clip_count = args.last_count
last_after = args.last_after
last_before = args.last_before


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
    template = cv2.imread("ko.png", cv2.IMREAD_GRAYSCALE)
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
else:
    print("impossible")
    exit()

template_w, template_h = template.shape
for x in range(len(template_cuts)):
    template_cuts[x].append(template_cuts[x][0] + template_h + 1)
    template_cuts[x].append(template_cuts[x][1] + template_w + 1)

frame_match_array_lock = threading.Lock()
frame_count_lock = threading.Lock()
frame_lock = threading.Lock()
graph_data_lock = threading.Lock()

work_dir = os.getcwd()
clips_dir = f"{work_dir}/clips"
if not os.path.exists(clips_dir):
    os.makedirs(clips_dir)

if not os.path.exists(f"{clips_dir}/grid"):
    os.makedirs(f"{clips_dir}/grid")

temp_dir = f"{work_dir}/tmp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

event_frame_file = f"{work_dir}/event-frames.json"
if restore_event_frames:
    if os.path.exists(event_frame_file):
        with open(event_frame_file, "r") as f:
            event_frame_json = json.load(f)
    else:
        event_frame_json = {}
else:
    event_frame_json = {}

max_frame_jump = 100

total_videos_length = 0
total_clips = 0
clips = []

phase_count = None
graph_data = [[], []]
graph_frame_offset = 0


def process_file(video):
    print(f"Processing file {video}")
    global cap, video_frame_count, video_fps, video_width, video_height, start_frame, end_frame, total_videos_length, graph_data, phase_count, beg_frames # noqa
    # Load the video file
    cap = cv2.VideoCapture(video)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_videos_length = total_videos_length + video_frame_count / video_fps

    start_frame = 0
    end_frame = -1

    tmp_json = {"frames": [], "phases": -1, "boss-hp-data": []}
    write_json = False

    if restore_event_frames and video in event_frame_json and "frames" in event_frame_json[video]: # noqa
        beg_frames = event_frame_json[video]["frames"]
    else:
        search_for_frames()
        beg_frames = get_unique_moments(frames_with_matches)
        if store_event_frames:
            write_json = True
            tmp_json["frames"] = beg_frames

    if restore_event_frames and progress_graph and video in event_frame_json and "boss-hp-data" in event_frame_json[video]: # noqa
        phase_count = event_frame_json[video]["phases"]
        graph_data = event_frame_json[video]["boss-hp-data"]
    else:
        get_boss_health_data()
        if store_event_frames:
            write_json = True
            tmp_json["phases"] = phase_count
            tmp_json["boss-hp-data"] = graph_data

    if write_json:
        with open(event_frame_file, "w") as f:
            json.dump(event_frame_json, f)

    # for value in frames_with_matches:
    #     if value > 20000 and value < 23000:
    #         print(value, end=' ')
    # print()

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 50000)
    # ret, img = cap.read()
    # show_image(img, 0)

    # for frame in beg_frames:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #     ret, img = cap.read()
    #     print(frame)
    #     show_image(img, 0)
    generate_clips(video)


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
        add_frame_to_array(frame_number)
        return True

    return False


def color_diff(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    diff = math.sqrt((int(r1) - r2)**2 + (int(g1) - g2)**2 + (int(b1) - b2)**2)
    return diff


def reverse_color(color):
    return [color[2], color[1], color[0]]


def smaller_color_diff(color1, arr):
    arr = [color_diff(color1, color2) for color2 in arr]
    arr.sort()
    return arr[0]


def get_boss_health_data():
    global graph_frame_offset, phase_count, beg_frames
    print("Getting boss health data")
    for frame_number in tqdm(beg_frames, unit=' frames'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = cap.read()
        if not ret:
            continue

        precentage = None

        img = cv2.resize(img, (568, 320))
        # show_image(img, 0)
        # exit()

        hp_color = reverse_color([255, 122, 122])
        phase_color = reverse_color([0, 0, 0])

        empty_colors = [reverse_color([76, 76, 76]),
                        reverse_color([95, 68, 70]),
                        reverse_color([255, 255, 255]),
                        reverse_color([0, 0, 0]),
                        reverse_color([36, 0, 0])]

        set_boss_phases = not phase_count
        print(f"set_boss_phases: {set_boss_phases}")

        start_x = 43
        end_x = 551
        x = start_x - 1
        y = 309
        total_bar_length = end_x - start_x + 1

        while x < end_x + 1:
            x += 1
            ccolor = img[y, x]
            ncolor = img[y, x+1]

            # print(ccolor)
            hp_diff = color_diff(ccolor, hp_color)
            # print(f"hp diff: {hp_diff}")
            if hp_diff < 55:
                # img[y, x] = (0, 255, 255)
                continue

            if x < end_x - 3:
                diff_phase = color_diff(ncolor, phase_color)
                # print(f"phase diff: {diff_phase}")
                if diff_phase < 20:
                    x += 2
                    if set_boss_phases:
                        phase_count = round(total_bar_length / (x - start_x)) # noqa
                        set_boss_phases = False
                    continue

            diff_empty = smaller_color_diff(ccolor, empty_colors)
            # print(f"empty diff: {diff_empty}")

            if diff_empty < 30 or x > end_x - 3:
                precentage = (x - start_x) / total_bar_length * 100
                break

            precentage = -2
            break
            # print(ccolor)
            # cv2.putText(img, "fatal error", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3) # noqa
            # img[y-1, x] = (255, 0, 0)
            # img[y+1, x] = (255, 0, 0)
            # show_image(img, 0)
            # exit()

        if precentage == -2:
            print(colored(f"WARNING: Boss hp bar reading error at {frame_number}, skipping", 'yellow')) # noqa
            beg_frames.remove(frame_number)
            continue

        if not precentage:
            print("Boss hp bar not detected. Not a bossfight? Consider disabling --progress-graph") # noqa
            show_image(img, 0)
        else:
            if precentage >= 101:
                print(f"Precentage higher than 101: {precentage}. Error reading bar hp") # noqa
                exit()

            if precentage > 100:
                precentage = 100
            with graph_data_lock:
                # cv2.putText(img, str(round(precentage, 3)), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3) # noqa
                # print(frame_number)
                # print(f"x: {x}")
                # show_image(img, 0)

                graph_data[0].append(frame_number + graph_frame_offset)
                graph_data[1].append(precentage)

    graph_frame_offset += video_frame_count


def add_frame_to_array(frame_number):
    with frame_match_array_lock:
        frames_with_matches.append(frame_number)


def process_frame():
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
    for _ in range(thread_count):
        t = threading.Thread(target=process_frame, args=())
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    pbar.close()

    frames_with_matches.sort()

    print(f"\nFound {len(frames_with_matches)} frames with matches\n")
    # print(frames_with_matches)
    # print("\n")


def get_unique_moments(frames):
    print("Searching for unique moments")

    global beg_frames
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


def generate_clips(video):
    print("Generating clips")

    global pbar
    pbar = tqdm(total=len(beg_frames), unit='clip')
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor: # noqa
        futures = [executor.submit(generate_video_of_frame, video, frame_number, i) for i, frame_number in enumerate(beg_frames)] # noqa

        concurrent.futures.wait(futures)
        pbar.close()

    print("\n")


def generate_video_of_frame(video, frame_number, index):
    filename = f"{file_index}_{frame_number}.mkv"

    if do_generate_clips:
        video_file = f"{temp_dir}/video-{filename}"
        audio_file = f"{temp_dir}/audio-{filename}"

        is_last_enougth = len(beg_frames)-last_clip_count > index
        time_before = (gen_video_before if is_last_enougth else last_before)
        time_after = (gen_video_after if is_last_enougth else last_after)

        start_frame = frame_number - int(time_before*video_fps/1000) # noqa
        end_frame = frame_number + int(time_after*video_fps/1000) # noqa

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file, fourcc, video_fps, (video_width, video_height), True) # noqa

        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        cap.release()

        # Extract audio
        start_time = str(max(0, frame_number / video_fps - time_before/1000)) # noqa
        end_time = str((time_before+time_after)/1000)
        cmd = f"ffmpeg -i {video} -ss {start_time} -t {end_time} -loglevel error -vn -acodec copy {audio_file}" # noqa
        os.system(cmd)

        # Combine video and audio
        cmd = f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -map 0:v:0 -map 1:a:0 -loglevel error -y {clips_dir}/{filename}" # noqa
        os.system(cmd)

        # Clean up
        os.remove(video_file)
        os.remove(audio_file)

    with frame_lock:
        clips.append(filename)
        pbar.update()

    pass


def combine_clips():
    global clips, total_clips, grid_size
    clips = natsorted(clips) # noqa
    total_clips = len(clips)
    clips_backup = clips.copy()
    if total_clips == 0:
        print(colored("Something went wrong, clips size is 0", 'red'))
        exit()

    if grid_size < 2:
        print("Combining clips")
        ffmpeg_combine([f"{clips_dir}/{f}" for f in clips], f"{out_filepath}", copy=True) # noqa
    else:
        print("Tiling clips")
        clips_per_screen = grid_size * grid_size
        tiled_clips = []

        pbar_len = len(clips) - last_clip_count
        if pbar_len == 0:
            pbar_len = len(clips)
            clips_per_screen = pbar_len

        pbar = tqdm(total=pbar_len, unit=' frames') # noqa

        while len(clips) >= last_clip_count or len(clips) <= 1:
            selected_clips = []
            for _ in range(clips_per_screen):
                if len(clips) == 0:
                    break
                selected_clips.append(f"{clips_dir}/{clips.pop(0)}")

            if len(selected_clips) < 1:
                break

            optimal_grid_size = int(math.ceil(math.sqrt(len(selected_clips))))
            grid_size = optimal_grid_size

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
                if index > 0 and index < len(selected_clips) - 1:
                    input_overlays += f"[tmp{index-1}][video{index}] overlay=shortest=1:x={video_width//grid_width * (index%grid_width)}:y={video_height//grid_height * (index//grid_height)} [tmp{index}];" # noqa
                if index == len(selected_clips) - 1:
                    input_overlays += f"[tmp{index-1}][video{index}] overlay=shortest=1:x={video_width//grid_width * (index%grid_width)}:y={video_height//grid_height * (index//grid_width)}" # noqa
                audio_volume += f"[{index}:a]volume={grid_size}[a{index}];"
                audio_merge += f"[a{index}]"

            cmd = f"ffmpeg{input_videos} -filter_complex \"{input_setpts}{input_overlays},premultiply=inplace=1;{audio_volume}{audio_merge}amix=inputs={len(selected_clips)}[audio_out]\" -map '[audio_out]' -fflags +genpts -loglevel error -y {temp_file_out}" # noqaa

            if do_tile:
                os.system(cmd)
            pbar.update(clips_per_screen)

            tiled_clips.append(temp_file_out)

        pbar.close()

        if len(clips) < last_clip_count:
            clips = clips_backup[-last_clip_count:]

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
        else:
            print("Combining tiled clips")
            ffmpeg_combine(tiled_clips, out_filepath)

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
            [fight_name, 70, 100],
            [f'Location: {fight_location}', 30, 100],
            ['Number of deaths: {event_count}', 30, 40],
            ]
    time_txt = 'Total time spent: '
    if hours > 0:
        time_txt += '{hours}h, '
    time_txt += '{minutes}m'

    text.append([time_txt, 30, 40])

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

    if progress_graph:
        assert phase_count is not None
        x_data = [frame_number/video_fps for frame_number in graph_data[0]]
        y_data = graph_data[1]
        death_count = len(x_data)

        fig, ax = plt.subplots(figsize=(max(8, min(video_width - 20 * 2, total_videos_length/60/10)), 6)) # noqa

        # Add labels and title
        plt.xlabel('time (hh:mm)', color='black')
        plt.ylabel('% boss hp left', color='black')
        ax.set_title('Boss hp on deaths over time', color='black', y=1.05)

        ax.plot(x_data, y_data, color='#1f77b4')

        ticks = [int(100/phase_count*i) for i in range(1, phase_count)]
        tick_labels = [f"{ticks[i]}%" for i in range(phase_count-1)]
        plt.yticks(ticks, tick_labels)
        ax.set_ylim(0, 110)

        # the colors are inverted here for some reson
        plt.grid(axis='y', color='#d8d8d8', linestyle='-', linewidth=2)
        fig.patch.set_facecolor('gray')
        ax.scatter(
                x_data,
                [y_data[i] for i in range(death_count)],
                s=[40 for _ in range(death_count)],
                c=['#2ca02c' for _ in range(death_count)],
                zorder=2)

        xtick_count = 8
        xticks = []
        xtick_labels = []
        for i in range(xtick_count + 1):
            pos = x_data[-1]/xtick_count*i
            xticks.append(pos)

            minutes, seconds = divmod(pos, 60)
            hours, minutes = divmod(minutes, 60)
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds)
            xtick_labels.append('{:02d}:{:02d}'.format(hours, minutes))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, minor=False)

        canvas = fig.canvas
        canvas.draw()
        graph_width, graph_height = canvas.get_width_height()
        plot_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((graph_height, graph_width, 3)) # noqa
        plot_data_copy = np.copy(plot_data)

        # fix colors being wrong for some reason
        for x1 in range(len(plot_data)):
            for y1 in range(len(plot_data[x1])):
                r, g, b = plot_data[x1][y1]
                plot_data_copy[x1][y1] = (b, g, r)

        graph_img = Image.fromarray(plot_data_copy)

        x = int((video_width - graph_img.size[0]) / 2)
        y += 100
        img.paste(graph_img, (x, y))

    # show_image(np.array(img), 0)

    intro_image = f"{temp_dir}/{out_basename}.png"
    img.save(intro_image)

    print("Combining intro with clips")
    temp_file = f"{temp_dir}/{out_filename}"
    shutil.copy(out_filepath, temp_file)
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

if cleanup_clips:
    print("Removing temporary clips")
    shutil.rmtree(clips_dir)

print()
print("All done")
print(f"Output file: {out_filepath}")
