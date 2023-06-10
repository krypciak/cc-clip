# cc-clip
Python script that makes clips of deaths from recordings from the game CrossCode

```
usage: cc-clip.py [-h] -t {ko,death} -o OUTPUT [-A AFTER] [-B BEFORE] [--grid GRID] [--threads THREADS] [--no-delete-clips]
                  [--no-generate-clips] [--no-tile] [--intro-duration INTRO_DURATION] [--progess-graph]
                  files [files ...]
```
<br>

Required arguments:
- `--type (ko|death)` search for ko events (in pvp battles) or search for death events (when hp reaches 0)<br>
- `--output OUT.mkv` where to write the video file, has to end with `.mkv`<br>
- `files` Requireds at least 1 file. All input files have to be in `.mkv` format<br>

Other arguments:
- `--after` time after the event to put in clips (in miliseconds) (default: 150)<br>
- `--before` time before the event to put in clips (in miliseconds) (default:150)<br>
- `--grid SIZE` create a SIZExSIZE grid of clips (default: no grid)<br>
- `--intro-duration SECONDS` Intro duration, `0` to disable (default: 2)<br>
- `--progress-graph` Enable generation of a graph that represents boss % hp on deaths. Use only on boss fights<br>
- `--threads N`use N threads (default: 4)<br>

Debug options:<br>
- `--no-delete-clips` Don't delete temporary files after completion<br>
- `--no-generate-clips` Skip clip generation (use exisiting ones, only useful when with `--no-delete-clips`)<br>
-  `--no-tile` Skip clip tiling (use exisiting ones, only useful when with `--no-delete-clips`)<br>

Example usage:
```
python cc-clip.py --type death --threads 16 -B 1000 -A 700 -o clip.mkv --grid 4 --intro-duration 6 --progess-graph \
  ~/Videos/2023-05-10_15-31-44.mkv ~/Videos/2023-05-10_16-25-09.mkv
```
