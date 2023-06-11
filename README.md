Python script that makes clips of deaths from recordings from the game CrossCode

```
usage: cc-clip.py [-h] -t {ko,death} -o OUTPUT [-A AFTER] [-B BEFORE] [--grid GRID] [--threads THREADS]
                  [--intro-duration INTRO_DURATION] [--fight-name FIGHT_NAME]
                  [--fight-location FIGHT_LOCATION] [--progess-graph] [--last-count LAST_COUNT]
                  [-lA LAST_AFTER] [-lB LAST_BEFORE] [--no-delete-clips] [--no-generate-clips] [--no-tile]
                  files [files ...]
```
<br>

Required arguments:
- `--type (ko|death)` Search for ko events (in pvp battles) or search for death events (when hp reaches 0)<br>
- `--output OUT.mkv` Where to write the video file, has to end with `.mkv`<br>
- `files` Requireds at least 1 file. All input files have to be in `.mkv` format<br>

Other arguments:
- `--after` Time after the event to put in clips (in miliseconds) (default: 150)<br>
- `--before` Time before the event to put in clips (in miliseconds) (default:150)<br>
- `--grid SIZE` Create a SIZExSIZE grid of clips (default: no grid)<br>
- `--intro-duration SECONDS` Intro duration, `0` to disable (default: 2)<br>
- `--fight-name NAME` This name will be put in the intro<br>
- `--fight-location LOCATION` This will be put in the intro<br>
- `--progress-graph` Enable generation of a graph that represents boss % hp on deaths. Use only on boss fights<br>
- `--last-count N` Make sure there are N clips at the end that aren't tiled<br>
- `--last-after` Only applies to last clips specified by `--last-count`, specifies time like `--after`<br>
- `--last-before` Only applies to last clips specified by `--last-count`, specifies time like `--before`<br>
- `--threads N`use N threads (default: 4)<br>

Debug options:<br>
- `--no-delete-clips` Don't delete temporary files after completion<br>
- `--no-generate-clips` Skip clip generation (use exisiting ones, only useful when with `--no-delete-clips`)<br>
-  `--no-tile` Skip clip tiling (use exisiting ones, only useful when with `--no-delete-clips`)<br>

Example usage:
```
python cc-clip.py --type death --threads 16 -B 1000 -A 700 -o clip.mkv --grid 4 \
    --intro-duration 6 "Master Magmoth" --fight-location "Faj'ro dungeon" --progess-graph \
    --last-count 5 -lB 4000 -lA 600 \
    ~/Videos/2023-05-10_15-31-44.mkv ~/Videos/2023-05-10_16-25-09.mkv
```
