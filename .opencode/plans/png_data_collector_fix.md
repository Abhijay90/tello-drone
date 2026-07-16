# Fix data_collector.py

### Problem
The current data collector saves frames as hex-embedded bytes inside JSON files, which makes them huge and hard to inspect. Also, the old data is unusable for model training with standard libraries.

### Solution

**Step 1: Delete old data**
```bash
rm -rf training/data/open_palm training/data/closed_fist training/data/thumbs_up training/data/thumbs_down training/data/palm_up training/data/palm_down training/data/palm_left training/data/palm_right
```

**Step 2: Modify data_collector.py**
Change the `collect_sample` method to save frames as PNG files and metadata as separate JSON files.

Changes needed:
- In `collect_sample`:
  - Remove frame from JSON output
  - Write frame as PNG to `data/<gesture>/sample_NNNN.png`
  - Write metadata (landmarks, width, height, timestamp) to `data/<gesture>/sample_NNNN.json`

- In `FPVDataCollector._collect_sample`:
  - Similar change: save PNG + JSON separately

**Step 3: Modify data_loader.py**
Update `load_dataset` to load PNG files along with JSON metadata and reconstruct frames from PNG.

**Steps to implement:**
1. Edit `training/data_collector.py` to save PNG images
2. Edit `training/data_loader.py` to load PNG + JSON format
3. Delete old data
4. Rerun data collection
5. Verify loader works with new format and model training proceeds

### Benefits
- Smaller JSON files
- Standard format easily loadable by any image library
- Better compatibility with existing tooling (OpenCV, PIL etc)
- Easier to inspect individual samples
- Less memory overhead during collection

### Files to modify
- `training/data_collector.py` - `collect_sample` and `FPVDataCollector._collect_sample`
- `training/data_loader.py` - `load_dataset` and `load_sample` methods
- `tello-drone/data/` - delete all old files
