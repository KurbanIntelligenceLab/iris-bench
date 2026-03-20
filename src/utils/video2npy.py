import numpy as np
import os
import cv2
import matplotlib
matplotlib.use("Agg")  # no GUI — avoids "Unable to open monitor" when no display
import matplotlib.pyplot as plt

# Function to iterate over folders and process video files
def iterate_and_process_videos(base_path):
    for root, dirs, files in os.walk(base_path):
        # If no subdirectories, we are in a terminal folder
        if not dirs:
            # Iterate over mp4 files (case-insensitive) that don't contain 'mask' in the filename
            mp4_files = [f for f in files if f.lower().endswith('.mp4') and 'mask' not in f.lower()]
            for mp4_file in mp4_files:
                video_path = os.path.join(root, mp4_file)
                print(f"Processing video: {video_path}")
                # One .npy per video file (e.g. 01.mp4 -> 01.npy) so multiple videos per folder work (e.g. IRIS)
                stem = os.path.splitext(mp4_file)[0]
                numpy_save_path = os.path.join(root, f"{stem}.npy")
                print(f"Saving numpy array to: {numpy_save_path}")
                video_array = process_video(video_path, new_width=100, new_height=56)
                if video_array is not None and len(video_array) > 0:
                    np.save(numpy_save_path, video_array)
                else:
                    print(f"Skipping save (no samples): {numpy_save_path}")

# Function to process video: resize, convert to grayscale, and normalize
def process_video(video_path, new_width, new_height):
    print('---------------------------------------------------------')
    # Use os.path so paths work on both Windows and Unix
    path_parts = os.path.normpath(video_path).split(os.sep)
    last_two_folders = '_'.join(path_parts[-3:-1]) if len(path_parts) >= 2 else path_parts[-1] if path_parts else 'frame'
    cap = cv2.VideoCapture(video_path)
    frames = []

    #get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'Original video fps: {fps}')
    

    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        # Normalize the frame to range [0, 1]
        normalized_frame = gray_frame / 255.0
        frames.append(normalized_frame)
    
    cap.release()
    
    # Convert list of frames to numpy array  
    dataset = []
    np_frames = np.array(frames)
   
    print("number of frames:", np_frames.shape)
    nframes = np_frames.shape[0]
    nf = 10  # number of frames per sample (must match main.py / loader expectation)

    # Support short videos: use smaller step when video has fewer than 60 frames.
    # step=6 needs 60 frames; step=2 needs 20; step=1 needs 10.
    if nframes >= 60:
        step = 6
    elif nframes >= 20:
        step = 2
    elif nframes >= nf:
        step = 1
    else:
        step = None

    if step is None:
        print(f"WARNING: Video too short. Need at least {nf} frames, but only have {nframes} frames.")
        print(f"Skipping visualization and returning None (will not save .npy).")
        return None

    max_frames = nframes - (nf * step)
    for i in range(max_frames):
        end = i + nf * step
        frames_temp = np_frames[i:end:step]
        dataset.append(np.expand_dims(frames_temp, axis=1))

    dt = step / 60.0
    dataset = np.array(dataset)
    print(f"Step (frame skip): {step}  ->  need at least {nf*step} frames for 1 sample")
    print(f"Time between frames: {dt}")
    print(f"Shape of dataset: {dataset.shape}")

    if len(dataset) == 0:
        print(f"WARNING: No samples produced. Need at least {nf*step} frames, have {nframes}.")
        return None

    samples_number = -1
    sample_frames = dataset[samples_number,0,:,:]
    for i in range(1, dataset.shape[1]):
        sample_frames = np.hstack((sample_frames, dataset[samples_number,i,:,:]))

    sample_frames = sample_frames.transpose(1,2,0)
    plt.figure(figsize=(5,25))
    plt.imshow(sample_frames)

    if not os.path.exists('./Frames'):
        os.makedirs('./Frames')
    plt.savefig(f'./Frames/{last_two_folders}_sample_frame.png', dpi=300)
    plt.close()

    print('---------------------------------------------------------')
    return dataset

if __name__ == "__main__":

    base_path = "../../texas26/"  # Change to your base folder path
    # base_path = "../../delfys75/"  # Change to your base folder path
    iterate_and_process_videos(base_path)
