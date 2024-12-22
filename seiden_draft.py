from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import numpy as np
import av, time
import pandas as pd
import torch
# load yolo model
model = YOLO("yolov8m.pt")
model.info()


# as comparision:
# run result on video directly
# start_time = time.perf_counter()
# model(source=vid_name)
# duration = time.perf_counter() - start_time
# print(f"return inference results in {duration} second.")
vids = ["Station1_in.mp4"]
# vid_name = "Station1_in.mp4"
for vid_name in vids:
  def convert2images(path, frame_count_limit = 300000, size = None):
    vid = cv2.VideoCapture(path)
    if (vid.isOpened() == False):
        print(f"Error opening video {path}")
        raise ValueError

    frame_count = min(vid.get(cv2.CAP_PROP_FRAME_COUNT), frame_count_limit)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    channels = 3

    if size is not None:
        height = size[0]
        width = size[1]

    assert (frame_count == int(frame_count))
    assert (width == int(width))
    assert (height == int(height))

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)

    print(f"meta data of the video {path} is {frame_count, height, width, channels}")
    image_matrix = np.ndarray(shape=(frame_count, height, width, channels), dtype = np.uint8)

    error_indices = []

    for i in tqdm(range(frame_count)):
        success, image = vid.read()
        if not success:
            print(f"Image {i} retrieval has failed")
            error_indices.append(i)
        else:
            ### need to resize the image matrix
            image = cv2.resize(image, (width, height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            image_matrix[i, :, :, :] = image  # stored in rgb format

    return image_matrix

  def load_dataset(video_path):
    images = convert2images(video_path)
    return images

  images = load_dataset(vid_name)
  # # for illustration purpose.
  # print(images)

  anchor_count = int(len(images) * 0.1)
  print(anchor_count)

  import matplotlib.pyplot as plt

  # def show_image(image_array, index):
  #   if index < 0 or index >= len(image_array):
  #     print("Index out of range.")
  #     return
  #   plt.figure(figsize=(10, 6))
  #   plt.imshow(image_array[index])
  #   plt.axis('off')
  #   plt.title(f"Image at index {index}")
  #   plt.show()

  # show_image(images, 604)


  # get video length
  def get_video_length(video: str):
    container = av.open(video, "r")
    stream = container.streams.video[0]
    return stream.frames

  # get I-frame indices
  def get_iframe_indices(video: str):
    start_time = time.perf_counter()
    video_container = av.open(video, "r")
    video_stream = video_container.streams.video[0]

    key_indexes = []
    count = 1
    for packets in video_container.demux(video_stream):
      if packets.is_keyframe:
        key_indexes.append(count)
      count += 1

    duration = time.perf_counter() - start_time
    print(f"return {len(key_indexes)} frames in {duration} second.")

    key_indexes = np.array(key_indexes)
    return key_indexes

    # index-construction algorithm based on the paper Algorithm 1
  def index_counstruction(video, budget, sampling_ratio = 0.8):
    dataset_length = len(images)
    iframe_indices = get_iframe_indices(video)
    print("alpha * N :", int(budget * sampling_ratio))
    index_counstruction_reps = int(budget * sampling_ratio)

    rep_indices = [1, dataset_length] # make sure head and tail of dataset can be covered
    if index_counstruction_reps >= len(iframe_indices) + 2:
      rep_indices.extend(iframe_indices)
      rep_indices = list(set(rep_indices))
      rep_indices = sorted(rep_indices)
    else:
      # random sample
      subset = list(np.random.choice(iframe_indices, index_counstruction_reps, replace = False))
      rep_indices.extend(subset)
      rep_indices = list(set(rep_indices))
      rep_indices = rep_indices[:index_counstruction_reps]
      rep_indices = sorted(rep_indices)
    print(f"final number of indices that be selected is {len(rep_indices)}")
    return rep_indices, dataset_length

  # for illustration purpose:
  iframes_list, frames_count = index_counstruction(vid_name, anchor_count)

  # 2. based on the video looping
  video = cv2.VideoCapture(vid_name)
  frame_to_sample = []

  # Extract I-frames for sampling
  while video.isOpened():
    # read each frame
    ret, frame = video.read()
    if not ret:
      break
    frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_id in iframes_list:
      frame_to_sample.append((frame_id, frame))

  # save frame_id and its detection labels
  pairs = {}
  pairs_cache = {}
  for frame_id, frame in frame_to_sample:
    results = model(frame, conf = 0.85)
    pairs[frame_id] = results
    confidences = [det[4] for det in results[0].boxes.data.cpu().numpy()]
    pairs_cache[frame_id] = confidences

  # Extract class labels
  for box in results[0].boxes:
      class_id = int(box.cls)  # Get class ID
      class_label = results[0].names[class_id]  # Get class label from class ID
      print(f'Detected class: {class_label}')  # Print class label

  # The first step is, we want to create video segments
  target_dnn_cache = pairs_cache
  def scores(dnn_output):
    return dnn_output if len(dnn_output) > 0 else [0.0]

  def create_segments(keyframes, target_dnn_cache, score_func, video_length):
    segments = {}
    step_size = video_length // 100

    rep_index = 0
    for i in range(1, video_length + 1, step_size):
      start, end = i, min(i + step_size - 1, video_length)
      # find the keyframes in each segments:
      reps = []
      while rep_index < len(keyframes):
        if keyframes[rep_index] <= end:
          reps.append(keyframes[rep_index])
        else:
          break
        rep_index += 1
      cache = []
      if reps:
        for rep in reps:
          cache.extend(score_func(target_dnn_cache[rep]))
      else: cache.append(0.0)
      segments[(start, end)] = {"keyframe": reps, "scores": cache, "dist": np.var(cache) if cache else 0.0}

    return segments

  segments = create_segments(iframes_list, target_dnn_cache, scores, len(images))
  # print(segments)
  # print([item for key, item in segments.items() if segments[key]["keyframe"] != []])
  # make a selection method based on UCB policy of multi-arm bandit
  def select_method(segments, reps, full_set):
    mab_values = []
    c_value = 2.0

    for key in segments:
      reward = segments[key]["dist"]
      members = segments[key]["keyframe"]
      # use the UCB formula: reward_{k, t - 1} + c * \sqrt{2ln N / N_{k, t-1}}
      mab_value = reward + c_value * np.sqrt(2 * np.log(len(reps)) / len(members)) if len(members) > 0 else reward
      mab_values.append(mab_value)

    select_segment = np.argmax(mab_values)
    if select_segment in full_set:
      # that means we need to choose another segment
      while select_segment in full_set:
        select_segment = np.random.randint(0, len(mab_values))

    start_index, end_index = list(segments.keys())[select_segment]
    # select in those index range
    choices = []
    for i in range(start_index, end_index + 1):
      if i not in segments[(start_index, end_index)]["keyframe"]:
        choices.append(i)

    # if not being choice:
    if len(choices) == 0:
      segments[(start_index, end_index)]["dist"] = 0.0
      full_set.add(select_segment)
      selected_frameID, (start_index, end_index) = select_method(segments, reps, full_set)
    else:
      selected_frameID = np.random.choice(choices, 1)[0]

    return selected_frameID, (start_index, end_index)

  # update for ucb policy
  def update(segments, key, selected_frameID, target_dnn_cache, scoring_func):
    segments[key]["keyframe"].append(selected_frameID)
    result = scoring_func(target_dnn_cache[selected_frameID])
    if isinstance(result, list):
        segments[key]["scores"].extend(result)
    else:
        segments[key]["scores"].append(result)
    segments[key]["dist"] = np.var(segments[key]["scores"])

    return segments

  # main function for query_execution, drafted from Algorithm 2.
  def query_execution(video, budget, iframes_list, target_dnn_cache, score_func, pairs):
    # create segments
    iframes_list = sorted(iframes_list)
    length = iframes_list[-1] - iframes_list[0] + 1
    segments = create_segments(iframes_list, target_dnn_cache, score_func, length)

    # find reps
    for i in tqdm(range(budget - len(iframes_list))):
      new_frameID, key = select_method(segments, iframes_list, set())
      # after selecting this new frame, run the inference on it and record in cache:
      iframes_list.append(new_frameID)
      # use the feature of random access in cv2.videoCapture
      video_item = cv2.VideoCapture(video)
      video_item.set(cv2.CAP_PROP_POS_FRAMES, new_frameID)
      ret, frame = video_item.read()
      video_item.release()
      # if ret is None, then move to next to re-select
      if not ret: continue
      # if we capture the frame, do inference and record the result
      results = model(frame, conf = 0.85)
      pairs[new_frameID] = results
      confidences = [det[4] for det in results[0].boxes.data.cpu().numpy()]
      target_dnn_cache[new_frameID] = confidences
      segments = update(segments, key, new_frameID, target_dnn_cache, score_func)

    # if everything is correctly done, return pairs and target_dnn_cache
    return pairs, target_dnn_cache

  # let us see the result
  pairs, target_dnn_cache = query_execution(vid_name, anchor_count, iframes_list, target_dnn_cache, scores, pairs)
  # print(sorted(pairs.keys()))

  def extract_frames_from_video(video_path, frames_dict, output_dir):
      """
      Extract specified frames from a video and save them as images.

      Args:
          video_path (str): Path to the input video.
          frames_dict (dict): Key: frame_id, value: any associated data.
          output_dir (str): Directory where the extracted frames will be saved.

      Returns:
          None
      """
      # Ensure the output directory exists
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)

      # Open the video
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          print("Error: Unable to open video.")
          return

      frame_count = 0
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break

          # Check if the current frame ID is in the frames_dict
          if frame_count in frames_dict:
              # Construct the output filename
              output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
              # Save the frame as an image
              cv2.imwrite(output_file, frame)
              print(f"Saved: {output_file}")

          frame_count += 1

      # Release resources
      cap.release()
      print("Frame extraction completed.")

  # Example usage
  # Assume you have a dictionary with frame IDs
  # frames_to_extract = {1: 'value1', 5: 'value2', 10: 'value3'}  # Example frame dictionary
  extract_frames_from_video(vid_name, pairs, "C:/Users/Hanzalah Choudhury/Desktop/SEIDEN/video_out")


  # in total, our sample video has 605 frames.
  # we propagate unsampling labels here
  def propagate_labels(pairs, total_frames):
    """
    We use the nearest frame to propagate the labels.
    """
    sampled_frames = sorted(pairs.keys())
    for frame_id in range(1, total_frames + 1):
      if frame_id in pairs:
        continue

      # find the nearest sampled frames
      closest_left = max([sf for sf in sampled_frames if sf <= frame_id], default=None)
      closest_right = min([sf for sf in sampled_frames if sf >= frame_id], default=None)

      if closest_left is not None and closest_right is not None:
        # we use the nearest label to propagate result
        if frame_id - closest_left <= closest_right - frame_id:
          pairs[frame_id] = pairs[closest_left]
        else:
          pairs[frame_id] = pairs[closest_right]
      elif closest_left is not None:
        # only left, we take left
        pairs[frame_id] = pairs[closest_left]
      elif closest_right is not None:
        # only right, we take right
        pairs[frame_id] = pairs[closest_right]

    return pairs
  full_pairs = propagate_labels(pairs, len(images))
  frame_ids = []
  for i in full_pairs:
  #    print(i)
    ts = full_pairs[i][0].boxes.cls
  #    print(i,ts)
    if torch.any(ts == 0):
        frame_ids.append(i)
  # import cv2
  frame_ids.sort()
  print(frame_ids)
  def create_video_from_frames(video_path, frame_ids, output_path):
      # Open the original video
      cap = cv2.VideoCapture(video_path)

      # Get the frames per second (fps) and frame size from the original video
      fps = cap.get(cv2.CAP_PROP_FPS)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

      # Define the codec and create VideoWriter object
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec based on your needs
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

      # Read frames based on frame_ids
      for i in range(0, len(frame_ids), 5):
          frame_id = frame_ids[i]
          # Set the video capture to the specific frame
          cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
          ret, frame = cap.read()
          if ret:
              out.write(frame)  # Write the frame to the output video
          else:
              print(f"Frame {frame_id} could not be read.")

      # Release everything
      cap.release()
      out.release()
      print(f"Video created successfully at {output_path}")

  # Example usage

  create_video_from_frames(vid_name, frame_ids, 'person_frames/{vid_name}.mp4')
