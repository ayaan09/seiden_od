import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import time

class SeidenVDBMS:
    def __init__(self, video_path: str, oracle_model, segment_size: int = 30):
        """
        Initialize Seiden VDBMS
        
        Args:
            video_path: Path to the video file
            oracle_model: Pre-trained object detection model (e.g., YOLOv5)
            segment_size: Number of frames in each video segment
        """
        self.video_path = video_path
        self.oracle_model = oracle_model
        self.segment_size = segment_size
        
        # Initialize video properties
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Index storage
        self.frame_index = {}  # Stores sampled frame results
        self.segments = []     # List of video segments
        
        # Initialize segments
        self._initialize_segments()

    #Use Iframe here
    def _initialize_segments(self):
        """Divide video into segments"""
        num_segments = self.total_frames // self.segment_size
        self.segments = [(i * self.segment_size, 
                         min((i + 1) * self.segment_size, self.total_frames))
                        for i in range(num_segments)]

    def build_index(self, sampling_rate: float = 0.1):
        """
        Build initial index using I-frames
        
        Args:
            sampling_rate: Fraction of frames to sample for initial index
        """
        num_samples = int(self.total_frames * sampling_rate)
        
        # Sample frames uniformly
        sample_indices = np.linspace(0, self.total_frames-1, num_samples, dtype=int)
        
        for frame_idx in sample_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                # Run oracle model on frame
                results = self._run_oracle_model(frame)
                self.frame_index[frame_idx] = results

    def _run_oracle_model(self, frame):
        """
        Run oracle model on a single frame
        
        Args:
            frame: Video frame
        Returns:
            Detection results
        """
        with torch.no_grad():
            results = self.oracle_model(frame)[0]  # YOLOv8 returns a Results object
            
            # Convert YOLOv8 results to desired format
            detections = []
            for box in results.boxes:
                detection = {
                    'class': results.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy()  # Convert to numpy array
                }
                detections.append(detection)
        
        return detections

    def temporal_interpolation(self):
        """
        Perform temporal interpolation between sampled frames
        """
        frame_indices = sorted(self.frame_index.keys())
        
        for i in range(len(frame_indices) - 1):
            start_idx = frame_indices[i]
            end_idx = frame_indices[i + 1]
            
            # Skip if frames are consecutive
            if end_idx - start_idx <= 1:
                continue
                
            start_labels = self.frame_index[start_idx]
            end_labels = self.frame_index[end_idx]
            
            # Interpolate for frames between start and end
            for frame_idx in range(start_idx + 1, end_idx):
                alpha = (frame_idx - start_idx) / (end_idx - start_idx)
                interpolated = self._interpolate_labels(start_labels, end_labels, alpha)
                self.frame_index[frame_idx] = interpolated


    def _interpolate_labels(self, start_labels: List, end_labels: List, alpha: float) -> List:
        """
        Interpolate between two sets of labels
        
        Args:
            start_labels: Starting frame labels
            end_labels: Ending frame labels
            alpha: Interpolation factor (0 to 1)
        Returns:
            Interpolated labels
        """
        # Helper function to interpolate bounding boxes
        def interpolate_bbox(bbox1, bbox2, alpha):
            return [
                bbox1[0] * (1 - alpha) + bbox2[0] * alpha,
                bbox1[1] * (1 - alpha) + bbox2[1] * alpha,
                bbox1[2] * (1 - alpha) + bbox2[2] * alpha,
                bbox1[3] * (1 - alpha) + bbox2[3] * alpha
            ]
        
        # Match detections between frames based on class
        interpolated_labels = []
        
        # Group detections by class
        start_by_class = {}
        end_by_class = {}
        
        for det in start_labels:
            if det['class'] not in start_by_class:
                start_by_class[det['class']] = []
            start_by_class[det['class']].append(det)
            
        for det in end_labels:
            if det['class'] not in end_by_class:
                end_by_class[det['class']] = []
            end_by_class[det['class']].append(det)
        
        # Interpolate for each class
        all_classes = set(start_by_class.keys()) | set(end_by_class.keys())
        
        for class_name in all_classes:
            start_dets = start_by_class.get(class_name, [])
            end_dets = end_by_class.get(class_name, [])
            
            # Handle different numbers of detections
            num_interpolated = round((1 - alpha) * len(start_dets) + alpha * len(end_dets))
            
            # Match detections (simple matching by order - could be improved with IoU matching)
            for i in range(num_interpolated):
                if i < len(start_dets) and i < len(end_dets):
                    # Interpolate between matching detections
                    start_det = start_dets[i]
                    end_det = end_dets[i]
                    
                    interpolated_det = {
                        'class': class_name,
                        'confidence': start_det['confidence'] * (1 - alpha) + end_det['confidence'] * alpha,
                        'bbox': interpolate_bbox(start_det['bbox'], end_det['bbox'], alpha)
                    }
                    interpolated_labels.append(interpolated_det)
                elif i < len(start_dets):
                    # Keep start detection with reduced confidence
                    interpolated_labels.append({
                        'class': class_name,
                        'confidence': start_dets[i]['confidence'] * (1 - alpha),
                        'bbox': start_dets[i]['bbox'].copy()
                    })
                else:
                    # Keep end detection with reduced confidence
                    interpolated_labels.append({
                        'class': class_name,
                        'confidence': end_dets[i]['confidence'] * alpha,
                        'bbox': end_dets[i]['bbox'].copy()
                    })
        
        return interpolated_labels

    def execute_query(self, query_type: str, query_params: Dict):
        """
        Execute a query on the video
        
        Args:
            query_type: Type of query ('aggregate' or 'retrieval')
            query_params: Query parameters
        """
        if query_type == 'aggregate':
            return self._execute_aggregate_query(query_params)
        elif query_type == 'retrieval':
            return self._execute_retrieval_query(query_params)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
    
    def _execute_aggregate_query(self, query_params: Dict):
        """
        Execute an aggregate query
        
        Args:
            query_params: {
                'object_class': str,
                'aggregate_function': str,
                'confidence_threshold': float,
                'error_bound': float
            }
        Returns:
            Aggregate result
        """
        object_class = query_params['object_class']
        agg_function = query_params['aggregate_function']
        confidence_threshold = query_params['confidence_threshold']
        error_bound = query_params['error_bound']

        # Multi-arm bandit based sampling
        additional_samples = self._sample_frames_mab(query_params)
        
        # Process additional samples
        for frame_idx in additional_samples:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                results = self._run_oracle_model(frame)
                self.frame_index[frame_idx] = results

        # Update temporal interpolation
        self.temporal_interpolation()

        # Compute aggregate
        if agg_function.upper() == 'AVG':
            return self._compute_average(object_class)
        elif agg_function.upper() == 'COUNT':
            return self._compute_count(object_class)
        else:
            raise ValueError(f"Unsupported aggregate function: {agg_function}")

    def _execute_retrieval_query(self, query_params: Dict):
        """
        Execute a retrieval query
        
        Args:
            query_params: {
                'object_class': str,
                'condition': str,
                'threshold': float,
                'precision_threshold': float
            }
        Returns:
            List of frame IDs matching the query
        """
        object_class = query_params['object_class']
        condition = query_params['condition']
        threshold = query_params['threshold']
        precision_threshold = query_params['precision_threshold']

        # Multi-arm bandit based sampling
        additional_samples = self._sample_frames_mab(query_params)
        
        # Process additional samples
        for frame_idx in additional_samples:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                results = self._run_oracle_model(frame)
                self.frame_index[frame_idx] = results

        # Update temporal interpolation
        self.temporal_interpolation()

        # Retrieve matching frames
        matching_frames = []
        for frame_idx, results in self.frame_index.items():
            if self._evaluate_condition(results, object_class, condition, threshold):
                matching_frames.append(frame_idx)

        return matching_frames

    def _sample_frames_mab(self, query_params: Dict) -> List[int]:
        """
        Multi-arm bandit based frame sampling
        
        Args:
            query_params: Query parameters
        Returns:
            List of frame indices to sample
        """
        class Arm:
            def __init__(self, segment):
                self.segment = segment
                self.rewards = []
                self.pulls = 0

        arms = [Arm(segment) for segment in self.segments]
        total_budget = int(self.total_frames * 0.1)  # Sample 10% additional frames
        samples = []

        while len(samples) < total_budget:
            # Exploration phase
            unexplored_arms = [arm for arm in arms if arm.pulls == 0]
            if unexplored_arms:
                arm = np.random.choice(unexplored_arms)
                frame_idx = self._sample_frame_from_segment(arm.segment)
                reward = self._compute_reward(frame_idx, query_params)
                arm.rewards.append(reward)
                arm.pulls += 1
                samples.append(frame_idx)
                continue

            # Exploitation phase
            ucb_values = []
            total_pulls = sum(arm.pulls for arm in arms)
            
            for arm in arms:
                avg_reward = np.mean(arm.rewards) if arm.rewards else 0
                exploration_term = np.sqrt(2 * np.log(total_pulls) / arm.pulls)
                ucb = avg_reward + exploration_term
                ucb_values.append(ucb)

            best_arm = arms[np.argmax(ucb_values)]
            frame_idx = self._sample_frame_from_segment(best_arm.segment)
            reward = self._compute_reward(frame_idx, query_params)
            best_arm.rewards.append(reward)
            best_arm.pulls += 1
            samples.append(frame_idx)

        return samples

    def _sample_frame_from_segment(self, segment: Tuple[int, int]) -> int:
        """Sample a frame from the given segment"""
        start, end = segment
        return np.random.randint(start, end)

    def _compute_reward(self, frame_idx: int, query_params: Dict) -> float:
        """
        Compute reward for a sampled frame
        
        Args:
            frame_idx: Frame index
            query_params: Query parameters
        Returns:
            Reward value
        """
        frame_indices = np.array(list(self.frame_index.keys()))
        
        # Find previous and next indices
        prev_indices = frame_indices[frame_indices < frame_idx]
        next_indices = frame_indices[frame_indices > frame_idx]
        
        if len(prev_indices) == 0 or len(next_indices) == 0:
            return 0.0
            
        prev_idx = prev_indices.max()
        next_idx = next_indices.min()
        
        prev_results = self.frame_index[prev_idx]
        next_results = self.frame_index[next_idx]

        # Compute label difference
        label_diff = self._compute_label_difference(prev_results, next_results)
        
        # Higher reward for frames with larger label differences
        return label_diff
    def _compute_label_difference(self, results1: List, results2: List) -> float:
        """
        Compute difference between two sets of detection results using numpy arrays
        
        Args:
            results1: First set of detection results
            results2: Second set of detection results
        Returns:
            Difference score
        """
        # Get all unique classes from both results
        classes = set()
        for result in results1 + results2:
            classes.add(result['class'])
        classes = sorted(list(classes))
        
        # Create class-count vectors
        def create_count_vector(results):
            counts = np.zeros(len(classes))
            for result in results:
                idx = classes.index(result['class'])
                counts[idx] += 1
            return counts
        
        vec1 = create_count_vector(results1)
        vec2 = create_count_vector(results2)
        
        # Compute normalized difference
        if len(classes) > 0:
            return np.mean(np.abs(vec1 - vec2))
        return 0.0

    def _compute_average(self, object_class: str) -> float:
        """Compute average count of specified object"""
        counts = []
        for results in self.frame_index.values():
            count = self._count_objects(results, object_class)
            counts.append(count)
        return np.mean(counts)

    def _compute_count(self, object_class: str) -> int:
        """Compute total count of specified object"""
        total_count = 0
        for results in self.frame_index.values():
            count = self._count_objects(results, object_class)
            total_count += count
        return total_count

    def _count_objects(self, results, object_class: str) -> int:
        """Count objects of specified class in detection results"""
        # Implement based on specific oracle model output format
        return len([det for det in results if det['class'] == object_class])

    def _evaluate_condition(self, results, object_class: str, 
                          condition: str, threshold: float) -> bool:
        """Evaluate if detection results meet the query condition"""
        count = self._count_objects(results, object_class)
        
        if condition == '>':
            return count > threshold
        elif condition == '>=':
            return count >= threshold
        elif condition == '<':
            return count < threshold
        elif condition == '<=':
            return count <= threshold
        elif condition == '=':
            return count == threshold
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        
from ultralytics import YOLO

# Initialize YOLOv8 model
yolo_model = YOLO('yolov8s.pt') 

# Initialize Seiden VDBMS
vdbms = SeidenVDBMS(
    video_path='video3_out.mp4',
    oracle_model=yolo_model
)

# Build initial index
vdbms.build_index(sampling_rate=0.1)



# Execute a retrieval query
retrieval_query_params = {
    'object_class': 'person',
    'condition': '>',
    'threshold': 0,
    'precision_threshold': 0.95
}
truck_frames = vdbms.execute_query('retrieval', retrieval_query_params)
print(truck_frames)