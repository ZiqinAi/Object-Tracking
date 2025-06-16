import cv2
import numpy as np
import random 
from collections import deque # 用于存储历史信息

class TLDTracker:
    def __init__(self, template_size=(32, 32), learning_rate=0.10, detection_threshold=0.30):
        """TLD跟踪器初始化"""
        self.template_size = template_size                  # 模板大小 (宽度, 高度)
        self.learning_rate = learning_rate                  # 学习率
        self.detection_threshold = detection_threshold      # 检测阈值
        
        # 跟踪器组件
        self.tracker = None                                 # 跟踪器
        self.detector = None                                # 检测器
        self.learning_module = None                         # 学习模块
        
        # 状态变量
        self.initialized = False                            # 是否初始化
        self.bbox = None                                    # 当前边界框
        self.template = None                                # 当前模板
        self.frame_idx = 0                                  # 当前帧索引
        
        # 用于存储历史信息
        self.positive_patches = deque(maxlen=100)           # 正样本
        self.negative_patches = deque(maxlen=300)           # 负样本
        self.confidence_history = deque(maxlen=10)          # 置信度历史
        
        # 光流跟踪器：winSize: 窗口大小；maxLevel: 金字塔层级；criteria: 迭代停止条件
        self.lk_params = dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # 用于光流跟踪的关键点
        self.keypoints = None                               # 光流关键点
        self.prev_gray = None                               # 上一帧灰度图像
        
        # 平滑策略相关参数
        self.bbox_history = deque(maxlen=5)                 # 边界框历史
        self.smooth_factor = 0.7                            # 平滑因子 (0-1)
        self.jump_threshold = 0.7                           # 跳跃检测阈值
        self.consecutive_detections = 0                     # 连续检测次数
        self.min_consecutive_for_jump = 5                   # 最小连续检测次数
        
         # P-N学习参数
        self.p_expert_activated = False                     # P-Expert激活
        self.n_expert_activated = False                     # N-Expert激活
        
    def init(self, frame, bbox):
        """初始化TLD跟踪器"""
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            self.bbox = bbox
            x, y, w, h = bbox
            
            # 验证边界框
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print("边界框超出图像范围")
                return False
            
            if w <= 0 or h <= 0:
                print("边界框尺寸无效")
                return False
            
            # 提取初始模板
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                print("ROI为空")
                return False
            
            self.template = cv2.resize(roi, self.template_size)
            
            # 初始化光流跟踪的关键点
            self._init_keypoints(gray, bbox)
            
            # 收集正负样本
            self._collect_initial_samples(gray, bbox)
            
            # 初始化各个组件
            self.tracker = MedianFlowTracker()
            
            self.detector = VarianceDetector(self.template_size, self.detection_threshold)
            self.learning_module = OnlineLearning(self.learning_rate, self.detector)
            
            # 训练初始检测器
            self.detector.train(self.positive_patches, self.negative_patches)
            
            self.prev_gray = gray.copy()
            self.initialized = True
            self.frame_idx = 0
            
            # 初始化边界框历史
            self.bbox_history.append(bbox)
            
            print("TLD跟踪器初始化成功")
            return True
            
        except Exception as e:
            print(f"TLD跟踪器初始化失败: {e}")
            return False
    
    def _init_keypoints(self, gray, bbox):
        """初始化用于光流跟踪的关键点"""
        x, y, w, h = bbox
        
        # 在目标区域内生成关键点网格
        points = []
        for i in range(3, w-3, max(1, w//10)):
            for j in range(3, h-3, max(1, h//10)):
                points.append([x + i, y + j])
        
        if len(points) > 0:
            self.keypoints = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            # 如果区域太小，使用中心点
            center_x, center_y = x + w//2, y + h//2
            self.keypoints = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
    
    def _collect_initial_samples(self, gray, bbox):
        """收集初始的正负样本"""
        x, y, w, h = bbox
        
        # 正样本：目标区域
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            positive_patch = cv2.resize(roi, self.template_size)
            self.positive_patches.append(positive_patch)
        
        # 负样本：周围区域，但是不与正样本重叠
        for _ in range(20):
            # 随机生成负样本位置
            neg_x = random.randint(0, gray.shape[1] - w)
            neg_y = random.randint(0, gray.shape[0] - h)
            
            # 确保负样本不与正样本重叠
            if abs(neg_x - x) > w//2 or abs(neg_y - y) > h//2:
                neg_roi = gray[neg_y:neg_y+h, neg_x:neg_x+w]
                if neg_roi.size > 0:
                    negative_patch = cv2.resize(neg_roi, self.template_size)
                    self.negative_patches.append(negative_patch)
    
    def _calculate_bbox_distance(self, bbox1, bbox2):
        """计算两个边界框之间的距离"""
        if bbox1 is None or bbox2 is None:
            return float('inf')
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算中心点距离
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 归一化距离（相对于边界框大小）
        avg_size = (w1 + h1 + w2 + h2) / 4
        normalized_distance = distance / avg_size if avg_size > 0 else float('inf')
        
        return normalized_distance
    
    def _smooth_bbox(self, current_bbox, new_bbox, confidence):
        """平滑边界框位置"""
        if current_bbox is None:
            return new_bbox
        
        if new_bbox is None:
            return current_bbox
        
        # 计算距离
        distance = self._calculate_bbox_distance(current_bbox, new_bbox)
        
        # 如果距离很小，直接使用新的边界框
        if distance < 0.1:
            return new_bbox
        
        # 如果距离很大，检查是否应该跳跃
        if distance > self.jump_threshold:
            # 只有在连续多次检测到相似位置时才允许跳跃
            if confidence > 0.7 and self.consecutive_detections >= self.min_consecutive_for_jump:
                return new_bbox
            else:
                # 使用更强的平滑
                smooth_factor = min(self.smooth_factor * 0.5, 0.1)
            
        else:
            # 正常情况下的平滑
            smooth_factor = self.smooth_factor * confidence
        
        # 加权平均
        x1, y1, w1, h1 = current_bbox
        x2, y2, w2, h2 = new_bbox
        
        # 对位置和大小进行平滑
        smooth_x = x1 + smooth_factor * (x2 - x1)
        smooth_y = y1 + smooth_factor * (y2 - y1)
        smooth_w = w1 + smooth_factor * (w2 - w1)
        smooth_h = h1 + smooth_factor * (h2 - h1)
        
        return (int(smooth_x), int(smooth_y), int(smooth_w), int(smooth_h))
    
    def _validate_bbox_consistency(self, bbox):
        """验证边界框与历史的一致性"""
        if len(self.bbox_history) == 0:
            return True
        
        # 计算与最近几个历史边界框的平均距离
        distances = []
        for hist_bbox in list(self.bbox_history)[-3:]:  # 检查最近3帧
            distance = self._calculate_bbox_distance(bbox, hist_bbox)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # 如果平均距离过大，认为不一致
        return avg_distance < self.jump_threshold * 1.5
    
    def update(self, frame):
        """更新TLD跟踪器"""
        if not self.initialized:
            return False, None
        
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            self.frame_idx += 1
            
            # 重置P-N专家状态
            self.p_expert_activated = False
            self.n_expert_activated = False
            
            # 跟踪模块
            track_bbox, track_confidence = self._track(gray)
            
            # 检测模块
            detect_bbox, detect_confidence = self._detect(gray)
            
            # 学习模块（融合跟踪和检测结果）
            raw_bbox, raw_confidence = self._learn(gray, track_bbox, track_confidence, detect_bbox, detect_confidence)
            
            # 应用平滑策略
            if raw_bbox is not None and raw_confidence > self.detection_threshold:
                # 验证边界框一致性
                if self._validate_bbox_consistency(raw_bbox):
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0
                
                # 平滑边界框
                smooth_bbox = self._smooth_bbox(self.bbox, raw_bbox, raw_confidence)
                
                # 更新状态
                self.bbox = smooth_bbox
                self.bbox_history.append(smooth_bbox)
                self.confidence_history.append(raw_confidence)
                
                # 更新模板
                self._update_template(gray, smooth_bbox)
                
                # 更新关键点
                self._update_keypoints(gray, smooth_bbox)
                
                self.prev_gray = gray.copy()
                
                # 返回字典格式的边界框
                return True, {
                    'x': int(smooth_bbox[0]),
                    'y': int(smooth_bbox[1]),
                    'width': int(smooth_bbox[2]),
                    'height': int(smooth_bbox[3])
                }
            else:
                # 跟踪失败时重置连续检测计数
                self.consecutive_detections = 0
                self.prev_gray = gray.copy()
                return False, None
                
        except Exception as e:
            print(f"TLD更新失败: {e}")
            return False, None
    
    def _track(self, gray):
        """跟踪模块：使用光流法跟踪关键点"""
        if self.prev_gray is None or self.keypoints is None:
            return None, 0.0
        
        try:
            # 使用光流法跟踪关键点
            new_keypoints, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.keypoints, None, **self.lk_params)
            
            # 过滤出成功跟踪的点
            good_new = new_keypoints[status == 1]
            good_old = self.keypoints[status == 1]
            
            if len(good_new) < 3:
                return None, 0.0
            
            # 计算运动向量的中位数
            motion_vectors = good_new - good_old
            median_motion = np.median(motion_vectors, axis=0)
            
            # 更新边界框位置
            x, y, w, h = self.bbox
            new_x = int(x + median_motion[0])
            new_y = int(y + median_motion[1])
            
            # 边界检查
            new_x = max(0, min(new_x, gray.shape[1] - w))
            new_y = max(0, min(new_y, gray.shape[0] - h))
            
            track_bbox = (new_x, new_y, w, h)
            
            # 计算跟踪置信度（基于运动向量的一致性）
            motion_consistency = 1.0 / (1.0 + np.std(motion_vectors))
            track_confidence = min(0.9, motion_consistency)
            
            return track_bbox, track_confidence
            
        except Exception as e:
            print(f"光流跟踪失败: {e}")
            return None, 0.0
    
    def _detect(self, gray):
        """检测模块：在整个图像中搜索目标"""
        if self.detector is None:
            return None, 0.0
        
        try:
            # 使用滑动窗口检测
            best_bbox = None
            best_confidence = 0.0
            
            x, y, w, h = self.bbox
            
            # 在目标周围区域搜索
            search_radius = max(w, h)
            start_x = max(0, x - search_radius)
            end_x = min(gray.shape[1] - w, x + search_radius)
            start_y = max(0, y - search_radius)
            end_y = min(gray.shape[0] - h, y + search_radius)
            
            # 降低搜索密度以提高效率
            step_size = max(5, min(w, h) // 6)
            
            for sx in range(start_x, end_x, step_size):
                for sy in range(start_y, end_y, step_size):
                    roi = gray[sy:sy+h, sx:sx+w]
                    if roi.size > 0:
                        confidence = self.detector.classify(roi)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_bbox = (sx, sy, w, h)
            
            return best_bbox, best_confidence
            
        except Exception as e:
            print(f"检测模块失败: {e}")
            return None, 0.0
    
    def _learn(self, gray, track_bbox, track_confidence, detect_bbox, detect_confidence):
        """学习模块：P-N学习机制"""
        # 选择最佳结果
        final_bbox = None
        final_confidence = 0.0
        
        # 优先考虑跟踪结果的稳定性
        if track_bbox is not None and track_confidence > 0.3:
            # 如果检测结果也存在，比较两者
            if detect_bbox is not None and detect_confidence > track_confidence + 0.2:
                # 检测结果明显更好时才使用检测结果
                distance = self._calculate_bbox_distance(track_bbox, detect_bbox)
                if distance < self.jump_threshold:  # 距离不能太远
                    final_bbox = detect_bbox
                    final_confidence = detect_confidence
                else:
                    final_bbox = track_bbox
                    final_confidence = track_confidence
            else:
                final_bbox = track_bbox
                final_confidence = track_confidence
        elif detect_bbox is not None:
            final_bbox = detect_bbox
            final_confidence = detect_confidence
        
        # =============== P-N 学习机制 ===============
        # P专家：跟踪成功但检测失败 -> 添加正样本
        if (track_bbox is not None and track_confidence > 0.6 and 
            (detect_bbox is None or detect_confidence < self.detection_threshold * 0.8)):
            self._add_positive_sample(gray, track_bbox)
            self.p_expert_activated = True
            print(f"帧 {self.frame_idx}: P专家激活 - 添加正样本")
        
        # N专家：检测到假阳性 -> 添加负样本
        if (detect_bbox is not None and detect_confidence > self.detection_threshold * 1.2 and
            track_bbox is not None and track_confidence > 0.6):
            distance = self._calculate_bbox_distance(track_bbox, detect_bbox)
            if distance > self.jump_threshold:  # 检测结果与跟踪结果不一致
                self._add_negative_sample(gray, detect_bbox)
                self.n_expert_activated = True
                print(f"帧 {self.frame_idx}: N专家激活 - 添加负样本")
        
        # 常规学习：当有可靠结果时更新模型
        if final_bbox is not None and final_confidence > self.detection_threshold:
            # 添加正样本
            self._add_positive_sample(gray, final_bbox)
            
            # 收集负样本
            if self.frame_idx % 5 == 0:  # 每5帧收集一次负样本
                self._collect_negative_samples(gray, final_bbox)
        
        # 更新检测器
        if self.p_expert_activated or self.n_expert_activated or (final_bbox is not None):
            if len(self.positive_patches) > 0 and len(self.negative_patches) > 0:
                self.detector.update(self.positive_patches, self.negative_patches)
                print(f"帧 {self.frame_idx}: 检测器更新 (正样本: {len(self.positive_patches)}, 负样本: {len(self.negative_patches)})")
        
        return final_bbox, final_confidence
    
    def _add_positive_sample(self, gray, bbox):
        """添加正样本"""
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            positive_patch = cv2.resize(roi, self.template_size)
            self.positive_patches.append(positive_patch)
    
    def _add_negative_sample(self, gray, bbox):
        """添加负样本"""
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            negative_patch = cv2.resize(roi, self.template_size)
            self.negative_patches.append(negative_patch)

    def _collect_negative_samples(self, gray, bbox):
        """收集负样本（远离目标区域）"""
        x, y, w, h = bbox
        
        for _ in range(3):  # 每次收集3个负样本
            neg_x = random.randint(0, gray.shape[1] - w)
            neg_y = random.randint(0, gray.shape[0] - h)
            
            # 确保负样本远离目标
            if abs(neg_x - x) > w or abs(neg_y - y) > h:
                neg_roi = gray[neg_y:neg_y+h, neg_x:neg_x+w]
                if neg_roi.size > 0:
                    negative_patch = cv2.resize(neg_roi, self.template_size)
                    self.negative_patches.append(negative_patch)

    
    def _update_template(self, gray, bbox):
        """更新模板"""
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            new_template = cv2.resize(roi, self.template_size)
            # 使用指数移动平均更新模板
            self.template = (1 - self.learning_rate) * self.template + self.learning_rate * new_template
    
    def _update_keypoints(self, gray, bbox):
        """更新关键点"""
        x, y, w, h = bbox
        
        # 重新生成关键点
        points = []
        for i in range(3, w-3, max(1, w//8)):
            for j in range(3, h-3, max(1, h//8)):
                points.append([x + i, y + j])
        
        if len(points) > 0:
            self.keypoints = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            center_x, center_y = x + w//2, y + h//2
            self.keypoints = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)


class MedianFlowTracker:
    """基于中值流的跟踪器"""
    def __init__(self):
        self.tracker = None
        self.initialized = False
    
    def init(self, frame, bbox):
        """初始化跟踪器"""
        self.tracker = cv2.legacy.TrackerMedianFlow_create()
        success = self.tracker.init(frame, tuple(bbox))
        self.initialized = success
        return success
    
    def track(self, frame):
        """跟踪当前帧"""
        if not self.initialized:
            return None, 0.0
        success, bbox = self.tracker.update(frame)
        if success:
            return bbox, 0.8
        else:
            return None, 0.0


class VarianceDetector:
    """基于方差的检测器"""
    def __init__(self, template_size, threshold):
        self.template_size = template_size
        self.threshold = threshold
        self.positive_mean = None
        self.negative_mean = None
        self.positive_var = None
        self.negative_var = None
    
    def train(self, positive_patches, negative_patches):
        """训练检测器"""
        if len(positive_patches) == 0 or len(negative_patches) == 0:
            return
        
        # 计算正样本的均值和方差
        pos_features = [self._extract_features(patch) for patch in positive_patches]
        self.positive_mean = np.mean(pos_features, axis=0)
        self.positive_var = np.var(pos_features, axis=0) + 1e-6
        
        # 计算负样本的均值和方差
        neg_features = [self._extract_features(patch) for patch in negative_patches]
        self.negative_mean = np.mean(neg_features, axis=0)
        self.negative_var = np.var(neg_features, axis=0) + 1e-6
    
    def update(self, positive_patches, negative_patches):
        """更新检测器"""
        self.train(positive_patches, negative_patches)
    
    def classify(self, patch):
        """分类给定的图像块"""
        if self.positive_mean is None or self.negative_mean is None:
            return 0.0
        
        patch_resized = cv2.resize(patch, self.template_size)
        features = self._extract_features(patch_resized)
        
        # 计算与正负样本的相似度
        pos_prob = self._gaussian_prob(features, self.positive_mean, self.positive_var)
        neg_prob = self._gaussian_prob(features, self.negative_mean, self.negative_var)
        
        # 返回归一化的置信度
        if pos_prob + neg_prob > 0:
            confidence = pos_prob / (pos_prob + neg_prob)
        else:
            confidence = 0.0
        
        return confidence
    
    def _extract_features(self, patch):
        """提取特征"""
        # 像素值的直方图
        hist = cv2.calcHist([patch], [0], None, [32], [0, 256])
        return hist.flatten()
    
    def _gaussian_prob(self, x, mean, var):
        """计算高斯概率"""
        diff = x - mean
        prob = np.exp(-0.5 * np.sum((diff ** 2) / var))
        return prob


class OnlineLearning:
    """在线学习模块"""
    def __init__(self, learning_rate, detector):
        self.learning_rate = learning_rate
        self.detector = detector
    
    def update_model(self, positive_samples, negative_samples):
        """
        根据正负样本更新检测器模型，调用VarianceDetector的update方法重新训练
        """
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            return
        self.detector.update(positive_samples, negative_samples)