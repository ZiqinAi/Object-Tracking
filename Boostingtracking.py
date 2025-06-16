import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cosine
from skimage import feature
import random
import warnings
warnings.filterwarnings('ignore')

class BoostingTracker:
    def __init__(self, num_classifiers=10, learning_rate=0.15, max_depth=2):
        """初始化Boosting跟踪器 """
        self.num_classifiers = num_classifiers          # 弱分类器数量
        self.learning_rate = learning_rate              # 学习率
        self.max_depth = max_depth                      # 决策树最大深度

        # 分类器相关
        self.classifiers = []                           # 弱分类器列表
        self.classifier_weights = []                    # 分类器权重
        
        # 跟踪状态
        self.initialized = False                        # 初始化标志
        self.current_bbox = None                        # 当前边界框
        self.search_window_scale = 1.6                  
        self.update_rate = 0.08                         # 更新频率
        
        # 特征相关
        self.feature_dim = 64                           # 特征维度
        self.patch_size = (32, 32)                      # patch尺寸
        
        # 采样相关
        self.positive_radius = 3                        
        self.negative_radius = 30                       
        self.num_positive_samples = 20                  
        self.num_negative_samples = 40                  
        
        # 稳定性相关
        self.confidence_threshold = 0.25                # 阈值 
        self.template_features = None                   # 模板特征
        self.template_update_rate = 0.02                # 模板更新率
        self.lost_count = 0                             # 丢失计数
        self.max_lost_frames = 8                        # 最大丢失帧数                         
        self.last_good_bbox = None                      # 上次好的边界框   
        
        # 尺寸稳定性相关
        self.initial_size = None                        # 初始目标大小
        self.size_history = []                          # 尺寸历史
        self.max_size_change_ratio = 0.3                # 最大尺寸变化比例
        self.min_size_ratio = 0.3                       # 最小尺寸比例
        self.max_size_ratio = 3.0                       # 最大尺寸比例
        
        # 多尺度相关
        self.scale_factors = [0.95, 0.98, 1.0, 1.02, 1.05]  # 尺度选项
        
    def is_valid_size(self, w, h):
        """检查尺寸是否合理"""
        if self.initial_size is None:
            return w > 8 and h > 8
        
        init_w, init_h = self.initial_size
        init_area = init_w * init_h
        current_area = w * h
        
        # 检查面积变化
        area_ratio = current_area / init_area
        if area_ratio < self.min_size_ratio**2 or area_ratio > self.max_size_ratio**2:
            return False
        
        # 检查宽高比变化
        init_aspect = init_w / init_h
        current_aspect = w / h
        aspect_ratio = max(init_aspect/current_aspect, current_aspect/init_aspect)
        if aspect_ratio > 2.0:  # 宽高比变化不能太大
            return False
        
        return True
    
    def constrain_bbox_size(self, bbox):
        """约束边界框尺寸"""
        x, y, w, h = bbox
        
        if self.initial_size is None:
            return bbox
        
        init_w, init_h = self.initial_size
        
        # 计算尺寸约束
        min_w = int(init_w * self.min_size_ratio)
        max_w = int(init_w * self.max_size_ratio)
        min_h = int(init_h * self.min_size_ratio)
        max_h = int(init_h * self.max_size_ratio)
        
        # 应用约束
        w = max(min_w, min(w, max_w))
        h = max(min_h, min(h, max_h))
        
        # 保持宽高比相对稳定
        if len(self.size_history) > 0:
            avg_w = np.mean([s[0] for s in self.size_history[-5:]])  # 最近5帧平均
            avg_h = np.mean([s[1] for s in self.size_history[-5:]])
            
            # 平滑尺寸变化
            smooth_factor = 0.7
            w = int(smooth_factor * avg_w + (1 - smooth_factor) * w)
            h = int(smooth_factor * avg_h + (1 - smooth_factor) * h)
        
        return (x, y, w, h)
    
    def extract_features(self, image, bbox):
        """提取特征"""
        x, y, w, h = bbox
        
        # 边界检查和patch提取
        x = max(0, min(x, image.shape[1] - w))
        y = max(0, min(y, image.shape[0] - h))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return np.zeros(self.feature_dim)
        
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        patch_resized = cv2.resize(patch, self.patch_size)
        
        features = []
        
        # HOG 特征
        hog_features = feature.hog(
            patch_resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True
        )
        features.extend(hog_features)
        
        # LBP 特征
        lbp = feature.local_binary_pattern(patch_resized, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        features.extend(lbp_hist / (np.sum(lbp_hist) + 1e-8))
        
        # 基本统计
        stats = [
            np.mean(patch_resized) / 255.0,
            np.std(patch_resized) / 255.0,
            np.median(patch_resized) / 255.0,
        ]
        features.extend(stats)
        
        # 确保维度
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
        else:
            features = features[:self.feature_dim]
        
        return features / (np.linalg.norm(features) + 1e-8)
    
    def generate_samples(self, image, center_bbox, positive=True):
        """生成训练样本"""
        x, y, w, h = center_bbox
        samples = []
        
        if positive:
            num_samples = self.num_positive_samples
            radius = self.positive_radius
        else:
            num_samples = self.num_negative_samples
            radius = self.negative_radius
        
        sample_count = 0
        attempts = 0
        max_attempts = num_samples * 2
        
        while sample_count < num_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机偏移
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            
            # 对于负样本，确保不与正样本重叠
            if not positive:
                if abs(dx) <= self.positive_radius and abs(dy) <= self.positive_radius:
                    continue
            
            # 尺度变化
            if positive:
                scale = random.choice([0.98, 1.0, 1.02])  # 正样本尺度变化更小
            else:
                scale = random.choice(self.scale_factors)
            
            new_w = max(8, int(w * scale))
            new_h = max(8, int(h * scale))
            
            # 检查尺寸合理性
            if not self.is_valid_size(new_w, new_h):
                continue
            
            new_x = x + dx - (new_w - w) // 2
            new_y = y + dy - (new_h - h) // 2
            
            # 确保采样框在图像范围内
            new_x = max(0, min(new_x, image.shape[1] - new_w))
            new_y = max(0, min(new_y, image.shape[0] - new_h))
            new_w = min(new_w, image.shape[1] - new_x)
            new_h = min(new_h, image.shape[0] - new_y)
            
            if new_w > 5 and new_h > 5:
                samples.append((new_x, new_y, new_w, new_h))
                sample_count += 1
        
        return samples
    
    def train_classifier(self, image, positive_samples, negative_samples):
        """训练分类器"""
        X = []
        y = []
        
        # 提取正样本特征
        for bbox in positive_samples:
            features = self.extract_features(image, bbox)
            X.append(features)
            y.append(1)
        
        # 提取负样本特征
        for bbox in negative_samples:
            features = self.extract_features(image, bbox)
            X.append(features)
            y.append(0)
        
        if len(X) < 4:
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # 检查数据平衡
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        if pos_count == 0 or neg_count == 0:
            return False
        
        try:
            # 使用分类器参数
            classifier = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=3,min_samples_leaf=1),
                n_estimators=self.num_classifiers,
                learning_rate=self.learning_rate,
                random_state=42
            )
            classifier.fit(X, y)
            
            # 更新分类器
            if not self.classifiers:
                self.classifiers = [classifier]
                self.classifier_weights = [1.0]
            else:
                score = classifier.score(X, y)
                weight = max(0.2, score)
                
                self.classifiers.append(classifier)
                self.classifier_weights.append(weight)
                
                # 保持最多3个分类器
                if len(self.classifiers) > 3:
                    min_idx = np.argmin(self.classifier_weights)
                    self.classifiers.pop(min_idx)
                    self.classifier_weights.pop(min_idx)
            
            return True
            
        except Exception as e:
            print(f"训练分类器失败: {e}")
            return False
    
    def predict_with_confidence(self, image, bbox):
        """预测函数"""
        if not self.classifiers:
            return 0.0, 0.0
        
        features = self.extract_features(image, bbox)
        features = features.reshape(1, -1)
        
        scores = []
        weights = []
        
        for classifier, weight in zip(self.classifiers, self.classifier_weights):
            try:
                proba = classifier.predict_proba(features)
                if len(proba[0]) >= 2:
                    score = proba[0][1]
                    scores.append(score)
                    weights.append(weight)
            except:
                continue
        
        if not scores:
            return 0.0, 0.0
        
        # 加权平均
        weighted_score = np.average(scores, weights=weights)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else weighted_score
        
        # 模板匹配
        if self.template_features is not None:
            template_similarity = 1.0 - cosine(features.flatten(), self.template_features)
            weighted_score = 0.8 * weighted_score + 0.2 * template_similarity
        
        # 尺寸合理性惩罚
        _, _, w, h = bbox
        if not self.is_valid_size(w, h):
            weighted_score *= 0.5  # 对不合理尺寸进行惩罚
        
        return weighted_score, confidence
    
    def init(self, image, bbox):
        """初始化跟踪器"""
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            self.current_bbox = bbox
            self.last_good_bbox = bbox
            self.initial_size = (bbox[2], bbox[3])  # 记录初始尺寸
            self.size_history = [(bbox[2], bbox[3])]
            
            # 生成训练样本
            positive_samples = self.generate_samples(image, bbox, positive=True)
            negative_samples = self.generate_samples(image, bbox, positive=False)
            
            # 训练初始分类器
            success = self.train_classifier(image, positive_samples, negative_samples)
            
            if success:
                self.template_features = self.extract_features(image, bbox)
                self.initialized = True
                self.lost_count = 0
                print("优化Boosting跟踪器初始化成功")
                return True
            else:
                print("优化Boosting跟踪器初始化失败")
                return False
                
        except Exception as e:
            print(f"跟踪器初始化失败: {e}")
            return False
    
    def update(self, image):
        """更新跟踪器"""
        if not self.initialized or self.current_bbox is None:
            return False, None
        
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            x, y, w, h = self.current_bbox
            
            # 搜索窗口
            scale_factor = min(1.8, self.search_window_scale * (1 + self.lost_count * 0.05))
            search_w = int(w * scale_factor)
            search_h = int(h * scale_factor)
            search_x = max(0, x - (search_w - w) // 2)
            search_y = max(0, y - (search_h - h) // 2)
            
            # 边界检查
            search_x = min(search_x, image.shape[1] - search_w)
            search_y = min(search_y, image.shape[0] - search_h)
            search_w = min(search_w, image.shape[1] - search_x)
            search_h = min(search_h, image.shape[0] - search_h)
            
            # 生成候选框
            candidates = []
            step_size = max(1, min(w, h) // 12)  # 步长
            
            # 首先在当前位置附近进行精细搜索
            for scale in self.scale_factors:
                cand_w = max(6, int(w * scale))
                cand_h = max(6, int(h * scale))
                
                # 检查候选尺寸合理性
                if not self.is_valid_size(cand_w, cand_h):
                    continue
                
                # 在搜索窗口内生成候选框
                for dx in range(0, max(1, search_w - cand_w + 1), step_size):
                    for dy in range(0, max(1, search_h - cand_h + 1), step_size):
                        candidate_bbox = (search_x + dx, search_y + dy, cand_w, cand_h)
                        candidates.append(candidate_bbox)
            
            # 如果没有合理的候选框，使用当前尺寸
            if not candidates:
                constrained_bbox = self.constrain_bbox_size((x, y, w, h))
                candidates = [constrained_bbox]
            
            # 评估候选框
            best_score = -1
            best_confidence = 0
            best_bbox = None
            
            for candidate in candidates:
                score, confidence = self.predict_with_confidence(image, candidate)
                combined_score = score * confidence
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_confidence = confidence
                    best_bbox = candidate
            
            # 阈值判断
            threshold = max(0.15, self.confidence_threshold - self.lost_count * 0.03)
            
            # 如果最佳得分低于阈值，认为跟踪失败
            if best_score < threshold:
                self.lost_count += 1
                if self.lost_count > self.max_lost_frames:
                    return False, None
                else:
                    # 返回约束后的上次好的边界框
                    safe_bbox = self.constrain_bbox_size(self.last_good_bbox)
                    return True, {
                        'x': safe_bbox[0],
                        'y': safe_bbox[1],
                        'width': safe_bbox[2],
                        'height': safe_bbox[3]
                    }
            
            # 约束最佳边界框的尺寸
            best_bbox = self.constrain_bbox_size(best_bbox)
            
            # 更新成功
            self.current_bbox = best_bbox
            self.last_good_bbox = best_bbox
            self.lost_count = 0
            
            # 更新尺寸历史
            self.size_history.append((best_bbox[2], best_bbox[3]))
            if len(self.size_history) > 10:  # 保持最近10帧的历史
                self.size_history.pop(0)
            
            # 在线学习
            if (random.random() < self.update_rate and 
                best_confidence > 0.6 and 
                len(self.size_history) >= 3):
                
                # 检查尺寸稳定性
                recent_sizes = self.size_history[-3:]
                size_variance = np.var([s[0]*s[1] for s in recent_sizes])
                avg_area = np.mean([s[0]*s[1] for s in recent_sizes])
                
                # 只有当尺寸相对稳定时才进行更新
                if size_variance / (avg_area + 1e-8) < 0.1:
                    positive_samples = self.generate_samples(image, best_bbox, positive=True)
                    negative_samples = self.generate_samples(image, best_bbox, positive=False)
                    
                    if self.train_classifier(image, positive_samples, negative_samples):
                        new_template = self.extract_features(image, best_bbox)
                        if self.template_features is not None:
                            self.template_features = ((1 - self.template_update_rate) * self.template_features + 
                                                    self.template_update_rate * new_template)
                        else:
                            self.template_features = new_template
            
            result_bbox = {
                'x': best_bbox[0],
                'y': best_bbox[1],
                'width': best_bbox[2],
                'height': best_bbox[3]
            }
            
            return True, result_bbox
            
        except Exception as e:
            print(f"跟踪器更新失败: {e}")
            self.lost_count += 1
            return False, None
