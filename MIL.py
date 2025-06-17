import cv2
import numpy as np
from skimage.feature import hog
import threading
import time

class MILTracker:
    """
    多示例学习(Multiple Instance Learning)目标跟踪器
    使用HOG特征，针对实时性能优化的版本
    """
    def __init__(self, 
                 search_radius=25,      # 搜索半径
                 positive_radius=4,     # 正样本半径
                 negative_radius=65,    # 负样本半径
                 learning_rate=0.85,    # 学习率
                 max_samples=100,       # 最大样本数
                 update_interval=5,     # 更新间隔（帧数）
                 # 新增多尺度参数
                 enable_multiscale=True,    # 启用多尺度检测
                 scale_factors=[0.8, 0.9, 1.0, 1.1, 1.2],  # 尺度因子
                 scale_update_threshold=0.7,  # 尺度更新阈值
                 scale_penalty=0.95,          # 尺度变化惩罚因子
                 max_scale_change=0.3,        # 最大尺度变化比例
                 # HOG参数 - 针对实时性优化
                 orientations=6,        # 减少方向数（默认9）
                 pixels_per_cell=(8, 8), # 增大cell大小
                 cells_per_block=(2, 2), # 保持block大小
                 block_norm='L2-Hys',   # 归一化方法
                 feature_vector=True,   # 返回特征向量
                 transform_sqrt=True):  # 平方根变换
        
        # 基本参数
        self.search_radius = search_radius
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
        self.learning_rate = learning_rate
        self.max_samples = max_samples
        self.update_interval = update_interval
        
        # 新增多尺度检测参数
        self.enable_multiscale = enable_multiscale
        self.scale_factors = scale_factors
        self.scale_update_threshold = scale_update_threshold
        self.scale_penalty = scale_penalty
        self.max_scale_change = max_scale_change
        
        # 多尺度状态变量
        self.current_scale = 1.0
        self.scale_history = []
        self.optimal_scale_cache = {}  # 缓存最优尺度
        
        # HOG参数
        self.hog_params = {
            'orientations': orientations,
            'pixels_per_cell': pixels_per_cell,
            'cells_per_block': cells_per_block,
            'block_norm': block_norm,
            'feature_vector': feature_vector,
            'transform_sqrt': transform_sqrt,
            'visualize': False  # 实时应用中关闭可视化
        }
        
        # 分类器相关
        self.classifier = None
        self.feature_selector = None
        self.feature_dim = None
        
        # 目标状态
        self.bbox = None
        self.center = None
        self.target_size = None
        self.initialized = False
        
        # 历史信息（用于自适应）
        self.confidence_history = []
        self.scale_history = []
        self.frame_count = 0
        
        # 性能优化相关
        self.feature_cache = {}
        self.last_update_frame = 0
        self.hog_cache = {}  # HOG特征缓存
        
        # 多线程支持
        self.lock = threading.Lock()
        
        print("HOG-MIL跟踪器初始化完成")
    
    def init(self, frame, bbox):
        """
        初始化跟踪器
        Args:
            frame: 输入图像 (numpy array)
            bbox: 边界框 (x, y, width, height)
        Returns:
            bool: 初始化是否成功
        """
        try:
            with self.lock:
                # 验证输入
                if frame is None or len(frame.shape) < 2:
                    print("无效的输入图像")
                    return False
                
                x, y, w, h = bbox
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    print("无效的边界框")
                    return False
                
                if x + w > frame.shape[1] or y + h > frame.shape[0]:
                    print("边界框超出图像范围")
                    return False
                
                # 转换为灰度图像
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame.copy()
                
                # 初始化目标状态
                self.bbox = {'x': x, 'y': y, 'width': w, 'height': h}
                self.center = (x + w // 2, y + h // 2)
                self.target_size = (w, h)
                
                # 确保目标尺寸适合HOG特征提取
                self.target_size = self._adjust_size_for_hog(w, h)
                
                # 提取初始特征并训练分类器
                if not self._extract_initial_features(gray):
                    print("提取初始HOG特征失败")
                    return False
                
                self.initialized = True
                self.frame_count = 0
                print(f"HOG-MIL跟踪器初始化成功: {bbox}")
                return True
                
        except Exception as e:
            print(f"HOG-MIL跟踪器初始化错误: {e}")
            return False
    
    def _adjust_size_for_hog(self, w, h):
        """调整尺寸以适合HOG特征提取"""
        # 确保尺寸是pixels_per_cell的倍数
        cell_w, cell_h = self.hog_params['pixels_per_cell']
        
        # 向上取整到最近的cell倍数
        new_w = max(cell_w * 2, ((w + cell_w - 1) // cell_w) * cell_w)
        new_h = max(cell_h * 2, ((h + cell_h - 1) // cell_h) * cell_h)
        
        return (new_w, new_h)
    
    # 多尺度搜索
    def _multiscale_search(self, gray):
        """
        多尺度目标搜索
        Args:
            gray: 灰度图像
        Returns:
            tuple: (best_center, best_scale, confidence)
        """
        try:
            best_score = -np.inf
            best_center = None
            best_scale = self.current_scale
            
            cx, cy = self.center
            base_w, base_h = self.target_size
            
            # 搜索不同尺度
            for scale in self.scale_factors:
                # 计算当前尺度下的目标尺寸
                scaled_w = int(base_w * scale)
                scaled_h = int(base_h * scale)
                
                # 确保尺寸合理
                if scaled_w < 16 or scaled_h < 16 or scaled_w > gray.shape[1]//2 or scaled_h > gray.shape[0]//2:
                    continue
                
                # 调整尺寸以适合HOG
                scaled_w, scaled_h = self._adjust_size_for_hog(scaled_w, scaled_h)
                
                # 在当前尺度下搜索最佳位置
                scale_best_center, scale_best_score = self._search_at_scale(gray, (scaled_w, scaled_h))
                
                if scale_best_center is not None:
                    # 应用尺度变化惩罚
                    scale_penalty = self._calculate_scale_penalty(scale)
                    penalized_score = scale_best_score * scale_penalty
                    
                    if penalized_score > best_score:
                        best_score = penalized_score
                        best_center = scale_best_center
                        best_scale = scale
            
            # 转换得分为置信度
            confidence = 1.0 / (1.0 + np.exp(-best_score))
            
            return best_center, best_scale, confidence
            
        except Exception as e:
            print(f"多尺度搜索错误: {e}")
            return None, self.current_scale, 0.0


    # 尺度搜索
    def _search_at_scale(self, gray, target_size):
        """
        在指定尺度下搜索目标
        Args:
            gray: 灰度图像
            target_size: 目标尺寸 (width, height)
        Returns:
            tuple: (best_center, best_score)
        """
        try:
            best_score = -np.inf
            best_center = None
            
            cx, cy = self.center
            w, h = target_size
            
            # 生成搜索候选位置（根据尺度调整搜索范围）
            scale_ratio = max(w, h) / max(self.target_size)
            adjusted_radius = int(self.search_radius * scale_ratio)
            
            candidates = []
            step = max(2, int(4 * scale_ratio))  # 根据尺度调整搜索步长
            
            for dx in range(-adjusted_radius, adjusted_radius + 1, step):
                for dy in range(-adjusted_radius, adjusted_radius + 1, step):
                    new_cx = cx + dx
                    new_cy = cy + dy
                    
                    # 检查边界
                    if (new_cx - w//2 >= 0 and new_cy - h//2 >= 0 and
                        new_cx + w//2 < gray.shape[1] and new_cy + h//2 < gray.shape[0]):
                        candidates.append((new_cx, new_cy))
            
            # 评估候选位置
            for new_cx, new_cy in candidates:
                bbox = (new_cx - w//2, new_cy - h//2, w, h)
                features = self._extract_hog_features(gray, bbox)
                
                if features is not None:
                    score = self._classify(features)
                    if score > best_score:
                        best_score = score
                        best_center = (new_cx, new_cy)
            
            return best_center, best_score
            
        except Exception as e:
            print(f"尺度搜索错误: {e}")
            return None, -np.inf


    # 尺度惩罚计算
    def _calculate_scale_penalty(self, scale):
        """
        计算尺度变化惩罚
        Args:
            scale: 候选尺度
        Returns:
            float: 惩罚因子 (0-1)
        """
        try:
            # 计算与当前尺度的差异
            scale_diff = abs(scale - self.current_scale)
            
            # 应用指数衰减惩罚
            penalty = np.exp(-scale_diff / 0.2)  # 0.2是衰减常数
            
            # 限制最大尺度变化
            max_change = self.max_scale_change
            if scale_diff > max_change:
                penalty *= 0.5  # 大幅度尺度变化额外惩罚
            
            return max(0.1, penalty)  # 确保最小惩罚值
            
        except:
            return 1.0


    # 尺度更新
    def _update_scale(self, new_scale, confidence):
        """
        更新目标尺度
        Args:
            new_scale: 新检测到的尺度
            confidence: 检测置信度
        Returns:
            bool: 是否更新了尺度
        """
        try:
            # 只有在高置信度时才更新尺度
            if confidence < self.scale_update_threshold:
                return False
            
            # 记录尺度历史
            self.scale_history.append(new_scale)
            if len(self.scale_history) > 10:
                self.scale_history.pop(0)
            
            # 计算平滑后的尺度（使用指数移动平均）
            alpha = 0.3  # 平滑因子
            smooth_scale = alpha * new_scale + (1 - alpha) * self.current_scale
            
            # 限制尺度变化幅度
            scale_change = abs(smooth_scale - self.current_scale) / self.current_scale
            if scale_change > self.max_scale_change:
                # 限制变化幅度
                if smooth_scale > self.current_scale:
                    smooth_scale = self.current_scale * (1 + self.max_scale_change)
                else:
                    smooth_scale = self.current_scale * (1 - self.max_scale_change)
            
            # 更新尺度和目标尺寸
            old_scale = self.current_scale
            self.current_scale = smooth_scale
            
            # 更新目标尺寸
            base_w, base_h = int(self.target_size[0] / old_scale), int(self.target_size[1] / old_scale)
            new_w = int(base_w * self.current_scale)
            new_h = int(base_h * self.current_scale)
            
            # 调整尺寸以适合HOG
            self.target_size = self._adjust_size_for_hog(new_w, new_h)
            
            print(f"尺度更新: {old_scale:.3f} -> {self.current_scale:.3f}, 尺寸: {self.target_size}")
            return True
            
        except Exception as e:
            print(f"尺度更新错误: {e}")
            return False
    
    def update(self, frame):
        """
        更新跟踪器
        Args:
            frame: 当前帧图像
        Returns:
            tuple: (success, bbox_dict) 或 (False, None)
        """
        if not self.initialized:
            return False, None
            
        try:
            with self.lock:
                start_time = time.time()
                
                # 转换为灰度图像
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame.copy()
                
                self.frame_count += 1
                
                # 选择搜索策略
                if self.enable_multiscale and self.frame_count % 3 == 0:  # 每3帧进行一次多尺度搜索
                    new_center, new_scale, confidence = self._multiscale_search(gray)
                    
                    # 更新尺度
                    if new_center is not None and confidence > 0.5:
                        scale_updated = self._update_scale(new_scale, confidence)
                        if scale_updated:
                            print(f"多尺度检测更新尺度: {new_scale:.3f}")
                else:
                    # 常规单尺度搜索
                    new_center, confidence = self._search_target(gray)
                    new_scale = self.current_scale
                
                # 如果主搜索失败，尝试扩大搜索范围
                if new_center is None or confidence < 0.3:
                    new_center, confidence = self._extended_search(gray)
                    if new_center is None or confidence < 0.2:
                        print(f"跟踪失败，置信度: {confidence}")
                        return False, None
                
                # 更新目标位置
                self.center = new_center
                self._update_bbox()
                
                # 记录置信度历史
                self.confidence_history.append(confidence)
                if len(self.confidence_history) > 20:
                    self.confidence_history.pop(0)
                
                # 周期性更新分类器
                if self.frame_count % self.update_interval == 0:
                    self._update_classifier(gray)
                
                # 清理缓存防止内存泄漏
                if self.frame_count % 50 == 0:
                    self.hog_cache.clear()
                    self.optimal_scale_cache.clear()
                
                # 性能监控
                process_time = time.time() - start_time
                if process_time > 0.033:  # 超过30fps警告
                    print(f"HOG-MIL处理时间警告: {process_time:.3f}s")
                
                return True, self.bbox.copy()
                
        except Exception as e:
            print(f"HOG-MIL更新错误: {e}")
            return False, None
    
    def _extract_initial_features(self, gray):
        """提取初始HOG特征并训练分类器"""
        try:
            # 生成正负样本
            positive_samples = self._generate_samples(gray, self.center, self.positive_radius, 15)
            negative_samples = self._generate_samples(gray, self.center, self.negative_radius, 30)
            
            if len(positive_samples) == 0 or len(negative_samples) == 0:
                return False
            
            # 提取HOG特征
            pos_features = []
            neg_features = []
            
            for sample in positive_samples:
                feat = self._extract_hog_features(gray, sample)
                if feat is not None:
                    pos_features.append(feat)
            
            for sample in negative_samples:
                feat = self._extract_hog_features(gray, sample)
                if feat is not None:
                    neg_features.append(feat)
            
            if len(pos_features) == 0 or len(neg_features) == 0:
                return False
            
            # 确保所有特征维度一致
            feature_dims = [len(f) for f in pos_features + neg_features]
            if len(set(feature_dims)) > 1:
                print(f"HOG特征维度不一致: {set(feature_dims)}")
                # 找到最小维度
                min_dim = min(feature_dims)
                pos_features = [f[:min_dim] for f in pos_features]
                neg_features = [f[:min_dim] for f in neg_features]
            
            # 创建训练数据
            X = np.vstack([pos_features, neg_features])
            y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
            
            # 记录特征维度
            self.feature_dim = X.shape[1]
            
            # 特征选择（选择最具区分性的特征）
            self.feature_selector = self._select_features(X, y)
            
            # 确保特征选择器索引有效
            self.feature_selector = self.feature_selector[self.feature_selector < self.feature_dim]
            
            if len(self.feature_selector) == 0:
                print("没有有效的特征选择器")
                return False
            
            X_selected = X[:, self.feature_selector]
            
            # 训练简单的线性分类器
            self.classifier = self._train_classifier(X_selected, y)
            
            print(f"HOG特征维度: {self.feature_dim}, 选择特征数: {len(self.feature_selector)}")
            return True
            
        except Exception as e:
            print(f"提取初始HOG特征错误: {e}")
            return False
    
    def _generate_samples(self, gray, center, radius, num_samples):
        """生成训练样本"""
        samples = []
        cx, cy = center
        w, h = self.target_size
        
        # 确保样本在图像范围内
        for _ in range(num_samples * 2):  # 生成更多候选样本
            if len(samples) >= num_samples:
                break
                
            # 在圆形区域内随机采样
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            dx = int(r * np.cos(angle))
            dy = int(r * np.sin(angle))
            
            sample_x = cx + dx - w // 2
            sample_y = cy + dy - h // 2
            
            # 检查边界
            if (sample_x >= 0 and sample_y >= 0 and 
                sample_x + w < gray.shape[1] and sample_y + h < gray.shape[0]):
                samples.append((sample_x, sample_y, w, h))
        
        return samples
    
    def _extract_hog_features(self, gray, bbox):
        """提取HOG特征"""
        try:
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w >= gray.shape[1] or y + h >= gray.shape[0]:
                return None
            
            # 提取ROI
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
                return None
            
            # 检查缓存
            roi_key = (x, y, w, h, self.frame_count // 5)  # 缓存键包含位置和时间戳
            if roi_key in self.hog_cache:
                return self.hog_cache[roi_key]
            
            # 预处理ROI以提高HOG特征质量
            roi_processed = self._preprocess_roi(roi)
            
            # 提取HOG特征
            try:
                features = hog(roi_processed, **self.hog_params)
                
                # 缓存特征（限制缓存大小）
                if len(self.hog_cache) < 100:
                    self.hog_cache[roi_key] = features
                
                return features.astype(np.float32)
                
            except Exception as e:
                print(f"HOG特征提取内部错误: {e}")
                return None
            
        except Exception as e:
            print(f"HOG特征提取错误: {e}")
            return None
    
    def _preprocess_roi(self, roi):
        """预处理ROI以提高HOG特征质量"""
        try:
            # 直方图均衡化增强对比度
            roi_eq = cv2.equalizeHist(roi)
            
            # 高斯平滑去噪Q
            roi_smooth = cv2.GaussianBlur(roi_eq, (3, 3), 0.8)
            
            return roi_smooth
            
        except:
            return roi
    
    def _select_features(self, X, y):
        """选择最具区分性的HOG特征"""
        try:
            if X.shape[0] == 0 or X.shape[1] == 0:
                return np.arange(min(200, X.shape[1]))
            
            # 方差阈值过滤（去除低方差特征）
            var_threshold = 0.01
            feature_vars = np.var(X, axis=0)
            var_mask = feature_vars > var_threshold
            
            # 如果没有通过方差阈值的特征，降低阈值
            if np.sum(var_mask) == 0:
                var_threshold = 0.001
                var_mask = feature_vars > var_threshold
            
            # 如果仍然没有，直接使用所有特征
            if np.sum(var_mask) == 0:
                var_mask = np.ones(X.shape[1], dtype=bool)
            
            # 计算剩余特征的Fisher分数
            X_filtered = X[:, var_mask]
            scores = []
            pos_mask = y == 1
            neg_mask = y == 0
            
            # 检查是否有足够的正负样本
            if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
                # 如果没有足够的样本，返回前N个特征
                num_select = min(200, X.shape[1])
                return np.arange(num_select)
            
            for i in range(X_filtered.shape[1]):
                try:
                    pos_data = X_filtered[pos_mask, i]
                    neg_data = X_filtered[neg_mask, i]
                    
                    pos_mean = np.mean(pos_data)
                    neg_mean = np.mean(neg_data)
                    pos_var = np.var(pos_data)
                    neg_var = np.var(neg_data)
                    
                    if pos_var + neg_var == 0:
                        scores.append(0)
                    else:
                        score = (pos_mean - neg_mean) ** 2 / (pos_var + neg_var + 1e-6)
                        scores.append(score)
                except:
                    scores.append(0)
            
            # 选择前N个最佳特征
            scores = np.array(scores)
            num_select = min(200, len(scores), X.shape[1])  # 确保不超过原始特征数
            
            if len(scores) == 0:
                return np.arange(num_select)
            
            best_indices = np.argsort(scores)[-num_select:]
            
            # 将索引映射回原始特征空间
            var_indices = np.where(var_mask)[0]
            selected = var_indices[best_indices]
            
            # 确保选择的索引不超出范围
            selected = selected[selected < X.shape[1]]
            
            # 如果选择的特征太少，补充一些
            if len(selected) < 50:
                additional_needed = min(50 - len(selected), X.shape[1] - len(selected))
                all_indices = np.arange(X.shape[1])
                available = np.setdiff1d(all_indices, selected)
                if len(available) > 0:
                    additional = np.random.choice(available, min(additional_needed, len(available)), replace=False)
                    selected = np.concatenate([selected, additional])
            
            return selected
            
        except Exception as e:
            print(f"HOG特征选择错误: {e}")
            return np.arange(min(200, X.shape[1]))
    
    def _train_classifier(self, X, y):
        """训练支持向量机分类器（简版）"""
        try:
            if X.shape[0] == 0 or X.shape[1] == 0:
                return np.random.randn(X.shape[1] + 1) * 0.1
            
            # 添加偏置项
            X_bias = np.column_stack([X, np.ones(X.shape[0])])
            
            # 使用岭回归作为简单的线性分类器
            lambda_reg = 0.01
            try:
                XTX = X_bias.T @ X_bias
                XTy = X_bias.T @ y
                
                # 添加正则化项
                reg_matrix = lambda_reg * np.eye(XTX.shape[0])
                
                weights = np.linalg.solve(XTX + reg_matrix, XTy)
                
            except np.linalg.LinAlgError:
                try:
                    # 如果矩阵奇异，使用伪逆
                    weights = np.linalg.pinv(X_bias) @ y
                except:
                    # 最后的fallback
                    weights = np.random.randn(X_bias.shape[1]) * 0.1
            
            return weights
            
        except Exception as e:
            print(f"HOG分类器训练错误: {e}")
            return np.random.randn(X.shape[1] + 1) * 0.1
    
    def _search_target(self, gray):
        """在搜索区域内寻找目标"""
        try:
            best_score = -np.inf
            best_center = None
            
            cx, cy = self.center
            w, h = self.target_size
            
            # 生成搜索候选位置（减少搜索密度提高速度）
            candidates = []
            step = 4  # 增大搜索步长
            for dx in range(-self.search_radius, self.search_radius + 1, step):
                for dy in range(-self.search_radius, self.search_radius + 1, step):
                    new_cx = cx + dx
                    new_cy = cy + dy
                    
                    # 检查边界
                    if (new_cx - w//2 >= 0 and new_cy - h//2 >= 0 and
                        new_cx + w//2 < gray.shape[1] and new_cy + h//2 < gray.shape[0]):
                        candidates.append((new_cx, new_cy))
            
            # 评估候选位置
            for new_cx, new_cy in candidates:
                bbox = (new_cx - w//2, new_cy - h//2, w, h)
                features = self._extract_hog_features(gray, bbox)
                
                if features is not None:
                    score = self._classify(features)
                    if score > best_score:
                        best_score = score
                        best_center = (new_cx, new_cy)
            
            # 转换得分为置信度
            confidence = 1.0 / (1.0 + np.exp(-best_score))  # Sigmoid
            
            return best_center, confidence
            
        except Exception as e:
            print(f"HOG目标搜索错误: {e}")
            return None, 0.0
    
    # 获取当前尺度信息
    def get_scale_info(self):
        """
        获取当前尺度信息
        Returns:
            dict: 尺度信息
        """
        return {
            'current_scale': self.current_scale,
            'scale_history': self.scale_history.copy() if hasattr(self, 'scale_history') else [],
            'target_size': self.target_size,
            'multiscale_enabled': self.enable_multiscale
        }


    # 设置多尺度参数
    def set_multiscale_params(self, enable=None, scale_factors=None, update_threshold=None):
        """
        设置多尺度检测参数
        Args:
            enable: 是否启用多尺度检测
            scale_factors: 尺度因子列表
            update_threshold: 尺度更新阈值
        """
        try:
            if enable is not None:
                self.enable_multiscale = enable
                print(f"多尺度检测: {'启用' if enable else '禁用'}")
            
            if scale_factors is not None:
                self.scale_factors = sorted(scale_factors)
                print(f"尺度因子更新: {self.scale_factors}")
            
            if update_threshold is not None:
                self.scale_update_threshold = max(0.1, min(0.9, update_threshold))
                print(f"尺度更新阈值: {self.scale_update_threshold}")
                
        except Exception as e:
            print(f"设置多尺度参数错误: {e}")
            
    def _extended_search(self, gray):
        """扩大搜索范围的搜索"""
        try:
            if self.enable_multiscale:
                # 多尺度扩展搜索
                best_score = -np.inf
                best_center = None
                best_scale = self.current_scale
                
                # 使用更激进的尺度范围进行搜索
                extended_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
                
                cx, cy = self.center
                base_w, base_h = self.target_size
                extended_radius = self.search_radius * 3
                
                for scale in extended_scales:
                    scaled_w = int(base_w * scale / self.current_scale)
                    scaled_h = int(base_h * scale / self.current_scale)
                    
                    if scaled_w < 16 or scaled_h < 16:
                        continue
                    
                    scaled_w, scaled_h = self._adjust_size_for_hog(scaled_w, scaled_h)
                    
                    # 在当前尺度下进行稀疏搜索
                    step = 12
                    for dx in range(-extended_radius, extended_radius + 1, step):
                        for dy in range(-extended_radius, extended_radius + 1, step):
                            new_cx = cx + dx
                            new_cy = cy + dy
                            
                            if (new_cx - scaled_w//2 >= 0 and new_cy - scaled_h//2 >= 0 and
                                new_cx + scaled_w//2 < gray.shape[1] and new_cy + scaled_h//2 < gray.shape[0]):
                                
                                bbox = (new_cx - scaled_w//2, new_cy - scaled_h//2, scaled_w, scaled_h)
                                features = self._extract_hog_features(gray, bbox)
                                
                                if features is not None:
                                    score = self._classify(features)
                                    # 应用尺度惩罚
                                    penalty = self._calculate_scale_penalty(scale)
                                    penalized_score = score * penalty * 0.8  # 扩展搜索额外惩罚
                                    
                                    if penalized_score > best_score:
                                        best_score = penalized_score
                                        best_center = (new_cx, new_cy)
                                        best_scale = scale
                
                # 如果找到更好的尺度，更新它
                if best_center is not None and best_scale != self.current_scale:
                    confidence = 1.0 / (1.0 + np.exp(-best_score))
                    if confidence > 0.4:  # 降低扩展搜索的尺度更新阈值
                        self._update_scale(best_scale, confidence)
                
                confidence = 1.0 / (1.0 + np.exp(-best_score))
                return best_center, confidence
            else:
                # 原有的单尺度扩展搜索
                extended_radius = self.search_radius * 2
                best_score = -np.inf
                best_center = None
                
                cx, cy = self.center
                w, h = self.target_size
                
                step = 8
                for dx in range(-extended_radius, extended_radius + 1, step):
                    for dy in range(-extended_radius, extended_radius + 1, step):
                        new_cx = cx + dx
                        new_cy = cy + dy
                        
                        if (new_cx - w//2 >= 0 and new_cy - h//2 >= 0 and
                            new_cx + w//2 < gray.shape[1] and new_cy + h//2 < gray.shape[0]):
                            
                            bbox = (new_cx - w//2, new_cy - h//2, w, h)
                            features = self._extract_hog_features(gray, bbox)
                            
                            if features is not None:
                                score = self._classify(features)
                                if score > best_score:
                                    best_score = score
                                    best_center = (new_cx, new_cy)
                
                confidence = 1.0 / (1.0 + np.exp(-best_score))
                return best_center, confidence
                
        except Exception as e:
            print(f"多尺度扩展搜索错误: {e}")
            return None, 0.0
    
    def _classify(self, features):
        """分类HOG特征向量"""
        try:
            if self.classifier is None or self.feature_selector is None:
                return 0.0
            
            # 检查特征维度一致性
            if len(features) != self.feature_dim:
                print(f"特征维度不匹配: 期望 {self.feature_dim}, 实际 {len(features)}")
                return 0.0
            
            # 检查特征选择器索引是否有效
            valid_indices = self.feature_selector < len(features)
            if not np.all(valid_indices):
                print(f"特征选择器索引超出范围，重新初始化...")
                # 重新初始化特征选择器
                self.feature_selector = np.arange(min(len(features), 200))
            
            # 选择有效的特征
            valid_selector = self.feature_selector[self.feature_selector < len(features)]
            if len(valid_selector) == 0:
                return 0.0
                
            selected_features = features[valid_selector]
            
            # 检查分类器维度
            expected_dim = len(valid_selector) + 1  # +1 for bias
            if len(self.classifier) != expected_dim:
                print(f"分类器维度不匹配，重新初始化...")
                self.classifier = np.random.randn(expected_dim) * 0.1
            
            # 添加偏置项并分类
            features_bias = np.append(selected_features, 1.0)
            score = np.dot(self.classifier, features_bias)
            
            return score
            
        except Exception as e:
            print(f"HOG分类错误: {e}")
            return 0.0
    
    def _update_classifier(self, gray):
        """更新分类器"""
        try:
            # 生成新的训练样本（减少样本数提高速度）
            positive_samples = self._generate_samples(gray, self.center, self.positive_radius, 8)
            negative_samples = self._generate_samples(gray, self.center, self.negative_radius, 15)
            
            if len(positive_samples) == 0 or len(negative_samples) == 0:
                return
            
            # 提取HOG特征
            pos_features = []
            neg_features = []
            
            for sample in positive_samples:
                feat = self._extract_hog_features(gray, sample)
                if feat is not None and len(feat) == self.feature_dim:  # 检查特征维度
                    pos_features.append(feat)
            
            for sample in negative_samples:
                feat = self._extract_hog_features(gray, sample)
                if feat is not None and len(feat) == self.feature_dim:  # 检查特征维度
                    neg_features.append(feat)
            
            if len(pos_features) == 0 or len(neg_features) == 0:
                return
            
            # 创建新的训练数据
            X_new = np.vstack([pos_features, neg_features])
            y_new = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
            
            # 检查特征维度一致性
            if X_new.shape[1] != self.feature_dim:
                print(f"更新时特征维度不匹配: 期望 {self.feature_dim}, 实际 {X_new.shape[1]}")
                return
            
            # 确保特征选择器有效
            valid_selector = self.feature_selector[self.feature_selector < X_new.shape[1]]
            if len(valid_selector) == 0:
                return
            
            # 选择特征并训练
            X_selected = X_new[:, valid_selector]
            
            # 在线更新权重（指数移动平均）
            X_bias = np.column_stack([X_selected, np.ones(X_selected.shape[0])])
            
            try:
                lambda_reg = 0.01
                XTX = X_bias.T @ X_bias
                XTy = X_bias.T @ y_new
                reg_matrix = lambda_reg * np.eye(XTX.shape[0])
                
                new_weights = np.linalg.solve(XTX + reg_matrix, XTy)
                
                # 检查分类器维度
                if len(self.classifier) == len(new_weights):
                    self.classifier = (self.learning_rate * self.classifier + 
                                    (1 - self.learning_rate) * new_weights)
                else:
                    print(f"分类器维度不匹配，重新初始化: {len(self.classifier)} vs {len(new_weights)}")
                    self.classifier = new_weights
            except:
                pass  # 保持原权重
                
        except Exception as e:
            print(f"HOG分类器更新错误: {e}")
    
    def _update_bbox(self):
        """更新边界框"""
        cx, cy = self.center
        w, h = self.target_size
        
        self.bbox = {
            'x': cx - w // 2,
            'y': cy - h // 2,
            'width': w,
            'height': h
        }
    
    def get_confidence(self):
        """获取当前置信度"""
        if len(self.confidence_history) == 0:
            return 0.0
        return self.confidence_history[-1]
    
    def get_avg_confidence(self, window=10):
        """获取平均置信度"""
        if len(self.confidence_history) == 0:
            return 0.0
        recent = self.confidence_history[-window:]
        return sum(recent) / len(recent)
    
    def get_hog_params(self):
        """获取当前HOG参数"""
        return self.hog_params.copy()