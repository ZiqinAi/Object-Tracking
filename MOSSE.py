import cv2
import numpy as np

class MOSSETracker:
    def __init__(self, learning_rate=0.25, sigma=2.0, lambda_reg=0.01):
        # 初始化 MOSSE 跟踪器的参数
        self.lr = learning_rate                 # 学习率，用于更新滤波器
        self.sigma = sigma                      # 高斯响应的sigma值，用于生成目标响应
        self.lambda_reg = lambda_reg            # 正则化项 lambda，防止分母为零
        self.tracker_size = (0, 0)              # (宽度, 高度) 模板/滤波器的尺寸
        self.A_num = None                       # 滤波器更新公式中的分子 A
        self.B_den = None                       # 滤波器更新公式中的分母 B
        self.H_filter = None                    # MOSSE 滤波器 H*
        self.center = None                      # 初始边界框的中心点 (x, y)
        self.current_bbox = None                # 当前边界框 {'x': x, 'y': y, 'width': w, 'height': h}
        self.initialized = False                # 跟踪器是否已初始化标志

    def _get_window(self, shape):
        """创建一个汉宁窗"""
        # 为行和列分别创建汉宁窗
        win_col = np.hanning(shape[1]) # 列方向的汉宁窗
        win_row = np.hanning(shape[0]) # 行方向的汉宁窗
        # 使用外积创建二维汉宁窗
        win_2d = np.outer(win_row, win_col)
        return win_2d

    def _get_gaussian_response(self, shape, sigma):
        """创建高斯响应图"""
        h, w = shape
        gk_x = cv2.getGaussianKernel(w, sigma)
        gk_y = cv2.getGaussianKernel(h, sigma)
        gaussian = gk_y @ gk_x.T
        return np.fft.ifftshift(gaussian)

    def _preprocess_image(self, frame, bbox_coords):
        """提取并预处理ROI"""
        x, y, w, h = bbox_coords 
        
        # 确保ROI在图像边界内
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h) # frame.shape[1]是宽度, frame.shape[0]是高度
        
        # 如果ROI无效
        if x2 <= x1 or y2 <= y1:
            # 如果跟踪器尺寸已定义，则返回一个标准尺寸的黑色图像，否则返回None
            if self.tracker_size[0] > 0 and self.tracker_size[1] > 0:
                return np.zeros((self.tracker_size[1], self.tracker_size[0]), dtype=np.float32)
            else: 
                return None

        roi = frame[y1:y2, x1:x2] # 提取ROI

        # 转换为灰度图
        if len(roi.shape) > 2 and roi.shape[2] > 1: # 如果是彩色图像
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi

        # 将ROI调整到标准的跟踪器尺寸
        resized_roi = cv2.resize(gray_roi, self.tracker_size, interpolation=cv2.INTER_AREA)

        # 转换为float32类型，用于FFT计算
        img_float = resized_roi.astype(np.float32)

        # 对数变换以减少光照影响
        img_float = np.log(img_float + 1.0) # 加1防止log(0)

        # 归一化
        img_float = (img_float - np.mean(img_float)) / (np.std(img_float) + 1e-5) # 加1e-5，防止除以零

        # 应用汉宁窗以减少边界效应
        hanning_window = self._get_window(img_float.shape)
        processed_img = img_float * hanning_window
        
        return processed_img

    def init(self, frame, bbox):
        """初始化MOSSE跟踪器"""
        try:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 检查边界框尺寸是否有效
            if w <= 0 or h <= 0:
                print("MOSSE 初始化错误: 边界框宽度或高度为零或负数。")
                self.initialized = False
                return False

            # 保存当前边界框和中心点
            self.current_bbox = {'x': x, 'y': y, 'width': w, 'height': h}
            self.center = (x + w / 2, y + h / 2)
            
            # 设置跟踪器尺寸，确保至少为32x32
            self.tracker_size = (max(32, w), max(32, h)) # 宽度, 高度

            # 提取并预处理初始图像块, 确保ROI在图像内
            fi = self._preprocess_image(frame, (x, y, w, h))
            if fi is None:
                print("MOSSE 初始化错误: 图像预处理失败，ROI可能在图像外或过小。")
                self.initialized = False
                return False

            # 创建期望的高斯响应
            g_shape = (self.tracker_size[1], self.tracker_size[0]) # (高度, 宽度)
            g = self._get_gaussian_response(g_shape, self.sigma)
            G = np.fft.fft2(g) # 高斯响应的傅里叶变换

            # 计算初始图像块的傅里叶变换 (F_fft)
            Fi = np.fft.fft2(fi)

            # 计算初始滤波器 H* = (G .* conj(F)) / (F .* conj(F) + lambda) .* 表示逐元素相乘，conj() 是复共轭
            self.A_num = G * np.conj(Fi) # 分子 A = G * F_conj
            self.B_den = Fi * np.conj(Fi) + self.lambda_reg # 分母 B = F * F_conj + lambda (正则化项)

            self.H_filter = self.A_num / self.B_den # 计算滤波器 H*
            self.initialized = True # 标记初始化成功
            return True
        
        except Exception as e:
            print(f"MOSSE 初始化错误: {e}")
            self.initialized = False
            return False

    def update(self, frame):
        """用新的一帧图像更新跟踪器状态"""
        if not self.initialized or self.H_filter is None: # 如果未初始化或滤波器不存在
            return False, None

        try:
            # 使用 self.current_bbox 获取坐标
            x, y, w, h = self.current_bbox['x'], self.current_bbox['y'], self.current_bbox['width'], self.current_bbox['height']
            
            # 提取并预处理当前帧的图像块 (F_new)
            fi_new = self._preprocess_image(frame, (x, y, w, h))
            if fi_new is None: # 如果预处理失败
                 return False, None

            # 计算新图像块的傅里叶变换 (F_new_fft)
            Fi_new = np.fft.fft2(fi_new)

            # 计算相关性响应图 R = F_new_fft * conj(H_filter)
            R_fft = Fi_new * np.conj(self.H_filter) # 注意使用滤波器的复共轭
            response_map_complex = np.fft.ifft2(R_fft) # 傅里叶反变换得到响应图
            response_map = np.real(response_map_complex) # 取实部

            # 在响应图中找到峰值位置
            peak_val = np.max(response_map) # 响应峰值
            max_loc = np.unravel_index(np.argmax(response_map), response_map.shape) # 峰值位置 (行, 列) -> (dy, dx)

            # 亚像素峰值定位
            r_int, c_int = max_loc # 整数峰值位置 (row, col)
            
            # 准备进行二次插值的值
            val_center = response_map[r_int, c_int]
            
            val_r_prev = response_map[r_int - 1, c_int] if r_int > 0 else val_center
            val_r_next = response_map[r_int + 1, c_int] if r_int < response_map.shape[0] - 1 else val_center
            
            val_c_prev = response_map[r_int, c_int - 1] if c_int > 0 else val_center
            val_c_next = response_map[r_int, c_int + 1] if c_int < response_map.shape[1] - 1 else val_center

            # 计算亚像素偏移
            denominator_r = (val_r_prev - 2 * val_center + val_r_next)
            r_subpixel_offset = 0.0
            if abs(denominator_r) > 1e-5: # 避免除以零
                r_subpixel_offset = 0.5 * (val_r_prev - val_r_next) / denominator_r
            
            denominator_c = (val_c_prev - 2 * val_center + val_c_next)
            c_subpixel_offset = 0.0
            if abs(denominator_c) > 1e-5: # 避免除以零
                c_subpixel_offset = 0.5 * (val_c_prev - val_c_next) / denominator_c

            # 确保偏移量在合理范围内
            r_subpixel_offset = np.clip(r_subpixel_offset, -0.5, 0.5)
            c_subpixel_offset = np.clip(c_subpixel_offset, -0.5, 0.5)

            r_final = r_int + r_subpixel_offset
            c_final = c_int + c_subpixel_offset
            # 亚像素峰值定位结束

            # 计算位移 (delta_y, delta_x)，使用亚像素精度的 r_final, c_final
            dy = r_final - self.tracker_size[1] if r_final > self.tracker_size[1] // 2 else r_final
            dx = c_final - self.tracker_size[0] if c_final > self.tracker_size[0] // 2 else c_final

            # 更新目标中心和边界框
            self.center = (self.center[0] + dx, self.center[1] + dy) # 更新中心点
            
            # 根据新的中心点计算新的边界框左上角坐标
            new_x = int(self.center[0] - w / 2)
            new_y = int(self.center[1] - h / 2)

            # 确保边界框在图像范围内
            new_x_clamped = max(0, min(new_x, frame.shape[1] - w))
            new_y_clamped = max(0, min(new_y, frame.shape[0] - h))
            
            self.current_bbox = {'x': new_x_clamped, 'y': new_y_clamped, 'width': w, 'height': h}

            # 更新滤波器 -- 自适应学习
            fi_at_new_loc = self._preprocess_image(frame, (new_x_clamped, new_y_clamped, w, h))
            if fi_at_new_loc is None: # 如果在新位置提取失败
                return False, None

            Fi_at_new_loc_fft = np.fft.fft2(fi_at_new_loc) # 新位置图像块的傅里叶变换
            
            # 创建用于更新的高斯目标 G_update
            g_shape_update = (self.tracker_size[1], self.tracker_size[0])
            g_update = self._get_gaussian_response(g_shape_update, self.sigma)
            G_update_fft = np.fft.fft2(g_update)

            # 更新滤波器分子 A 和分母 B
            current_A_num = G_update_fft * np.conj(Fi_at_new_loc_fft)
            current_B_den = Fi_at_new_loc_fft * np.conj(Fi_at_new_loc_fft) + self.lambda_reg

            # 线性插值更新 A 和 B
            self.A_num = (1 - self.lr) * self.A_num + self.lr * current_A_num
            self.B_den = (1 - self.lr) * self.B_den + self.lr * current_B_den
            
            # 更新滤波器 H*
            self.H_filter = self.A_num / self.B_den

            return True, self.current_bbox # 返回成功和新的边界框

        except Exception as e:
            print(f"MOSSE 更新错误: {e}")
            return False, None
