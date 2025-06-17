from numpy.fft import fft2, ifft2, fftshift
import cv2
import numpy as np
 
class HOG:
    def __init__(self, winSize):
        """
        :param winSize: 检测窗口的大小
        """
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nBins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nBins)
 
    def get_feature(self, image):
        winStride = self.winSize
        w, h = self.winSize
        w_block, h_block = self.blockStride
        w = w//w_block - 1
        h = h//h_block - 1
        # 计算给定图像的HOG特征描述子，一个n*1的特征向量
        hist = self.hog.compute(img=image, winStride=winStride, padding=(0, 0))
        return hist.reshape(w, h, 36).transpose(2, 1, 0)    # 交换轴的顺序
 
    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h*grid, w*grid))
 
        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(x=np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(x=np.pi / 9 * k))
                    cv2.rectangle(img=img, pt1=(j*grid, i*grid), pt2=((j + 1) * grid, (i + 1) * grid),
                                  color=(255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
 
class KCFTracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False
 
        # 算法变量定义
        self.scale_h = 0.
        self.scale_w = 0.
 
        self.ph = 0
        self.pw = 0
        self.hog = None
        self.alphaf = None
        self.x = None
        self.roi = None
        self.initialized = False
 
    def init(self, image, roi):
        """
        初始化跟踪器
        :param image: 第一帧图像
        :param roi: 初始ROI元组 (x, y, w, h)
        :return: 初始化是否成功
        """
        try:
            x1, y1, w, h = roi
            cx = x1 + w // 2
            cy = y1 + h // 2
            roi_center = (cx, cy, w, h)
 
            # 确定Patch的大小，并在此Patch中提取HOG特征描述子
            scale = self.max_patch_size / float(max(w, h))
            self.ph = int(h * scale) // 4 * 4 + 4
            self.pw = int(w * scale) // 4 * 4 + 4
            self.hog = HOG((self.pw, self.ph))
 
            # 在矩形框的中心采样、提取特征
            x = self.get_feature(image, roi_center)
            y = self.gaussian_peak(x.shape[2], x.shape[1])
 
            self.alphaf = self.train(x, y, self.sigma, self.lambdar)
            self.x = x
            self.roi = roi_center
            self.initialized = True
            
            return True
            
        except Exception as e:
            print(f"KCF初始化失败: {e}")
            self.initialized = False
            return False
 
    def update(self, image):
        """
        更新跟踪器，返回新的边界框
        :param image: 当前帧图像
        :return: (success, bbox_dict) 其中bbox_dict包含x,y,width,height
        """
        if not self.initialized:
            return False, None
            
        try:
            # 包含矩形框信息的四元组(cx, cy, w, h)
            cx, cy, w, h = self.roi
            max_response = -1   # 最大响应值
            
            best_bbox = None
            best_z = None
 
            for scale in [0.95, 1.0, 1.05]:
                # 将ROI值处理为整数
                roi = (int(cx), int(cy), int(w * scale), int(h * scale))
 
                z = self.get_feature(image, roi)    # tuple(36, h, w)
                # 计算响应
                responses = self.detect(self.x, z, self.sigma)
                height, width = responses.shape
                
                if self.debug:
                    cv2.imshow("res", responses)
                    cv2.waitKey(0)
                    
                idx = np.argmax(responses)
                res = np.max(responses)
                
                if res > max_response:
                    max_response = res
                    dx = int((idx % width - width / 2) / self.scale_w)
                    dy = int((idx // width - height / 2) / self.scale_h)
                    best_w = int(w * scale)
                    best_h = int(h * scale)
                    best_z = z
                    
                    # 更新矩形框的相关参数
                    new_cx = cx + dx
                    new_cy = cy + dy
                    
                    # 转换为左上角坐标格式
                    best_bbox = {
                        'x': new_cx - best_w // 2,
                        'y': new_cy - best_h // 2,
                        'width': best_w,
                        'height': best_h
                    }
 
            if best_bbox is None:
                return False, None
                
            # 更新内部状态
            self.roi = (best_bbox['x'] + best_bbox['width'] // 2, 
                       best_bbox['y'] + best_bbox['height'] // 2,
                       best_bbox['width'], best_bbox['height'])
 
            # 更新模板
            self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
            y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
            new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
            self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate
            
            return True, best_bbox
            
        except Exception as e:
            print(f"KCF更新失败: {e}")
            return False, None
 
    def get_feature(self, image, roi):
        """
        对特征进行采样
        :param image: 输入图像
        :param roi: 包含矩形框信息的四元组(cx, cy, w, h)
        :return: 特征向量
        """
        # 对矩形框做2.5倍的Padding处理
        cx, cy, w, h = roi
        w = int(w*self.padding)//2*2
        h = int(h*self.padding)//2*2
        x = int(cx - w//2)
        y = int(cy - h//2)
        
        # 确保坐标在图像范围内
        x = max(0, min(x, image.shape[1] - w))
        y = max(0, min(y, image.shape[0] - h))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
 
        # 矩形框所覆盖的区域
        sub_img = image[y:y+h, x:x+w]
        
        if sub_img.size == 0:
            # 如果子图像为空，返回零特征
            return np.zeros((36, self.ph, self.pw))
            
        resized_img = cv2.resize(src=sub_img, dsize=(self.pw, self.ph))
 
        if self.gray_feature:
            feature = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw)/255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_img)
            if self.debug:
                self.hog.show_hog(feature)
 
        # Hog特征的通道数、高度、宽度
        fc, fh, fw = feature.shape
        self.scale_h = float(fh)/h if h > 0 else 1.0
        self.scale_w = float(fw)/w if w > 0 else 1.0
 
        # 两个二维数组，前者(fh，1)，后者(1，fw)
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
 
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
 
        # 一个fh x fw的矩阵
        hann2d = hann2t * hann1t
 
        feature = feature * hann2d
        return feature
 
    def gaussian_peak(self, w, h):
        """
        生成高斯峰值
        :param w: 宽度
        :param h: 高度
        :return: 一个w*h的高斯矩阵
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h//2, w//2
 
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g
 
    def kernel_correlation(self, x1, x2, sigma):
        """
        核化的相关滤波操作
        :param x1: 第一个特征向量
        :param x2: 第二个特征向量
        :param sigma: 高斯参数sigma
        :return: 核相关结果
        """
        # 转换到傅里叶空间
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        # \hat{x^*} \otimes \hat{x}'，x*的共轭转置与x'的乘积
        tmp = np.conj(fx1) * fx2
        # 离散傅里叶逆变换转换回真实空间
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        # 将零频率分量移到频谱中心
        idft_rbf = fftshift(idft_rbf)
 
        # 高斯核的径向基函数
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k
 
    def train(self, x, y, sigma, lambdar):
        """
        训练相关滤波器
        :param x: 输入特征
        :param y: 目标响应
        :param sigma: 高斯参数
        :param lambdar: 正则化参数
        :return: 滤波器系数
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)
 
    def detect(self, x, z, sigma):
        """
        检测目标位置
        :param x: 模板特征
        :param z: 当前帧特征
        :param sigma: 高斯参数
        :return: 响应图
        """
        k = self.kernel_correlation(x, z, sigma)
        # 傅里叶逆变换的实部
        return np.real(ifft2(self.alphaf * fft2(k)))

class CSRTTracker(KCFTracker):
    def __init__(self):
        super().__init__()
        self.enable_mask = True
        self.mask = None

    def get_spatial_mask(self, shape, sigma=0.2):
        """
        生成空间可靠性掩码（Spatial Reliability Map），鼓励中心区域、抑制边缘。
        shape: (h, w)
        """
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w // 2, h // 2
        gaussian = np.exp(-0.5 * (((x - cx) ** 2 + (y - cy) ** 2) / (sigma * w * h)))
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
        return gaussian.astype(np.float32)

    def get_feature(self, image, roi):
        """
        获取加权后的特征
        """
        feature = super().get_feature(image, roi)
        fh, fw = feature.shape[1:]
        if self.enable_mask:
            self.mask = self.get_spatial_mask((fh, fw))  # 保存在成员变量中
            feature = feature * self.mask  # 加权空间掩码
        return feature

    def detect(self, x, z, sigma):
        """
        使用空间掩码后的检测
        """
        k = self.kernel_correlation(x, z, sigma)
        response = np.real(ifft2(self.alphaf * fft2(k)))

        if self.enable_mask and self.mask is not None:
            response *= self.mask  # 对响应图加权掩码
        return response
