from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import atexit # 注册清理函数
import numpy as np
import os
import json
import tempfile
import uuid
from datetime import datetime
import threading
import time
import base64
from io import BytesIO
import queue
from MOSSE import MOSSETracker  # 导入自定义MOSSE跟踪器
# from KCF import KCFTracker  # 导入自定义KCF跟踪器
# from CSRT import CSRTTracker  # 导入自定义CSRT跟踪器
from TLD import TLDTracker  # 导入自定义TLD跟踪器
from Boostingtracking import BoostingTracker  # 导入自定义Boosting跟踪器
# from MIL import MILTracker  # 导入自定义MIL跟踪器


app = Flask(__name__)
CORS(app)

# 配置文件上传和存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEMP_FOLDER = 'temp'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 全局变量存储跟踪任务状态和实时摄像头状态
tracking_tasks = {}
camera_sessions = {}
camera_lock = threading.Lock()

class ObjectTracker:
    def __init__(self, algorithm='kcf'):
        self.algorithm = algorithm
        self.tracker = None
        self.custom_mosse = None  # 自定义MOSSE跟踪器
        self.custom_tld = None    # 自定义TLD跟踪器
        self.custom_boosting = None # 自定义Boosting跟踪器
        self.initialized = False
        
    def create_tracker(self):
        try:
            if self.algorithm == 'mosse':
                # 使用自定义MOSSE跟踪器
                print("使用自定义MOSSE跟踪器")
                self.custom_mosse = MOSSETracker(learning_rate=0.125, sigma=2.0)
                return True
            elif self.algorithm == 'kcf':
                tracker = cv2.legacy.TrackerKCF_create()
            elif self.algorithm == 'csrt':
                tracker = cv2.legacy.TrackerCSRT_create()
            elif self.algorithm == 'mil':
                tracker = cv2.legacy.TrackerMIL_create()
            elif self.algorithm == 'boosting':
                print("使用自定义Boosting跟踪器")
                self.custom_boosting = BoostingTracker()
                return True
            elif self.algorithm == 'tld':
                print("使用自定义TLD跟踪器")
                self.custom_tld = TLDTracker()
                return True
            else:
                tracker = cv2.legacy.TrackerKCF_create()

            print(f"成功创建 {self.algorithm} 跟踪器")
            return tracker
        except Exception as e:
            print(f"创建 {self.algorithm} 跟踪器失败: {e}")
            return None

    def initialize(self, frame, bbox):
        if self.algorithm == 'mosse':
            # 使用自定义MOSSE跟踪器
            self.custom_mosse = MOSSETracker(learning_rate=0.125, sigma=2.0)
            bbox_tuple = (int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height']))
            try:
                self.initialized = self.custom_mosse.init(frame, bbox_tuple)
                if not self.initialized:
                    print("自定义MOSSE跟踪器初始化失败")
                    print("图像尺寸:", frame.shape)
                    print("初始框:", bbox)
                else:
                    print("自定义MOSSE跟踪器初始化成功")
                return self.initialized
            except Exception as e:
                print(f"自定义MOSSE跟踪器初始化时发生错误: {e}")
                return False
        
        elif self.algorithm == 'boosting':
            # 使用自定义Boosting跟踪器
            self.custom_boosting = BoostingTracker()
            bbox_tuple = (int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height']))
            try:
                self.initialized = self.custom_boosting.init(frame, bbox_tuple)
                if not self.initialized:
                    print("自定义Boosting跟踪器初始化失败")
                    print("图像尺寸:", frame.shape)
                    print("初始框:", bbox)
                else:
                    print("自定义Boosting跟踪器初始化成功")
                return self.initialized
            except Exception as e:
                print(f"自定义Boosting跟踪器初始化时发生错误: {e}")
                return False
        
        elif self.algorithm == 'tld':
            # 使用自定义TLD跟踪器
            self.custom_tld = TLDTracker()
            bbox_tuple = (int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height']))
            try:
                self.initialized = self.custom_tld.init(frame, bbox_tuple)
                if not self.initialized:
                    print("自定义TLD跟踪器初始化失败")
                    print("图像尺寸:", frame.shape)
                    print("初始框:", bbox)
                else:
                    print("自定义TLD跟踪器初始化成功")
                return self.initialized
            except Exception as e:
                print(f"自定义TLD跟踪器初始化时发生错误: {e}")
        
        else:
            # 使用OpenCV跟踪器
            self.tracker = self.create_tracker()
            if self.tracker is None:
                print(f"跟踪器创建失败，无法初始化 {self.algorithm}")
                return False

            bbox_tuple = (int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height']))
            try:
                self.initialized = self.tracker.init(frame, bbox_tuple)
                if not self.initialized:
                    print(f"{self.algorithm} 跟踪器初始化失败")
                    print("图像尺寸:", frame.shape)
                    print("初始框:", bbox)
                    print("是否在图像内:", 0 <= bbox['x'] < frame.shape[1], 0 <= bbox['y'] < frame.shape[0])
                else:
                    print(f"{self.algorithm} 跟踪器初始化成功")
                return self.initialized
            except Exception as e:
                print(f"{self.algorithm} 跟踪器初始化时发生错误: {e}")
                return False
    
    def update(self, frame):
        """更新跟踪器"""
        if not self.initialized:
            return False, None
        
        if self.algorithm == 'mosse':
            # 使用自定义MOSSE跟踪器
            success, bbox = self.custom_mosse.update(frame)
            return success, bbox
        elif self.algorithm == 'boosting':
            # 使用自定义Boosting跟踪器
            success, bbox = self.custom_boosting.update(frame)
            return success, bbox
        elif self.algorithm == 'tld':
            # 使用自定义TLD跟踪器
            success, bbox = self.custom_tld.update(frame)
            return success, bbox
        else:
            # 使用OpenCV跟踪器
            success, bbox = self.tracker.update(frame)
            if success:
                return True, {
                    'x': int(bbox[0]),
                    'y': int(bbox[1]), 
                    'width': int(bbox[2]),
                    'height': int(bbox[3])
                }
            return False, None

class CameraSession:
    def __init__(self, session_id, camera_id=0):
        self.session_id = session_id
        self.camera_id = camera_id
        self.cap = None
        self.tracker = None
        self.tracking_active = False
        self.bbox = None
        self.algorithm = 'kcf'
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.fps = 30
        self.width = 640
        self.height = 480
        # 添加轨迹相关属性
        self.trajectory = []  # 存储轨迹点
        self.max_trajectory_length = 50  # 最大轨迹长度
        self.show_trajectory = True  # 是否显示轨迹
        
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.camera_id}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 获取实际参数
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"摄像头启动成功: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"启动摄像头失败: {e}")
            return False
    
    def _capture_loop(self):
        """摄像头捕获循环"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 如果启用了跟踪，进行目标跟踪
                if self.tracking_active and self.tracker:
                    success, bbox = self.tracker.update(frame)
                    if success and bbox:
                        self.bbox = bbox
                        # 绘制跟踪框
                        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                        
                        # 计算目标中心点并添加到轨迹
                        center_x = x + w // 2
                        center_y = y + h // 2
                        self.trajectory.append((center_x, center_y))
                        
                        # 限制轨迹长度
                        if len(self.trajectory) > self.max_trajectory_length:
                            self.trajectory.pop(0)
                        
                        # 绘制轨迹线（蓝色）
                        if self.show_trajectory and len(self.trajectory) > 1:
                            for i in range(1, len(self.trajectory)):
                                # 计算透明度，越新的点越不透明
                                alpha = i / len(self.trajectory)
                                thickness = max(1, int(3 * alpha))
                                
                                cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                        (255, 0, 0), thickness)  # 蓝色轨迹
                        
                        # 绘制当前目标中心点（红色小圆点）
                        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                        
                        # 绘制跟踪框（绿色）
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # 添加算法标签
                        label = f'{self.algorithm.upper()} Tracker'
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 添加帧率和轨迹点数信息
                        cv2.putText(frame, f'FPS: {self.fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f'Trajectory: {len(self.trajectory)} points', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # 跟踪失败
                        cv2.putText(frame, 'Tracking Lost', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.bbox = None
                        # 跟踪失败时不清空轨迹，保持历史轨迹显示
                
                # 更新最新帧
                self.latest_frame = frame.copy()
                
                # 将帧添加到队列（用于流式传输）
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # 控制帧率
                time.sleep(1.0 / max(30, self.fps))
                
            except Exception as e:
                print(f"摄像头捕获错误: {e}")
                break
    
    def start_tracking(self, bbox, algorithm='kcf'):
        """开始目标跟踪"""
        if self.latest_frame is None:
            return False, "没有可用的摄像头帧"
        
        self.algorithm = algorithm
        self.tracker = ObjectTracker(algorithm)
        
        success = self.tracker.initialize(self.latest_frame, bbox)
        if success:
            self.tracking_active = True
            self.bbox = bbox
            # 重置轨迹，添加初始中心点
            center_x = int(bbox['x'] + bbox['width'] // 2)
            center_y = int(bbox['y'] + bbox['height'] // 2)
            self.trajectory = [(center_x, center_y)]
            return True, "跟踪启动成功"
        else:
            return False, "跟踪器初始化失败"
    
    def stop_tracking(self, clear_trajectory=False):
        """停止目标跟踪"""
        self.tracking_active = False
        self.tracker = None
        self.bbox = None
        if clear_trajectory:
            self.trajectory = []
    
    def get_latest_frame(self):
        """获取最新帧"""
        return self.latest_frame
    
    def get_frame_stream(self):
        """获取帧流（用于视频流）"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                yield frame
            except queue.Empty:
                continue
            except Exception as e:
                print(f"帧流错误: {e}")
                break
    
    def stop_camera(self):
        """停止摄像头"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print(f"摄像头会话 {self.session_id} 已停止")

    def toggle_trajectory(self):
        """切换轨迹显示"""
        self.show_trajectory = not self.show_trajectory
        return self.show_trajectory
    
    def clear_trajectory(self):
        """清空轨迹"""
        self.trajectory = []
    
    def set_trajectory_length(self, length):
        """设置轨迹最大长度"""
        self.max_trajectory_length = max(10, min(200, length))
        # 如果当前轨迹超过新长度，进行裁剪
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory = self.trajectory[-self.max_trajectory_length:]

# 视频处理跟踪函数
def process_video_tracking(task_id, video_path, bbox, algorithm):
    """处理视频跟踪的后台任务"""
    try:
        # 更新任务状态
        tracking_tasks[task_id]['status'] = 'processing'
        tracking_tasks[task_id]['progress'] = 0
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            tracking_tasks[task_id]['status'] = 'error'
            tracking_tasks[task_id]['error'] = '无法打开视频文件'
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建跟踪器
        tracker = ObjectTracker(algorithm)
        
        # 读取第一帧并初始化跟踪器
        ret, first_frame = cap.read()
        if not ret:
            tracking_tasks[task_id]['status'] = 'error'
            tracking_tasks[task_id]['error'] = '无法读取视频帧'
            cap.release()
            return
        
        if not tracker.initialize(first_frame, bbox):
            tracking_tasks[task_id]['status'] = 'error'
            tracking_tasks[task_id]['error'] = '跟踪器初始化失败'
            cap.release()
            return
        
        # 准备输出视频写入器
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_tracked.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 存储跟踪数据和轨迹
        tracking_data = []
        trajectory = []
        frame_count = 0
        
        # 重置视频到开头
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count == 0:
                # 第一帧，使用初始边界框
                current_bbox = bbox
                success = True
            else:
                # 更新跟踪器
                success, current_bbox = tracker.update(frame)
            
            if success and current_bbox:
                # 确保坐标是整数
                x, y, w, h = int(current_bbox['x']), int(current_bbox['y']), int(current_bbox['width']), int(current_bbox['height'])
                
                # 计算中心点并添加到轨迹
                center_x = x + w // 2
                center_y = y + h // 2
                trajectory.append((center_x, center_y))
                
                # 绘制轨迹线（蓝色）
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        # 计算透明度和线条粗细
                        alpha = i / len(trajectory)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), thickness)
                
                # 绘制当前目标中心点（红色）
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                
                # 绘制跟踪框（绿色）
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 添加算法标签
                label = f'{algorithm.upper()} Tracker'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 添加轨迹信息
                cv2.putText(frame, f'Trajectory: {len(trajectory)} points', (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 保存跟踪数据
                tracking_data.append({
                    'frame': frame_count,
                    'bbox': current_bbox,
                    'center': {'x': center_x, 'y': center_y},
                    'confidence': 0.8 if success else 0.0
                })
            else:
                # 跟踪失败，绘制红色提示，但保持轨迹显示
                cv2.putText(frame, 'Tracking Lost', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 继续绘制已有轨迹
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        alpha = i / len(trajectory)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), thickness)
                
                tracking_data.append({
                    'frame': frame_count,
                    'bbox': None,
                    'center': None,
                    'confidence': 0.0
                })
            
            # 写入输出视频
            out.write(frame)
            
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            tracking_tasks[task_id]['progress'] = progress
            
            # 每处理10帧更新一次状态
            if frame_count % 10 == 0:
                tracking_tasks[task_id]['current_frame'] = frame_count
                tracking_tasks[task_id]['total_frames'] = total_frames
        
        # 清理资源
        cap.release()
        out.release()
        
        # 保存跟踪数据到JSON文件（包含轨迹信息）
        data_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_data.json')
        result_data = {
            'tracking_data': tracking_data,
            'trajectory': trajectory,
            'total_frames': total_frames
        }
        with open(data_path, 'w') as f:
            json.dump(result_data, f)
        
        # 更新任务状态为完成
        tracking_tasks[task_id]['status'] = 'completed'
        tracking_tasks[task_id]['progress'] = 100
        tracking_tasks[task_id]['output_video'] = output_path
        tracking_tasks[task_id]['tracking_data'] = data_path
        tracking_tasks[task_id]['completed_at'] = datetime.now().isoformat()
        
        # 计算统计信息
        successful_tracks = [d for d in tracking_data if d['confidence'] > 0]
        avg_confidence = sum(d['confidence'] for d in successful_tracks) / len(successful_tracks) if successful_tracks else 0
        
        tracking_tasks[task_id]['statistics'] = {
            'total_frames': total_frames,
            'successful_frames': len(successful_tracks),
            'success_rate': len(successful_tracks) / total_frames * 100,
            'avg_confidence': avg_confidence,
            'trajectory_length': len(trajectory)
        }
        
    except Exception as e:
        tracking_tasks[task_id]['status'] = 'error'
        tracking_tasks[task_id]['error'] = str(e)

# ==================== 摄像头相关API ====================
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头会话"""
    data = request.json or {}
    camera_id = data.get('camera_id', 0)
    
    with camera_lock:
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 创建摄像头会话
        session = CameraSession(session_id, camera_id)
        
        if session.start_camera():
            camera_sessions[session_id] = session
            return jsonify({
                'session_id': session_id,
                'status': 'success',
                'message': '摄像头启动成功',
                'camera_info': {
                    'width': session.width,
                    'height': session.height,
                    'fps': session.fps
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '摄像头启动失败'
            }), 500

@app.route('/api/camera/stop/<session_id>', methods=['POST'])
def stop_camera(session_id):
    """停止摄像头会话"""
    with camera_lock:
        if session_id in camera_sessions:
            camera_sessions[session_id].stop_camera()
            del camera_sessions[session_id]
            return jsonify({
                'status': 'success',
                'message': '摄像头已停止'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '会话不存在'
            }), 404

@app.route('/api/camera/frame/<session_id>')
def get_camera_frame(session_id):
    """获取摄像头当前帧（用于选择跟踪目标）"""
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    frame = session.get_latest_frame()
    
    if frame is None:
        return jsonify({'error': '没有可用帧'}), 404
    
    # 将帧转换为base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'frame': frame_b64,
        'width': frame.shape[1],
        'height': frame.shape[0],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/camera/stream/<session_id>')
def camera_stream(session_id):
    """摄像头视频流"""
    if session_id not in camera_sessions:
        return "Session not found", 404
    
    session = camera_sessions[session_id]
    
    def generate():
        for frame in session.get_frame_stream():
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"流生成错误: {e}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/track/start', methods=['POST'])
def start_camera_tracking():
    """开始摄像头实时跟踪"""
    data = request.json
    session_id = data.get('session_id')
    bbox = data.get('bbox')
    algorithm = data.get('algorithm', 'kcf')
    
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    if not bbox:
        return jsonify({'error': '未指定跟踪目标'}), 400
    
    session = camera_sessions[session_id]
    success, message = session.start_tracking(bbox, algorithm)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': message,
            'tracking_active': True
        })
    else:
        return jsonify({
            'status': 'error',
            'message': message
        }), 500

@app.route('/api/camera/track/stop', methods=['POST'])
def stop_camera_tracking():
    """停止摄像头实时跟踪"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    session.stop_tracking()
    
    return jsonify({
        'status': 'success',
        'message': '跟踪已停止',
        'tracking_active': False
    })

@app.route('/api/camera/status/<session_id>')
def get_camera_status(session_id):
    """获取摄像头会话状态"""
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    
    return jsonify({
        'session_id': session_id,
        'running': session.running,
        'tracking_active': session.tracking_active,
        'algorithm': session.algorithm,
        'current_bbox': session.bbox,
        'trajectory_info': {
            'show_trajectory': session.show_trajectory,
            'trajectory_length': len(session.trajectory),
            'max_trajectory_length': session.max_trajectory_length
        },
        'camera_info': {
            'width': session.width,
            'height': session.height,
            'fps': session.fps
        }
    })

# ==================== 视频方面API ====================
@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上传视频文件"""
    if 'video' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file:
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取视频信息
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            os.remove(filepath)
            return jsonify({'error': '无效的视频文件'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'filepath': filepath,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
        })

@app.route('/api/extract_frame', methods=['POST'])
def extract_frame():
    """提取指定时间的视频帧"""
    data = request.json
    filepath = data.get('filepath')
    timestamp = data.get('timestamp', 0)
    
    if not os.path.exists(filepath):
        return jsonify({'error': '视频文件不存在'}), 404
    
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频文件'}), 400
    
    # 设置视频位置到指定时间戳
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': '无法读取视频帧'}), 400
    
    # 保存临时图片文件
    temp_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_FOLDER, f'{temp_id}.jpg')
    cv2.imwrite(temp_path, frame)
    
    return jsonify({
        'frame_id': temp_id,
        'frame_path': temp_path,
        'width': frame.shape[1],
        'height': frame.shape[0]
    })

@app.route('/api/start_tracking', methods=['POST'])
def start_tracking():
    """开始目标跟踪"""
    data = request.json
    video_path = data.get('video_path')
    bbox = data.get('bbox')
    algorithm = data.get('algorithm', 'kcf')
    
    if not os.path.exists(video_path):
        return jsonify({'error': '视频文件不存在'}), 404
    
    if not bbox:
        return jsonify({'error': '未指定跟踪目标'}), 400
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tracking_tasks[task_id] = {
        'status': 'queued',
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'algorithm': algorithm,
        'video_path': video_path,
        'bbox': bbox
    }
    
    # 启动后台跟踪任务
    thread = threading.Thread(
        target=process_video_tracking,
        args=(task_id, video_path, bbox, algorithm)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': '跟踪任务已启动'
    })

@app.route('/api/tracking_status/<task_id>', methods=['GET'])
def get_tracking_status(task_id):
    """获取跟踪任务状态"""
    if task_id not in tracking_tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(tracking_tasks[task_id])

@app.route('/api/download_result/<task_id>', methods=['GET'])
def download_result(task_id):
    """下载跟踪结果视频"""
    if task_id not in tracking_tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    task = tracking_tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': '任务未完成'}), 400
    
    output_path = task.get('output_video')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': '结果文件不存在'}), 404
    
    return send_file(output_path, as_attachment=True, download_name=f'tracked_video_{task_id}.mp4')

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用的跟踪算法列表"""
    algorithms = {
        'kcf': {
            'name': 'KCF',
            'full_name': 'Kernelized Correlation Filters',
            'description': 'KCF (核相关滤波器) 是一种高效的目标跟踪算法，利用循环矩阵的性质在频域中进行快速计算。优点：速度快，精度较高；适用于大多数跟踪场景。',
            'speed': 'fast',
            'accuracy': 'high'
        },
        'csrt': {
            'name': 'CSRT',
            'full_name': 'Discriminative Correlation Filter with Channel and Spatial Reliability',
            'description': 'CSRT 是OpenCV中较新的跟踪算法，在KCF基础上增加了通道和空间可靠性。优点：跟踪精度高，对遮挡和形变有较好的鲁棒性。',
            'speed': 'medium',
            'accuracy': 'very_high'
        },
        'mosse': {
            'name': 'MOSSE',
            'full_name': 'Minimum Output Sum of Squared Error',
            'description': 'MOSSE 是最早的相关滤波跟踪算法之一，计算简单高效。优点：速度极快；缺点：精度相对较低，适合对速度要求极高的场景。',
            'speed': 'very_fast',
            'accuracy': 'medium'
        },
        'mil': {
            'name': 'MIL',
            'full_name': 'Multiple Instance Learning',
            'description': 'MIL 使用多实例学习的方法进行目标跟踪。优点：对部分遮挡有一定鲁棒性；缺点：速度较慢，容易出现漂移。',
            'speed': 'slow',
            'accuracy': 'medium'
        },
        'boosting': {
            'name': 'Boosting',
            'full_name': 'Boosting Tracker',
            'description': 'Boosting跟踪器使用在线AdaBoost算法。优点：经典算法，理论基础扎实；缺点：速度慢，对背景变化敞感。',
            'speed': 'slow',
            'accuracy': 'medium'
        },
        'tld': {
            'name': 'TLD',
            'full_name': 'Tracking-Learning-Detection',
            'description': 'TLD结合了跟踪、学习和检测三个模块。优点：能够重新检测丢失的目标；缺点：计算复杂度高，参数调节困难。',
            'speed': 'slow',
            'accuracy': 'high'
        }
    }
    return jsonify(algorithms)

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """清理临时文件"""
    data = request.json
    files_to_remove = data.get('files', [])
    
    removed_files = []
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
    
    return jsonify({
        'removed_files': removed_files,
        'count': len(removed_files)
    })

@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """列出可用的摄像头设备"""
    available_cameras = []
    
    # 测试前几个摄像头索引
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            available_cameras.append({
                'id': i,
                'name': f'Camera {i}',
                'resolution': f'{width}x{height}',
                'fps': fps
            })
            cap.release()
    
    return jsonify({
        'cameras': available_cameras,
        'count': len(available_cameras)
    })

# 清理函数，在应用关闭时调用
def cleanup_all_sessions():
    """清理所有摄像头会话"""
    with camera_lock:
        for session_id, session in list(camera_sessions.items()):
            session.stop_camera()
        camera_sessions.clear()


# ==================== 轨迹显示相关API ====================
@app.route('/api/camera/trajectory/toggle', methods=['POST'])
def toggle_trajectory():
    """切换轨迹显示"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    show_trajectory = session.toggle_trajectory()
    
    return jsonify({
        'status': 'success',
        'show_trajectory': show_trajectory,
        'message': f'轨迹显示已{"开启" if show_trajectory else "关闭"}'
    })

@app.route('/api/camera/trajectory/clear', methods=['POST'])
def clear_trajectory():
    """清空轨迹"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    session.clear_trajectory()
    
    return jsonify({
        'status': 'success',
        'message': '轨迹已清空'
    })

@app.route('/api/camera/trajectory/length', methods=['POST'])
def set_trajectory_length():
    """设置轨迹长度"""
    data = request.json
    session_id = data.get('session_id')
    length = data.get('length', 50)
    
    if session_id not in camera_sessions:
        return jsonify({'error': '会话不存在'}), 404
    
    session = camera_sessions[session_id]
    session.set_trajectory_length(length)
    
    return jsonify({
        'status': 'success',
        'trajectory_length': session.max_trajectory_length,
        'message': f'轨迹长度已设置为 {session.max_trajectory_length}'
    })


atexit.register(cleanup_all_sessions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
