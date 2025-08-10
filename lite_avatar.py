import os
import numpy as np
import cv2
import json
import time
import librosa
import threading
import queue
from loguru import logger
import base64
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_silence
from torchvision import transforms
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d
import wave
import shutil
import subprocess
import concurrent.futures
import gc


def geneHeadInfo(sampleRate, bits, sampleNum):
    import struct
    rHeadInfo = b'\x52\x49\x46\x46'
    fileLength = struct.pack('i', sampleNum + 36)
    rHeadInfo += fileLength
    rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'
    rHeadInfo += struct.pack('i', sampleRate)
    rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))
    rHeadInfo += b'\x02\x00'
    rHeadInfo += struct.pack('H', bits)
    rHeadInfo += b'\x64\x61\x74\x61'
    rHeadInfo += struct.pack('i', sampleNum)
    return rHeadInfo

class liteAvatar(object):
    def __init__(self,
                 data_dir=None,
                 language='ZH',
                 a2m_path=None,
                 num_threads=1,
                 use_bg_as_idle=False,
                 fps=30,
                 generate_offline=False,
                 use_gpu=False,
                 batch_processing=True):
        
        logger.info('liteAvatar init start...')
        
        self.data_dir = data_dir
        self.fps = fps
        self.use_bg_as_idle = use_bg_as_idle
        self.use_gpu = use_gpu
        self.batch_processing = batch_processing
        self.device = "cuda" if use_gpu else "cpu"
        
        s = time.time()
        from audio2mouth_cpu import Audio2Mouth
        
        self.audio2mouth = Audio2Mouth(use_gpu)
        logger.info(f'audio2mouth init over in {time.time() - s}s')
        
        self.p_list = [str(ii) for ii in range(32)]
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.load_data_thread: threading.Thread = None

        logger.info('liteAvatar init over')
        self._generate_offline = generate_offline
        self.num_threads = num_threads
        if generate_offline:
            self.load_dynamic_model(data_dir)
        
    def stop_algo(self):
        pass

    def load_dynamic_model(self, data_dir):
        logger.info("start to load dynamic data")
        start_time = time.time()
        self.encoder = torch.jit.load(f'{data_dir}/net_encode.pt').to(self.device)
        self.generator = torch.jit.load(f'{data_dir}/net_decode.pt').to(self.device)

        # Combined loading - eliminate redundant load_data_sync
        self.load_data_combined(data_dir=data_dir, bg_frame_cnt=150)
        self.ref_data_list = [0 for x in range(150)]
        logger.info("load dynamic model in {:.3f}s", time.time() - start_time)

    def unload_dynamic_model(self):
        pass
    
    def load_data_combined(self, data_dir, bg_frame_cnt=None):
        """Combined loading function - eliminates redundant operations"""
        t = time.time()
        
        # Load basic data
        self.neutral_pose = np.load(f'{data_dir}/neutral_pose.npy')
        self.mouth_scale = None
    
        # Optimized video loading with lazy approach
        self.bg_video_path = f'{data_dir}/bg_video.mp4'
        bg_video = cv2.VideoCapture(self.bg_video_path)
        total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bg_video_frame_count = total_frames if bg_frame_cnt is None else min(bg_frame_cnt, total_frames)
        
        # Load only required frames
        self.bg_data_list = []
        for i in range(self.bg_video_frame_count):
            ret, img = bg_video.read()
            if ret:
                self.bg_data_list.append(img)
            else:
                break
        bg_video.release()
        
        # Face box data
        with open(f'{data_dir}/face_box.txt', 'r') as f:
            y1, y2, x1, x2 = f.readline().strip().split()
        self.y1, self.y2, self.x1, self.x2 = int(y1), int(y2), int(x1), int(x2)
        
        # Pre-compute merge mask
        self.merge_mask = (np.ones((self.y2-self.y1, self.x2-self.x1, 3)) * 255).astype(np.uint8)
        self.merge_mask[10:-10, 10:-10, :] *= 0
        self.merge_mask = cv2.GaussianBlur(self.merge_mask, (21, 21), 15)
        self.merge_mask = self.merge_mask / 255
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Batch load and process reference images
        self.ref_img_list = []
        logger.info(f'Loading {self.bg_video_frame_count} reference images...')
        
        # Batch process images
        ref_images = []
        for ii in range(self.bg_video_frame_count):
            img_file_path = os.path.join(data_dir, 'ref_frames', f'ref_{ii:05d}.jpg')
            image = cv2.imread(img_file_path)
            if image is not None:
                image = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
                ref_images.append(image)
        
        # Batch encode images - process all 150 images
        for image in tqdm(ref_images, desc="Encoding reference images"):
            ref_img = self.image_transforms(np.uint8(image))
            encoder_input = ref_img.unsqueeze(0).float().to(self.device)
            with torch.no_grad():  # Save memory
                x = self.encoder(encoder_input)
            # Encoder returns list - keep original format for generator compatibility
            self.ref_img_list.append(x)
        
        logger.info(f'Combined data loading completed in {time.time() - t:.3f}s')
    
    # Removed redundant load_data - functionality moved to load_data_combined
    
    def face_gen_loop(self, thread_id, barrier, in_queue, out_queue):
        """Face generation loop - selects batch or standard processing"""
        if self.batch_processing:
            # Use batch processing for better GPU efficiency
            self.face_gen_loop_batch(thread_id, barrier, in_queue, out_queue, batch_size=8)
        else:
            # Use original single-frame processing loop
            self.face_gen_loop_original(thread_id, barrier, in_queue, out_queue)
            
    def param2img(self, param_res, bg_frame_id, global_frame_id=0, is_idle=False):
        """Single frame processing - kept for compatibility"""
        param_val = []
        for key in param_res:
            val = param_res[key]
            param_val.append(val)
        param_val = np.asarray(param_val)
        
        with torch.no_grad():
            source_img = self.generator(self.ref_img_list[bg_frame_id], torch.from_numpy(param_val).unsqueeze(0).float().to(self.device))
            source_img = source_img.detach().to("cpu")
        
        return source_img
    
    def param2img_batch(self, param_batch_data):
        """Batch processing for maximum GPU efficiency"""
        if not param_batch_data:
            return []
        
        batch_size = len(param_batch_data)
        param_tensors = []
        
        # Prepare batch data
        for param_res, bg_frame_id, global_frame_id in param_batch_data:
            param_val = []
            for key in param_res:
                val = param_res[key]
                param_val.append(val)
            param_val = np.asarray(param_val)
            param_tensors.append(torch.from_numpy(param_val).float())
        
        # Stack param batch
        param_batch = torch.stack(param_tensors).to(self.device)
        
        # Process frames individually since generator expects List[Tensor] format
        result_frames = []
        with torch.no_grad():
            for i, (param_res, bg_frame_id, global_frame_id) in enumerate(param_batch_data):
                # Use individual param tensor and reference 
                param_single = param_batch[i:i+1]  # Keep batch dimension
                ref_single = self.ref_img_list[bg_frame_id]  # Already in List[Tensor] format
                
                source_img = self.generator(ref_single, param_single)
                source_img = source_img.detach().to("cpu")
                result_frames.append(source_img)
        
        return result_frames
    
    def face_gen_loop_original(self, thread_id, barrier, in_queue, out_queue):
        """Original single-frame face generation loop"""
        while True:
            data = in_queue.get()
            if data is None:
                in_queue.put(None)  # Re-queue for other threads
                break
            
            param_res, bg_frame_id, global_frame_id = data
            
            s = time.time()
            # Process single frame
            frame_tensor = self.param2img(param_res, bg_frame_id, global_frame_id)
            
            # Post-process frame
            full_img, mouth_img = self.merge_mouth_to_bg(frame_tensor, bg_frame_id)
            out_queue.put((global_frame_id, full_img, mouth_img))
            
            processing_time = time.time() - s
            logger.info(f'Thread {thread_id} processed frame {global_frame_id} in {processing_time:.3f}s')
        
        barrier.wait()
        if thread_id == 0:
            out_queue.put(None)  # Signal completion
    
    def get_idle_param(self):
        bg_param = self.neutral_pose
        tmp_json = {}
        for ii in range(len(self.p_list)):
            tmp_json[str(ii)] = float(bg_param[ii])
        return tmp_json
    
    def merge_mouth_to_bg(self, mouth_image, bg_frame_id, use_bg=False):
        mouth_image = (mouth_image / 2 + 0.5).clamp(0, 1)
        mouth_image = mouth_image[0].permute(1,2,0)*255
        
        mouth_image = mouth_image.numpy().astype(np.uint8)
        mouth_image = cv2.resize(mouth_image, (self.x2-self.x1, self.y2-self.y1))
        mouth_image = mouth_image[:,:,::-1]
        full_img = self.bg_data_list[bg_frame_id].copy()
        if not use_bg:
            full_img[self.y1:self.y2,self.x1:self.x2,:] = mouth_image * (1 - self.merge_mask) + full_img[self.y1:self.y2,self.x1:self.x2,:] * self.merge_mask
        full_img = full_img.astype(np.uint8)
        return full_img, mouth_image.astype(np.uint8)
    
    def interp_param(self, param_res, fps=25):
        old_len = len(param_res)
        new_len = int(old_len / 30 * fps + 0.5)
            
        interp_list = {}
        for key in param_res[0]:
            tmp_list = []
            for ii in range(len(param_res)):
                tmp_list.append(param_res[ii][key])
            tmp_list = np.asarray(tmp_list)
            
            
            x = np.linspace(0, old_len - 1, num=old_len, endpoint=True)
            newx = np.linspace(0, old_len - 1, num=new_len, endpoint=True)
            f = interp1d(x, tmp_list)
            y = f(newx)
            interp_list[key] = y
        
        new_param_res = []
        for ii in range(new_len):
            tmp_json = {}
            for key in interp_list:
                tmp_json[key] = interp_list[key][ii]
            new_param_res.append(tmp_json)
        
        return new_param_res
    
    def padding_last(self, param_res, last_end=None):
        bg_param = self.neutral_pose
        
        if last_end is None:
            last_end = len(param_res)
        
        padding_cnt = 5
        final_end = max(last_end + 5, len(param_res))
        param_res = param_res[:last_end]
        padding_list = []
        for ii in range(last_end, final_end):
            tmp_json = {}
            for key in param_res[-1]:
                kk = ii - last_end
                scale = max((padding_cnt - kk - 1) / padding_cnt, 0.0)
                
                end_value = bg_param[int(key)]
                tmp_json[key] = (param_res[-1][key] - end_value) * scale + end_value
            padding_list.append(tmp_json)
        
        print('padding_cnt:', len(padding_list))
        param_res = param_res + padding_list
        return param_res
    
    def audio2param(self, input_audio_file_path, prefix_padding_size=0, is_complete=False, audio_status=-1):
        # Direct file loading - no need to reconstruct WAV header
        input_audio, sr = sf.read(input_audio_file_path)
        
        param_res, _, _ = self.audio2mouth.inference(subtitles=None, input_audio=input_audio)
        
        # Use same audio data for silence detection - avoid reloading
        sil_scale = np.zeros(len(param_res))
        sound = AudioSegment.from_file(input_audio_file_path)
        start_end_list = detect_silence(sound, 500, -50)
        if len(start_end_list) > 0:
            for start, end in start_end_list:
                start_frame = int(start / 1000 * 30)
                end_frame = int(end / 1000 * 30)
                logger.info(f'silence part: {start_frame}-{end_frame} frames')
                sil_scale[start_frame:end_frame] = 1
        sil_scale = np.pad(sil_scale, 2, mode='edge')
        kernel = np.array([0.1,0.2,0.4,0.2,0.1])
        sil_scale = np.convolve(sil_scale, kernel, 'same')
        sil_scale = sil_scale[2:-2]
        self.make_silence(param_res, sil_scale)
        if self.fps != 30:
            param_res = self.interp_param(param_res, fps=self.fps)
        
        if is_complete:
            param_res = self.padding_last(param_res)
            
        return param_res
    
    def make_silence(self, param_res, sil_scale):
        bg_param = self.neutral_pose
        
        for ii in range(len(param_res)):
            for key in param_res[ii]:
                neu_value = bg_param[int(key)]
                param_res[ii][key] = param_res[ii][key] * (1 - sil_scale[ii]) + neu_value * sil_scale[ii]
        return param_res
    
    def handle(self, audio_file_path, result_dir, param_res=None):
        
        if param_res is None:
            param_res = self.audio2param(audio_file_path)
        
        # Create fresh queues and threads for each call
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Create fresh threads for this call
        threads_prep = []
        barrier_prep = threading.Barrier(self.num_threads, action=None, timeout=None)
        for i in range(self.num_threads):
            t = threading.Thread(target=self.face_gen_loop, args=(i, barrier_prep, input_queue, output_queue))
            threads_prep.append(t)
        
        # Start threads
        for t in threads_prep:
            t.daemon = True
            t.start()
        
        # Queue processing tasks
        for ii in range(len(param_res)):
            frame_id = ii
            if int(frame_id / self.bg_video_frame_count) % 2 == 0:
                frame_id = frame_id % self.bg_video_frame_count
            else:
                frame_id = self.bg_video_frame_count - 1 - frame_id % self.bg_video_frame_count
            input_queue.put((param_res[ii], frame_id, ii))
        
        # Signal end of input
        input_queue.put(None)
        
        tmp_frame_dir = os.path.join(result_dir, 'tmp_frames')
        if os.path.exists(tmp_frame_dir):
            shutil.rmtree(tmp_frame_dir, ignore_errors=True)
        os.makedirs(tmp_frame_dir, exist_ok=True)
        
        # Batch collect frames for more efficient I/O
        frame_data = []
        while True:
            res_data = output_queue.get()
            if res_data is None:
                break
            frame_data.append(res_data)
        
        # Sort and write frames in batch
        frame_data.sort(key=lambda x: x[0])  # Sort by frame index
        
        # Use threading for parallel frame writing
        import concurrent.futures
        def write_frame(frame_info):
            global_frame_index, frame_img, _ = frame_info
            target_path = f'{tmp_frame_dir}/{str(global_frame_index+1).zfill(5)}.jpg'
            cv2.imwrite(target_path, frame_img)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(write_frame, frame_data)
        
        # Wait for all processing threads to complete
        for p in threads_prep:
            p.join()
        
        # Cleanup memory after processing - threads and queues are automatically cleaned up
        # since they are local variables that go out of scope
        if self.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        
        # Use cross-platform path handling and subprocess for better error handling
        output_video = os.path.join(result_dir, 'test_demo.mp4')
        frame_pattern = os.path.join(tmp_frame_dir, '%05d.jpg')
        
        cmd = [
            'ffmpeg',
            '-r', '30',
            '-i', frame_pattern,
            '-i', audio_file_path,
            '-framerate', '30',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-b:v', '5000k',
            '-strict', 'experimental',
            '-loglevel', 'error',
            '-y',
            output_video
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Video generation completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise e
        
        # Clean up temporary frame directory after successful video generation
        try:
            shutil.rmtree(tmp_frame_dir, ignore_errors=True)
            logger.info("Temporary frames cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp frames: {e}")
    
    def handle_stream(self, audio_file_path, job_id):
        """Stream frames as they are generated - Progressive Streaming with Audio"""
        
        param_res = self.audio2param(audio_file_path)
        total_frames = len(param_res)
        audio_length_seconds = len(param_res) / self.fps
        
        # Read original audio for streaming
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file_path)
        
        # Calculate audio chunk size per frame
        audio_samples_per_frame = int(sample_rate / self.fps)
        
        # Yield initial info with audio metadata
        yield {
            "status": "started",
            "total_frames": total_frames,
            "audio_length": audio_length_seconds,
            "sample_rate": sample_rate,
            "audio_samples_per_frame": audio_samples_per_frame
        }
        
        # Create fresh queues and threads for streaming
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Create threads for this streaming call
        threads_prep = []
        barrier_prep = threading.Barrier(self.num_threads, action=None, timeout=None)
        for i in range(self.num_threads):
            # Use the same selector logic for streaming
            target_func = self.face_gen_loop_stream_batch if self.batch_processing else self.face_gen_loop_stream_original
            t = threading.Thread(target=target_func, args=(i, barrier_prep, input_queue, output_queue))
            threads_prep.append(t)
        
        # Start threads
        for t in threads_prep:
            t.daemon = True
            t.start()
        
        # Queue processing tasks
        for ii in range(total_frames):
            frame_id = ii
            if int(frame_id / self.bg_video_frame_count) % 2 == 0:
                frame_id = frame_id % self.bg_video_frame_count
            else:
                frame_id = self.bg_video_frame_count - 1 - frame_id % self.bg_video_frame_count
            input_queue.put((param_res[ii], frame_id, ii))
        
        # Signal end of input
        input_queue.put(None)
        
        # Stream frames as they complete
        frames_received = 0
        frame_buffer = {}  # Buffer to ensure ordered delivery
        next_frame_id = 0
        
        while frames_received < total_frames:
            try:
                res_data = output_queue.get(timeout=30)  # 30 second timeout
                if res_data is None:
                    break
                    
                global_frame_index, full_img, mouth_img = res_data
                frame_buffer[global_frame_index] = full_img
                frames_received += 1
                
                # Stream frames in order with synchronized audio
                while next_frame_id in frame_buffer:
                    frame_img = frame_buffer.pop(next_frame_id)
                    
                    # Extract audio chunk for this frame
                    audio_start_sample = next_frame_id * audio_samples_per_frame
                    audio_end_sample = min((next_frame_id + 1) * audio_samples_per_frame, len(audio_data))
                    
                    if audio_start_sample < len(audio_data):
                        audio_chunk = audio_data[audio_start_sample:audio_end_sample]
                        
                        # Convert audio chunk to base64 for streaming
                        import wave
                        from io import BytesIO
                        
                        # Create WAV data in memory
                        wav_buffer = BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1 if len(audio_chunk.shape) == 1 else audio_chunk.shape[1])
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
                        
                        wav_data = wav_buffer.getvalue()
                        audio_chunk_b64 = base64.b64encode(wav_data).decode('utf-8')
                    else:
                        audio_chunk_b64 = None
                    
                    yield {
                        "frame": frame_img,
                        "frame_id": next_frame_id,
                        "progress": int((next_frame_id + 1) / total_frames * 100),
                        "total_frames": total_frames,
                        "audio_chunk": audio_chunk_b64,
                        "audio_chunk_duration": 1.0 / self.fps  # Duration in seconds
                    }
                    
                    next_frame_id += 1
                    
            except queue.Empty:
                logger.warning("Timeout waiting for frame")
                break
        
        # Wait for all threads to complete
        for p in threads_prep:
            p.join(timeout=5)
        
        # Create final video from streamed frames if needed
        # This could be optional since frames are already streamed
        yield {
            "completed": True,
            "status": "completed",
            "total_frames_streamed": next_frame_id
        }
        
        # Cleanup memory
        if self.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
    
    def face_gen_loop_batch(self, thread_id, barrier, in_queue, out_queue, batch_size=8):
        """Batched face generation loop for maximum GPU efficiency"""
        while True:
            # Collect batch of frames
            batch_data = []
            
            # Try to collect batch_size frames
            for _ in range(batch_size):
                try:
                    data = in_queue.get(timeout=0.1)  # Short timeout to avoid blocking
                except queue.Empty:
                    break
                
                if data is None:
                    in_queue.put(None)  # Re-queue for other threads
                    break
                
                batch_data.append(data)
            
            # If no data collected, thread is done
            if not batch_data:
                break
            
            # Check for termination signal in batch
            if any(d is None for d in batch_data):
                # Re-queue non-None data
                for d in batch_data:
                    if d is not None:
                        in_queue.put(d)
                break
            
            s = time.time()
            
            # Process batch with GPU efficiency
            batch_frames = self.param2img_batch(batch_data)
            
            # Post-process each frame in batch
            for i, (frame_tensor, (param_res, bg_frame_id, global_frame_id)) in enumerate(zip(batch_frames, batch_data)):
                full_img, mouth_img = self.merge_mouth_to_bg(frame_tensor, bg_frame_id)
                out_queue.put((global_frame_id, full_img, mouth_img))
            
            processing_time = time.time() - s
            avg_time_per_frame = processing_time / len(batch_data)
            logger.info(f'Batch processed {len(batch_data)} frames in {processing_time:.3f}s ({avg_time_per_frame:.3f}s/frame avg)')
        
        barrier.wait()
        if thread_id == 0:
            out_queue.put(None)  # Signal completion
    
    def face_gen_loop_stream_batch(self, thread_id, barrier, in_queue, out_queue):
        """Batched face generation loop for streaming"""
        # Same as face_gen_loop_batch but for streaming
        self.face_gen_loop_batch(thread_id, barrier, in_queue, out_queue, batch_size=8)
    
    def face_gen_loop_stream_original(self, thread_id, barrier, in_queue, out_queue):
        """Original single-frame face generation loop for streaming"""  
        # Same as face_gen_loop_original
        self.face_gen_loop_original(thread_id, barrier, in_queue, out_queue)
    
    @staticmethod
    def read_wav_to_bytes(file_path):
        try:
            # 打开WAV文件
            with wave.open(file_path, 'rb') as wav_file:
                # 获取WAV文件的参数
                params = wav_file.getparams()
                print(f"Channels: {params.nchannels}, Sample Width: {params.sampwidth}, Frame Rate: {params.framerate}, Number of Frames: {params.nframes}")
                
                # 读取所有帧
                frames = wav_file.readframes(params.nframes)
                return frames
        except wave.Error as e:
            print(f"Error reading WAV file: {e}")
            return None
        

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--audio_file', type=str)
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    
    audio_file = args.audio_file
    tmp_frame_dir = args.result_dir
    
    lite_avatar = liteAvatar(data_dir=args.data_dir, num_threads=1, generate_offline=True)
    
    lite_avatar.handle(audio_file, tmp_frame_dir)
    
