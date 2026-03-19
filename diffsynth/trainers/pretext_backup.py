import torch, torchvision, imageio, os, json, pandas, hashlib, signal, time
import numpy as np
import imageio.v3 as iio
from PIL import Image



class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


# class LoadVideo(DataProcessingOperator):
#     def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
#         self.num_frames = num_frames
#         self.time_division_factor = time_division_factor
#         self.time_division_remainder = time_division_remainder
#         # frame_processor is build in the video loader for high efficiency.
#         self.frame_processor = frame_processor
        
#     def get_num_frames(self, reader):
#         num_frames = self.num_frames
#         if int(reader.count_frames()) < num_frames:
#             num_frames = int(reader.count_frames())
#             while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
#                 num_frames -= 1
#         return num_frames
        
#     def __call__(self, data: str):
#         reader = imageio.get_reader(data)
#         num_frames = self.get_num_frames(reader)
#         frames = []
#         for frame_id in range(num_frames):
#             frame = reader.get_data(frame_id)
#             frame = Image.fromarray(frame)
#             frame = self.frame_processor(frame)
#             frames.append(frame)
#         reader.close()
#         return frames

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        try:
            total = int(reader.count_frames())
        except Exception:
            total = None
        if total is not None and total < num_frames:
            num_frames = int(total)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames
        
    def _decode_once(self, data: str):
        reader = imageio.get_reader(data)
        try:
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                try:
                    frame = reader.get_data(frame_id)
                except Exception:
                    break
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            if len(frames) == 0:
                raise RuntimeError("No frames decoded.")
            if len(frames) > 1:
                while len(frames) > 1 and len(frames) % self.time_division_factor != self.time_division_remainder:
                    frames.pop()
            return frames
        finally:
            reader.close()

    def _decode_with_timeout(self, data: str, timeout_sec: int):
        if timeout_sec <= 0 or not hasattr(signal, "SIGALRM"):
            return self._decode_once(data)
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"LoadVideo timeout after {timeout_sec}s: {data}")
        old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            return self._decode_once(data)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def __call__(self, data: str):
        max_retries = int(os.environ.get("WAN_LOADVIDEO_RETRIES", "3"))
        backoff_base = float(os.environ.get("WAN_LOADVIDEO_BACKOFF_SEC", "0.5"))
        timeout_sec = int(os.environ.get("WAN_LOADVIDEO_TIMEOUT_SEC", "60"))
        max_retries = max(1, max_retries)
        last_err = None
        for attempt in range(max_retries):
            try:
                return self._decode_with_timeout(data, timeout_sec)
            except Exception as err:
                last_err = err
                if attempt + 1 < max_retries:
                    time.sleep(backoff_base * (2 ** attempt))
        raise last_err

class SwapFourBlocks(DataProcessingOperator):
    """
    Split T/H/W into blocks and randomly shuffle spatiotemporal tubes.
    """

    def __init__(self, t_splits=4, h_splits=4, w_splits=4, seed=0, keep_last_frame=False):
        self.t_splits = int(t_splits)
        self.h_splits = int(h_splits)
        self.w_splits = int(w_splits)
        self.seed = None if seed is None else int(seed)
        self.keep_last_frame = bool(keep_last_frame)
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)

    def _split_sizes(self, n, parts):
        base = n // parts
        rem = n % parts
        return [base + (1 if i < rem else 0) for i in range(parts)]

    def __call__(self, data):
        if data is None or not isinstance(data, list):
            return data
        t_len = len(data)
        if t_len < self.t_splits:
            return data
        first = np.asarray(data[0])
        if first.ndim < 2:
            return data
        h, w = first.shape[0], first.shape[1]
        if h < self.h_splits or w < self.w_splits:
            return data

        shapes = [np.asarray(f).shape for f in data]
        if any(s != shapes[0] for s in shapes):
            return data

        arr = np.stack([np.asarray(f) for f in data], axis=0)
        if self.keep_last_frame and t_len > 1:
            arr_main = arr[:-1]
            last_frame = arr[-1:]
        else:
            arr_main = arr
            last_frame = None
        t_len_eff = arr_main.shape[0]
        if t_len_eff < self.t_splits:
            return data

        t_sizes = self._split_sizes(t_len_eff, self.t_splits)
        h_sizes = self._split_sizes(h, self.h_splits)
        w_sizes = self._split_sizes(w, self.w_splits)
        if min(t_sizes) == 0 or min(h_sizes) == 0 or min(w_sizes) == 0:
            return data

        t_ranges = []
        h_ranges = []
        w_ranges = []
        t0 = 0
        for s in t_sizes:
            t_ranges.append((t0, t0 + s))
            t0 += s
        h0 = 0
        for s in h_sizes:
            h_ranges.append((h0, h0 + s))
            h0 += s
        w0 = 0
        for s in w_sizes:
            w_ranges.append((w0, w0 + s))
            w0 += s

        blocks = []
        ranges = []
        shapes = []
        for t0, t1 in t_ranges:
            for h0, h1 in h_ranges:
                for w0, w1 in w_ranges:
                    ranges.append((t0, t1, h0, h1, w0, w1))
                    block = arr_main[t0:t1, h0:h1, w0:w1, ...]
                    blocks.append(block)
                    shapes.append(block.shape)

        groups = {}
        for idx, shape in enumerate(shapes):
            groups.setdefault(shape, []).append(idx)

        perm_map = list(range(len(blocks)))
        for idxs in groups.values():
            if len(idxs) <= 1:
                continue
            if self._generator is None:
                perm = torch.randperm(len(idxs)).tolist()
            else:
                perm = torch.randperm(len(idxs), generator=self._generator).tolist()
            shuffled = [idxs[p] for p in perm]
            for dst_idx, src_idx in zip(idxs, shuffled):
                perm_map[dst_idx] = src_idx
        out_main = np.empty_like(arr_main)
        for dst_idx, src_idx in enumerate(perm_map):
            t0, t1, h0, h1, w0, w1 = ranges[dst_idx]
            out_main[t0:t1, h0:h1, w0:w1, ...] = blocks[src_idx]
        out = out_main if last_frame is None else np.concatenate([out_main, last_frame], axis=0)

        if os.environ.get("WAN_DEBUG_SWAP") == "1" and not getattr(self, "_debug_logged", False):
            preview = min(4, t_len)
            before = [hashlib.md5(np.asarray(f).tobytes()).hexdigest()[:8] for f in data[:preview]]
            after = [hashlib.md5(out[i].tobytes()).hexdigest()[:8] for i in range(preview)]
            print(
                f"[SwapFourBlocks] T/H/W splits=({self.t_splits},{self.h_splits},{self.w_splits}) "
                f"seed={self.seed} keep_last_frame={self.keep_last_frame} "
                f"t_sizes={t_sizes} h_sizes={h_sizes} w_sizes={w_sizes} "
                f"before={before} after={after}",
                flush=True,
            )
            save_dir = os.environ.get("WAN_DEBUG_SWAP_DIR")
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                for i in range(preview):
                    Image.fromarray(out[i]).save(os.path.join(save_dir, f"tube_after_{i:02d}.png"))
                    data[i].save(os.path.join(save_dir, f"tube_before_{i:02d}.png"))
            self._debug_logged = True

        return [Image.fromarray(out[i]) for i in range(out.shape[0])]


# class MaskRandomFrames(DataProcessingOperator):
#     """
#     Randomly mask a ratio of frames to black.
#     """

#     def __init__(self, mask_ratio=0.3, seed=0):
#         self.mask_ratio = float(mask_ratio)
#         self.seed = None if seed is None else int(seed)
#         self._generator = None
#         if self.seed is not None:
#             self._generator = torch.Generator()
#             self._generator.manual_seed(self.seed)

#     def __call__(self, data):
#         if data is None or not isinstance(data, list):
#             return data
#         n = len(data)
#         if n == 0 or self.mask_ratio <= 0:
#             return data
#         k = int(round(n * self.mask_ratio))
#         if k <= 0:
#             return data
#         k = min(k, n)
#         if self._generator is None:
#             perm = torch.randperm(n)
#         else:
#             perm = torch.randperm(n, generator=self._generator)
#         mask_idx = set(perm[:k].tolist())
#         out = []
#         for i, frame in enumerate(data):
#             if i in mask_idx:
#                 out.append(Image.new(frame.mode, frame.size))
#             else:
#                 out.append(frame)
#         return out

class MaskRandomFrames(DataProcessingOperator):
    """
    Randomly mask ONE contiguous segment of frames to black.
    Exclude the first frame (index 0) from being masked.
    """

    def __init__(self, mask_ratio=0.3, seed=0):
        self.mask_ratio = float(mask_ratio)
        self.seed = None if seed is None else int(seed)
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)

    def __call__(self, data):
        if data is None or not isinstance(data, list):
            return data
        n = len(data)
        if n <= 1 or self.mask_ratio <= 0:
            return data

        # 计算要 mask 的连续长度 k
        k = int(round(n * self.mask_ratio))
        k = max(0, min(k, n))  # clamp
        if k <= 0:
            return data

        # 首帧不能被 mask，所以最多只能 mask n-1 帧
        k = min(k, n - 1)
        if k <= 0:
            return data

        # 可选起点：1..(n-k)  (保证区间 [start, start+k) 不越界，且 start != 0)
        start_min = 1
        start_max = n - k  # inclusive
        if start_max < start_min:
            # 理论上不会发生（因为 k<=n-1），保险处理
            return data

        if self._generator is None:
            start = int(torch.randint(start_min, start_max + 1, (1,)).item())
        else:
            start = int(torch.randint(start_min, start_max + 1, (1,), generator=self._generator).item())

        end = start + k  # [start, end)

        out = []
        for i, frame in enumerate(data):
            if start <= i < end:
                out.append(Image.new(frame.mode, frame.size))
            else:
                out.append(frame)
        return out


class TimeWarpFrames(DataProcessingOperator):
    """
    Apply a random time-warp with speed factor s and resample back to length T.
    """

    def __init__(self, speed_choices=None, speed_range=None, seed=0):
        if speed_choices is None and speed_range is None:
            speed_choices = (0.5, 2.0)
        self.speed_choices = None if speed_choices is None else tuple(float(x) for x in speed_choices)
        self.speed_range = None if speed_range is None else (float(speed_range[0]), float(speed_range[1]))
        self.seed = None if seed is None else int(seed)
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)

    def _rand(self):
        if self._generator is None:
            return torch.rand(1).item()
        return torch.rand(1, generator=self._generator).item()

    def _randint(self, low, high):
        if self._generator is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._generator).item())

    def _sample_speed(self):
        if self.speed_range is not None:
            s0, s1 = self.speed_range
            if s1 <= 0:
                return None
            r = self._rand()
            return s0 + (s1 - s0) * r
        if not self.speed_choices:
            return None
        idx = self._randint(0, len(self.speed_choices))
        return self.speed_choices[idx]

    def __call__(self, data):
        if data is None or not isinstance(data, list):
            return data
        n = len(data)
        if n < 2:
            return data
        s = self._sample_speed()
        if s is None or s <= 0:
            return data
        out = []
        last = n - 1
        for t in range(n):
            if last == 0:
                src_idx = 0
            else:
                u = float(t) / float(last)
                src_time = u * s
                if src_time > 1.0:
                    src_time = 1.0
                src_idx = int(round(src_time * last))
            out.append(data[src_idx])
        return out

        
class SpeedChangeFrames(DataProcessingOperator):
    """
    Change playback speed by resampling frames.
    Deterministic: alternates between two speeds (default 0.5x, 2x).
    """

    def __init__(self, speed_choices=None, speed_range=None, seed=0, keep_length=False):
        if speed_choices is None and speed_range is None:
            speed_choices = (0.5, 2.0)
        self.speed_choices = None if speed_choices is None else tuple(float(x) for x in speed_choices)
        self.speed_range = None if speed_range is None else (float(speed_range[0]), float(speed_range[1]))
        self.seed = None if seed is None else int(seed)
        self.keep_length = bool(keep_length)
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self._toggle = False

    def _rand(self):
        if self._generator is None:
            return torch.rand(1).item()
        return torch.rand(1, generator=self._generator).item()

    def _randint(self, low, high):
        if self._generator is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._generator).item())

    def _sample_speed(self):
        if self.speed_range is not None:
            s0, s1 = self.speed_range
            if s1 <= 0:
                return None
            # deterministic alternate between range endpoints
            self._toggle = not self._toggle
            return s1 if self._toggle else s0
        if not self.speed_choices:
            return None
        if len(self.speed_choices) == 1:
            return self.speed_choices[0]
        self._toggle = not self._toggle
        return self.speed_choices[1] if self._toggle else self.speed_choices[0]

    def __call__(self, data):
        if data is None or not isinstance(data, list):
            return data
        n = len(data)
        if n < 2:
            return data
        s = self._sample_speed()
        if s is None or s <= 0:
            return data
        out = []
        last = n - 1
        if self.keep_length:
            for t in range(n):
                if last == 0:
                    src_idx = 0
                else:
                    u = float(t) / float(last)
                    src_time = u * s
                    if src_time > 1.0:
                        src_time = 1.0
                    src_idx = int(round(src_time * last))
                out.append(data[src_idx])
            return out

        n_out = int(round(n / s))
        if n_out <= 0:
            n_out = 1
        for i in range(n_out):
            src_idx = int(round(i * s))
            if src_idx > last:
                src_idx = last
            out.append(data[src_idx])
        # Align to VAE time division: (T - 1) % 4 == 0
        if len(out) > 1:
            rem = (len(out) - 1) % 4
            if rem != 0:
                target_len = max(1, len(out) - rem)
                out = out[:target_len]
        return out

class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
