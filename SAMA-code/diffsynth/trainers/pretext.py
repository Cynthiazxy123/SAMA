import torch, os, hashlib, time
import numpy as np
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