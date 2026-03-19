import math
def find_closest_resolution(w, h, ratios, resolutions):
    """
    给定输入的宽高 (w, h)，找到 RESOLUTIONS 中宽高比最接近的分辨率。
    
    Args:
        w: 输入宽度
        h: 输入高度
    
    Returns:
        (target_w, target_h): 最接近的目标分辨率
    """
    RESOLUTION_ARS = ratios
    input_ar = w / h
    
    best_idx = 0
    best_diff = float('inf')
    
    for i, res_ar in enumerate(RESOLUTION_ARS):
        diff = abs(res_ar - input_ar)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    
    return resolutions[best_idx]

def get_all_resolution_new(target_pixels, factor=32):
    """Get all possible resolutions"""
    all_resolution = []
    image_size = math.sqrt(target_pixels)
    divided_by = factor
    min_edge = int(image_size / 1.4)
    max_edge = int(image_size * 1.4)
    token_number = image_size * image_size / divided_by / divided_by
    for i in range(min_edge // divided_by, max_edge // divided_by + 1):
        all_resolution.append([i * divided_by, int(token_number // i * divided_by)])
    print(all_resolution)
    return all_resolution


def get_all_resolution(target_pixels, factor=32, min_ar=0.5, max_ar=2.0):

    print(f"目标像素数: {target_pixels}")
    print(f"宽高比范围: {min_ar} ~ {max_ar}")
    print("-" * 60)

    # 计算 height 的理论范围
    # ar = w/h, w*h = pixels
    # => h = sqrt(pixels/ar)
    # 当 ar 最大时，h 最小；当 ar 最小时，h 最大
    h_min = math.sqrt(target_pixels / max_ar)  # ar=2.0 时 h 最小
    h_max = math.sqrt(target_pixels / min_ar)  # ar=0.5 时 h 最大

    print(f"Height 理论范围: {h_min:.1f} ~ {h_max:.1f}")

    # 对齐到 32 的倍数
    h_min_aligned = math.floor(h_min / factor) * factor
    h_max_aligned = math.ceil(h_max / factor) * factor

    print(f"Height 对齐后范围: {h_min_aligned} ~ {h_max_aligned}")
    print("-" * 60)

    resolutions = []

    # 枚举所有 height (32 的倍数)
    for h in range(h_min_aligned, h_max_aligned + 1, factor):
        # 根据目标像素数计算 width
        w_float = target_pixels / h
        
        # 对齐到 32 的倍数
        w = round(w_float / factor) * factor
        
        # 检查宽高比是否在范围内
        ar = w / h
        if min_ar <= ar <= max_ar:
            resolutions.append((w, h))
    print(resolutions)
    return resolutions


# 测试
if __name__ == "__main__":
    test_cases = [
        (1920, 1080),  # 16:9 -> ar=1.78
        (1080, 1920),  # 9:16 -> ar=0.56
        (1024, 1024),  # 1:1  -> ar=1.00
        (800, 600),    # 4:3  -> ar=1.33
        (600, 800),    # 3:4  -> ar=0.75
        (1280, 720),   # 16:9 -> ar=1.78
        (720, 1280),   # 9:16 -> ar=0.56
    ]
    resolutions = get_all_resolution(832 * 480, min_ar=0.5, max_ar=2.0)
    # resolutions = get_all_resolution_new(832 * 480, 64)
    ratios = [(w / h) for w, h in resolutions]
    
    print("输入分辨率 -> 匹配的目标分辨率")
    print("-" * 50)
    for w, h in test_cases:
        target_w, target_h = find_closest_resolution(w, h, ratios, resolutions)
        input_ar = w / h
        target_ar = target_w / target_h
        print(f"{w:4d}x{h:4d} (ar={input_ar:.2f}) -> {target_w}x{target_h} (ar={target_ar:.2f})")