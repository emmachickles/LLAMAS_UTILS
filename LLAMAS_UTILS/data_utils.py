import numpy as np

def stretch_img(img, lo=5, up=95):
    log_img = np.log10(img - np.min(img) + 1)
    log_min = np.percentile(log_img, lo) # Clip bottom lo% in log scale
    log_max = np.percentile(log_img, up)  # Clip top up% in log scale
    clipped_log_img = np.clip(log_img, log_min, log_max)
    return cliped_log_img
