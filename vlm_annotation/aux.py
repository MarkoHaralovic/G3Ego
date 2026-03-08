from projectaria_tools.core.sensor_data import TimeDomain
import numpy as np 

def stream_rgb_vrs_recording(aea_data_provider, rgb_stream_id):
    timestamps = aea_data_provider.vrs.get_timestamps_ns(
        rgb_stream_id, TimeDomain.DEVICE_TIME
    )

    images = []
    for idx in range(len(timestamps)):
        img = aea_data_provider.vrs.get_image_data_by_index(rgb_stream_id, idx)[0]
        images.append(img.to_numpy_array())

    return images, timestamps


def get_every_nth_second_frame(timestamps_ns, n_seconds=1):
    if not isinstance(timestamps_ns, np.ndarray):
        timestamps_ns = np.array(timestamps_ns)

    timestamps_sec = (timestamps_ns - timestamps_ns[0]) / 1e9

    max_time = int(timestamps_sec[-1])
    target_times = np.arange(0, max_time + n_seconds + 1, n_seconds)

    sampled_indices = []
    for target in target_times:
        indexes = np.argmin(np.abs(timestamps_sec - target))
        sampled_indices.append(indexes)
    sampled_indices = sorted(set(sampled_indices))

    total_frames = len(timestamps_ns)
    sampled_frames = len(sampled_indices)
    percentage = (sampled_frames / total_frames) * 100

    print(
        f"Total frames: {total_frames}, sampled frames: {sampled_frames}, sampling each {int(total_frames/sampled_frames)}th frame, pct : {percentage:.2f}%"
    )
    return np.array(sampled_indices)