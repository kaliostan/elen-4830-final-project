import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from math import sqrt




def compression_KMeans(input_image, output_folder, k):
    image = Image.open(input_image)
    image = image.convert('L')    # grayscale
    is_grayscale = image.mode == "L"

    if is_grayscale:
        image_np = np.array(image)
        pixels = image_np.reshape(-1, 1)
        image_type = "Grayscale"
    else:
        image = image.convert("RGB")
        image_np = np.array(image)
        pixels = image_np.reshape(-1, 3)
        image_type = "RGB"

    # Apply K-Means
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto') # use the same random seed to avoid changes
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_.astype("uint8")

    # Format and save the compressed image
    compressed_pixels = new_colors[kmeans.labels_]
    compressed_image = compressed_pixels.reshape(image_np.shape)

    image_name = os.path.splitext(os.path.basename(input_image))[0]
    output_path = os.path.join(output_folder, f"{image_name}_compressed.jpg")
    mode = "L" if is_grayscale else "RGB"
    
    compressed_image_PIL = Image.fromarray(compressed_image, mode=mode)
    compressed_image_PIL.save(output_path)
    end_time = time.time()


    height, width = image_np.shape[:2]
    original_size = os.path.getsize(input_image) / 1024
    compressed_size = os.path.getsize(output_path) / 1024
    compression_ratio = original_size / compressed_size
    mse = np.mean((image_np.astype("float32") - compressed_image.astype("float32")) ** 2)
    rmse = sqrt(mse)
    psnr = cv2.PSNR(image_np, compressed_image)

    if is_grayscale:
        SSIM = ssim(image_np, compressed_image, data_range=255)
    else:
        SSIM = ssim(image_np, compressed_image, data_range=255, channel_axis=-1)

    global total_CR, total_RMSE, total_PSNR, total_SSIM, total_time, image_count
    total_CR += compression_ratio
    total_RMSE += rmse
    total_PSNR += psnr
    total_SSIM += SSIM
    total_time += (end_time - start_time)
    image_count += 1

    return {
        "image_name": os.path.basename(input_image),
        "k": k,
        "image_type": image_type,
        "width": width,
        "height": height,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": SSIM,
        "time": end_time - start_time
    }





input_folder = "sample_image"
output_folder = "compressed_image"
k_cluster = [5, 10, 15, 20, 25, 30]

output_txt = os.path.join(output_folder, "compression_average.txt")

avg_CR_list = []
avg_RMSE_list = []
avg_PSNR_list = []
avg_SSIM_list = []
avg_time_list = []

with open(output_txt, "w") as f:
    for k in k_cluster:
        total_CR = 0.0
        total_RMSE = 0.0
        total_PSNR = 0.0
        total_SSIM = 0.0
        total_time = 0.0
        image_count = 0

        for image_name in os.listdir(input_folder):
            ext = os.path.splitext(image_name)[1].lower()
            if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
                input_image = os.path.join(input_folder, image_name)
                compression_KMeans(input_image, output_folder, k)

        if image_count == 0:
            print(f"No images processed for k={k}.")
            continue

        avg_CR = total_CR / image_count
        avg_RMSE = total_RMSE / image_count
        avg_PSNR = total_PSNR / image_count
        avg_SSIM = total_SSIM / image_count
        avg_time = total_time / image_count * 1000

        avg_CR_list.append(avg_CR)
        avg_RMSE_list.append(avg_RMSE)
        avg_PSNR_list.append(avg_PSNR)
        avg_SSIM_list.append(avg_SSIM)
        avg_time_list.append(avg_time)

        # Save results
        avg_line = (
            f"K={k} | "
            f"Avg CR: {avg_CR:.2f}, "
            f"Avg RMSE: {avg_RMSE:.2f}, "
            f"Avg PSNR: {avg_PSNR:.2f} dB, "
            f"Avg SSIM: {avg_SSIM:.4f}, "
            f"Avg Time: {avg_time:.2f} ms\n"
            f"Total Time: {total_time:.2f} s\n"
        )
        print(avg_line)
        f.write(avg_line)

# plt.plot(k_cluster, avg_CR_list, label="avg compression ratio", color='red')
# plt.plot(k_cluster, avg_RMSE_list, label="avg rmse", linestyle='--', color='green')
# plt.plot(k_cluster, avg_PSNR_list, label="avg psnr (dB)", color='blue', marker='o')
# plt.plot(k_cluster, avg_SSIM_list, label="avg ssim", color='brown', marker='x')
# plt.plot(k_cluster, avg_time_list, label="avg compression time (ms)", color='black')
#
# plt.xlabel("K-Means k clusters")
# plt.ylabel("Metrics")
# plt.title(f"Performance metrics vs K (KMeans RGB Compression) {image_count} image(s)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

fig, sub = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top subplot: image quality metrics
sub[0].plot(k_cluster, avg_CR_list, label="avg compression ratio", color='red')
sub[0].plot(k_cluster, avg_RMSE_list, label="avg rmse", linestyle='--', color='green')
sub[0].plot(k_cluster, avg_PSNR_list, label="avg psnr (dB)", color='blue', marker='o')
sub[0].plot(k_cluster, avg_SSIM_list, label="avg ssim", color='brown', marker='x')
sub[0].set_ylabel("Metrics")
sub[0].set_title(f"Performance Metrics KMeans Grayscale Compression - {image_count} image(s)")
sub[0].grid(True)
sub[0].legend()

# Bottom subplot: compression time
sub[1].plot(k_cluster, avg_time_list, label="avg compression time (ms)", color='black', marker='s')
sub[1].set_xlabel("K-Means k clusters")
sub[1].set_ylabel("Time (ms)")
sub[1].grid(True)
sub[1].legend()

plt.tight_layout()
plt.show()


print("Compression Completed")






















# def compress_image(input_folder, output_folder, k, extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
#     text = []
#
#     for image_name in os.listdir(input_folder):
#         ext = os.path.splitext(image_name)[1].lower()
#         if ext in extensions:
#             input_image = os.path.join(input_folder, image_name)
#             info = compression_KMeans(input_image, output_folder, k=k)
#
#             # Compose data text
#             data = (
#                 f"--- {info['image_name']} ---\n"
#                 f"Original Image: {info['width']}x{info['height']}, {info['image_type']}, {info['original_size']:.2f} KB\n"
#                 f"Compressed Image: {info['compressed_size']:.2f} KB\n"
#                 f"Compression Ratio: {info['compression_ratio']:.2f}\n"
#                 f"MSE: {info['mse']:.2f}\n"
#                 f"PSNR: {info['psnr']:.2f} dB\n"
#                 f"SSIM: {info['ssim']:.4f}\n"
#                 f"Compression Time: {info['time']:.2f} seconds\n\n"
#             )
#             print(data)
#             text.append(data)
#
#     # save to .txt
#     os.makedirs(output_folder, exist_ok=True)
#     with open(os.path.join(output_folder, "compression_data.txt"), "w") as f:
#         f.writelines(text)


# input_folder = "sample_image"
# output_folder = "compressed_image"
#
# total_mse = 0.0
# total_psnr = 0.0
# total_ssim = 0.0
# total_CR = 0.0
# total_time = 0.0
# image_count = 0
#
# compress_image(input_folder, output_folder, k=32)
#
# average = (
#     f"\n=== AVERAGE RESULTS FOR {image_count} IMAGES ===\n"
#     f"Average Compression Ratio: {total_CR / image_count:.2f}%\n"
#     f"Average MSE: {total_mse / image_count:.2f}\n"
#     f"Average PSNR: {total_psnr / image_count:.2f} dB\n"
#     f"Average SSIM: {total_ssim / image_count:.4f}\n"
#     f"Average Compression Time: {total_time / image_count:.2f} seconds\n"
#     f"Total Compression Time: {total_time:.2f} seconds\n"
# )
# print(average)
#
# # save to .txt
# with open(os.path.join(output_folder, "compression_average.txt"), "w") as f:
#     f.write(average)
#
# print ("Compression Completed")
