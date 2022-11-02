"""Convert all yuv420 files to jpg file in target folder.

Target folder is current directory if there is no argument after command line.

Usage example:

  python3 yuv420_to_img.py
  python3 yuv420_to_img.py path/to/target/folder
"""
import os
import sys
import cv2
import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT * 3 / 2)

Y_WIDTH = IMG_WIDTH
Y_HEIGHT = IMG_HEIGHT
Y_SIZE = int(Y_WIDTH * Y_HEIGHT)

U_V_WIDTH = int(IMG_WIDTH / 2)
U_V_HEIGHT = int(IMG_HEIGHT / 2)
U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)


def read_yuv420(yuv_data, frames):
  """Read yuv420 file to y, u, v.

  Args:
    yuv_data: Absolute path of target file.
  Returns:
    y, u, v: luma, chrominance, chroma.
  """
  y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
  u = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
  v = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

  for frame_idx in range(0, frames):
    y_start = frame_idx * IMG_SIZE
    u_v_start = y_start + Y_SIZE
    u_v_end = u_v_start + (U_V_SIZE * 2)

    y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
    u_v = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
    v[frame_idx, :, :] = u_v[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
    u[frame_idx, :, :] = u_v[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
  return y, u, v


def yuv2rgb(y, u, v):
  """convert y, u, v to bgr_data.

  Args:
    y: luma obtained from read_yuv420()
    u: chrominance obtained from read_yuv420()
    v: chroma obtained from read_yuv420()
  Returns:
    bgr_data: convert result.
  """
  bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
  v = np.repeat(v, 2, 0)
  v = np.repeat(v, 2, 1)
  u = np.repeat(u, 2, 0)
  u = np.repeat(u, 2, 1)

  c = (y - np.array([16])) * 298
  d = u - np.array([128])
  e = v - np.array([128])

  b = (c + 409 * e + 128) // 256
  g = (c - 100 * d - 208 * e + 128) // 256
  r = (c + 516 * d + 128) // 256

  r = np.where(r < 0, 0, r)
  r = np.where(r > 255, 255, r)

  g = np.where(g < 0, 0, g)
  g = np.where(g > 255, 255, g)

  b = np.where(b < 0, 0, b)
  b = np.where(b > 255, 255, b)

  bgr_data[:, :, 2] = r
  bgr_data[:, :, 1] = g
  bgr_data[:, :, 0] = b

  return bgr_data


def yuv420_to_img(file):
  """Convert yuv420 file to image.

  Generate the same name .jpg from .yuv420.

  single frame case:
  example1.yuv420 -> example1.jpg, example2.yuv420 -> example2.jpg ...

  multi-frame case:
  example1.yuv420 -> example1_0.jpg, example1_1.jpg ...

  Args:
    file: Absolute path of target file.
  """
  frames = int(os.path.getsize(file) / IMG_SIZE)
  with open(file, "rb") as f:
    yuv_bytes = f.read()
    yuv_data = np.frombuffer(yuv_bytes, np.uint8)
    y, u, v = read_yuv420(yuv_data, frames)
    for frame_idx in range(frames):
      bgr_data = yuv2rgb(y[frame_idx, :, :], u[frame_idx, :, :], v[frame_idx, :, :])
      if bgr_data is not None:
        if frames == 1: # single frame case
          cv2.imwrite("{}.jpg".format(file.split(".")[0]), bgr_data)
        else: # multi-frame case
          cv2.imwrite("{}_{}.jpg".format(file.split(".")[0], frame_idx), bgr_data)


def main():
  args = sys.argv[1:]
  if args:
    target_folder = os.path.abspath(args[-1])
  else:
    target_folder = os.path.abspath(".")

  target_files = os.listdir(target_folder)
  for target_file in target_files:
    if target_file.split(".")[-1] == "yuv420":
      yuv420_to_img(os.path.join(target_folder, target_file))

if __name__ == "__main__":
  main()
