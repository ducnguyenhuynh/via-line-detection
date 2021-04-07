# via-line-detection

repo này huấn luyện mạng phân đoạn vạch kẻ đường, được tích hợp trong dự án [via]().

## Công việc đã thực hiện

- [ ] Triển khai mạng với các frame-work khác nhau.
    - [x] Pytorch.
    - [ ] Onnx.
    - [ ] TensorRT.

- [x] Cung cấp dữ liệu.

- [x] Xây dựng mạng.
    - [x] PiNet

- [x] Huấn luyện mạng.
    - [x] PiNet

- [x] Cung cấp pre-train model.
    - [x] PiNet Pytorch
    - [ ] PiNet Onnx
    - [ ] PiNet TensorRT

- [ ] Xây dựng metrics đánh giá.

- [ ] Demo kết quả.
    - [x] Demo kết quả trên ảnh
    - [ ] Demo kết quả trên video

## Kết quả

![demo1](images_test/result.png "demo")

## Cài đặt môi trường

- Cài đặt python >= 3.6

- Cài đặt môi trường và tạo môi trường mới:
```
sudo pip install virtualenv

virtualenv venv

source venv/bin/activate
```
- Cài đặt thư viện: 
    Các thư viện yêu cầu trong requirements.txt 
```
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
via-lines_detection
├──dataset
|     ├── train.txt
|     ├── val.txt
|     ├── data
│     |   ├── *.txt
│     |   ├── *.jpg
├── images_test                          # put images you want to test here
│     ├── demo_image_.jpg   
|    
├── src
|    ├── *.py
|    ├── savefile
here                                    # put pre-train models here
│    |   ├── *.pkl
|
├── video                           # put videos your want to test here
|    ├── *.mp4
|
├── README.md
├── LICENSE               
├── demo_line_detection.py
├── demo_line_detection_onnx.py
├── demo_line_detection_trt.py
```
## Các bước huấn luyện mạng

B1: Tải dữ liệu.

B2: Xử lý dữ liệu.

B3: Xây dựng mạng.

B4: Viết Code augmenter.

B5: Xây dựng DataSeuqence bao gồm augment và xử lý dữ liệu.

B6: Viết metrics và hàm loss đánh giá.

B7: Huấn luyện.

B8: Chạy Demo.


