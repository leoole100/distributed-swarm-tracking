# Distributed Swarm Tracking

Smartphone‑based multi‑camera system for 3D tracking of particles/birds/insects/planes/etc.

## Install
```bash
# Run the following command in the project root directory
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=local"
```

## Development

1. get synchronized video streaming and recording to work. Record a test dataset. 
    1. ~~HTML+WS server~~
    2. ~~Time distribution~~ 
        - _RMS error of `1.6 ms` at `10 ms` ping, with a sync every `100 ms`._
        - revisit once RTC Data streams work. Implement 4 timestamps, best of, and PLL.
    3. Video Stream
        1. ~~get the publishing of the feeds to work~~
        2. ~~add time stamping~~
        3. integrate time distribution and video stream
        - add a option to preview the video feeds
    4. add recording and file readers.
2. calibrate of the cameras.
    1. ~~intrinsic~~
    2. extrinsic
3. write detection and tracking algorithms
    - _Voxel Rendering like [(YouTube: Dunking on Elon by actually tracking Stealth fighters using cheap webcameras without AI. #SoME4, 2025)](https://youtube.com/watch?v=zFiubdrJqqI) is way more expensive than eager 2d detection filtered to 3d detection.\
    Let there for example be $n\sim 10$ Cameras with on average $N\sim10^6$ pixels or $D\sim 100$ possible detections and a voxel grid of $V\sim 10^3$ resolution.\
    The voxel algorithm will do $V^3 n N\sim 10^{16}$ projections (4x4 matrix vector multiplication + accumulate), a 2d detection and 3d filtering system will handle $n^2 D^2\sim 10^6$ detection-detection comparisons._
    1. ~~preprocessing [(OpenCV: Background Subtraction)](https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html)~~
    2. 2d detections
    3. 3d detections / filtering
3. speed up the algorithms to run in real time and integrate them. 