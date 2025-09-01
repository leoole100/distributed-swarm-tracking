# Distributed Swarm Tracking

Smartphone‑based multi‑camera system for background subtraction, multi‑view reprojection, and 3D tracking of birds/insects for collective behavior studies.
Inspired by [(YouTube: Dunking on Elon by actually tracking Stealth fighters using cheap webcameras without AI. #SoME4, 2025)](https://youtube.com/watch?v=zFiubdrJqqI).


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
        - RMS error of `1.6 ms` at `10 ms` ping, with a sync every `100 ms`.
        - revisit once RTC Data streams work. Implement 4 timestamps, best of, and PLL.
    3. Video Stream
        1. ~~get the publishing of the feeds to work~~
        2. ~~add time stamping~~
        3. add a option to preview the video feeds
    4. add recording and file readers.
2. try preprocessing, projection and detection algorithms offline. 
    1. get the last frame from each camera, if it is not older than let's say `100 ms` and assume they are concurrently. Alternative approaches can be implemented in the future.
    2. High pass filtering in time is the easiest way to subtract the background. 
        - Try other schemes in the future. [(OpenCV: Background Subtraction)](https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html)
    3. Voxel rendering, like [(YouTube: Dunking on Elon by actually tracking Stealth fighters using cheap webcameras without AI. #SoME4, 2025)](https://youtube.com/watch?v=zFiubdrJqqI), produces something similar to *limited angle tomography*. But seems costly. For $N$ captured pixels and $n^3$ Voxels the complexity is in the order $\mathcal{O}(N n)$ to create the volume.
    4. A alternative Approach is to do detections in the frame and filter out false positives. For $M$ detections (worst case the pixel number) with $k$ Cameras there have to be $\mathcal{O}(M^2 k)$ operations, as every detection has to be compared to each and the possible ray intersection evaluated. **TODO: Double Check**
    This is similar to $\mathcal{O}(M^2) > \mathcal{O}(N n)$, as $n<M\lesssim N$.

3. speed up the algorithms to run in real time and implement them to interface with the rest. 
4. just save the tracking data, to enable longer observations.