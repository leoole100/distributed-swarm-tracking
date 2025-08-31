Smartphone‑based multi‑camera system for background subtraction, multi‑view reprojection, and 3D tracking of birds/insects for collective behavior studies.

```bash
# Run the following command in the project root directory
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=local"
```

1. get synchronized video streaming and recording to work. Record a test dataset. 
    1. ~~HTML+WS server~~
    2. ~~Time distribution~~ 
        - RMS error of `1.6 ms` at `10 ms` ping, with a sync every `100 ms`.
        - revisit once RTC Data streams work. Implement 4 timestamps, best of, and PLL.
    3. Video Stream
2. try preprocessing, projection and detection algorithms offline. 
3. speed up the algorithms to run in real time and implement them to interface with the rest. 
4. just save the tracking data, to enable longer observations.

Video streams are usually `640x480` and at roughly `2 Mbit/s`. Reported by the [Janus example](https://janus.conf.meetecho.com/demos/mvideoroom.html).