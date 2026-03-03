# Realtime Object Detection Example

## Setup

```
cd object-detection/web-server
npm ci
```

### Run

```sh
mkfifo /tmp/demo-video.fifo

# Terminal 1
cat /tmp/demo-video.fifo | RUST_LOG=kafu_serve=debug,kafu_runtime=debug kafu_singlenode serve ./kafu-config.yaml | nc localhost 9998

# Terminal 2
ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i path/to/video.mp4 \
  -vf scale=352:352,fps=3 -pix_fmt rgb24 -f rawvideo - \
  | tee /tmp/demo-video.fifo \
  | node web-server/server.js 352 352
```
