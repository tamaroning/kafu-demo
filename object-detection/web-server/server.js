// server.js
// Receives: (1) raw RGB24 video on stdin, (2) detection NDJSON lines on TCP (detectionPort).
// WebSocket split: /video = binary frames only, /events = text (meta + detection) only.
//
// Usage: node server.js <width> <height> [detectionPort] [videoFramesPerInferFrame] [inferWidth] [inferHeight]
const http = require("http");
const fs = require("fs");
const path = require("path");
const net = require("net");
const WebSocket = require("ws");

const streamW = parseInt(process.argv[2], 10);
const streamH = parseInt(process.argv[3], 10);
const detectionPort = parseInt(process.argv[4], 10) || 9998;
// If preview video FPS differs from inference FPS, pass the ratio here.
// Example: preview=10fps, infer=0.5fps => ratio=20 (one inference frame per 20 video frames).
const videoFramesPerInferFrame = (() => {
  const v = parseInt(process.argv[5], 10);
  return Number.isFinite(v) && v > 0 ? v : 1;
})();
// Inference resolution (bbox coordinates are in this space).
// Defaults to stream (preview) dimensions when omitted.
const inferW = (() => {
  const v = parseInt(process.argv[6], 10);
  return Number.isFinite(v) && v > 0 ? v : streamW;
})();
const inferH = (() => {
  const v = parseInt(process.argv[7], 10);
  return Number.isFinite(v) && v > 0 ? v : streamH;
})();

if (!Number.isFinite(streamW) || !Number.isFinite(streamH)) {
  console.error("Usage: node server.js <width> <height> [detectionPort] [videoFramesPerInferFrame] [inferWidth] [inferHeight]");
  process.exit(2);
}

const streamFmt = "rgb24";
const bytesPerPixel = 3;
const frameSizeBytes = streamW * streamH * bytesPerPixel;
const videoHeaderBytes = 4; // uint32le frame id
const packetSizeBytes = videoHeaderBytes + frameSizeBytes;

// Track video RX (stdin) throughput (bytes/sec) in a sliding window and expose it to the browser.
const RX_WINDOW_MS = 5000;
const rxTimesMs = [];
const rxBytes = [];
let rxHead = 0;
let rxSumBytes = 0;

function recordRx(nowMs, nbytes) {
  rxTimesMs.push(nowMs);
  rxBytes.push(nbytes);
  rxSumBytes += nbytes;
  while (rxHead < rxTimesMs.length && nowMs - rxTimesMs[rxHead] > RX_WINDOW_MS) {
    rxSumBytes -= rxBytes[rxHead];
    rxHead++;
  }
  if (rxHead > 1024) {
    rxTimesMs.splice(0, rxHead);
    rxBytes.splice(0, rxHead);
    rxHead = 0;
  }
}

function computeRxMbps(nowMs) {
  while (rxHead < rxTimesMs.length && nowMs - rxTimesMs[rxHead] > RX_WINDOW_MS) {
    rxSumBytes -= rxBytes[rxHead];
    rxHead++;
  }
  if (rxHead >= rxTimesMs.length) return 0;
  const durationMs = Math.min(RX_WINDOW_MS, nowMs - rxTimesMs[rxHead]);
  if (durationMs <= 0) return 0;
  // Decimal Mbps (megabit/sec)
  return (rxSumBytes / (durationMs / 1000)) * 8 / 1e6;
}

const server = http.createServer((req, res) => {
  const p = req.url === "/" ? "/index.html" : req.url;
  const file = path.join(__dirname, p);
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    const ext = path.extname(file);
    res.writeHead(200, {
      "Content-Type":
        ext === ".html" ? "text/html; charset=utf-8" :
          ext === ".js" ? "text/javascript; charset=utf-8" :
            "application/octet-stream",
    });
    res.end(data);
  });
});

// Separate WebSocket endpoints: /video = binary only, /events = text only
const wssVideo = new WebSocket.Server({ noServer: true });
const wssEvents = new WebSocket.Server({ noServer: true });

server.on("upgrade", (request, socket, head) => {
  const pathname = request.url ? request.url.split("?")[0] : "";
  if (pathname === "/video" || pathname === "/video/") {
    wssVideo.handleUpgrade(request, socket, head, (ws) => {
      wssVideo.emit("connection", ws, request);
    });
  } else if (pathname === "/events" || pathname === "/events/") {
    wssEvents.handleUpgrade(request, socket, head, (ws) => {
      wssEvents.emit("connection", ws, request);
    });
  } else {
    socket.destroy();
  }
});

const maxBufferedBytes = () => Math.max(packetSizeBytes * 8, 32 * 1024 * 1024);
let videoFrameCounter = 0;

function broadcastVideo(buf) {
  const frameId = videoFrameCounter++;

  // Frame id is prefixed to the binary payload so the browser can sync overlay to the *displayed*
  // frame even when it adds an artificial delay buffer.
  const packet = Buffer.allocUnsafe(packetSizeBytes);
  packet.writeUInt32LE(frameId >>> 0, 0);
  buf.copy(packet, videoHeaderBytes);

  // Also publish it on /events for debugging/HUD (not used for sync logic).
  if (wssEvents.clients.size > 0) {
    broadcastEvents(JSON.stringify({ type: "vframe", frame: frameId }));
  }
  for (const ws of wssVideo.clients) {
    if (ws.readyState !== WebSocket.OPEN) continue;
    if (ws.bufferedAmount > maxBufferedBytes()) continue;
    ws.send(packet);
  }
}

function broadcastEvents(text) {
  for (const ws of wssEvents.clients) {
    if (ws.readyState !== WebSocket.OPEN) continue;
    ws.send(text);
  }
}

wssEvents.on("connection", (ws) => {
  ws.send(
    JSON.stringify({
      type: "meta",
      w: streamW,
      h: streamH,
      fmt: streamFmt,
      videoFramesPerInferFrame,
      videoHeaderBytes,
      inferW,
      inferH,
    })
  );
});

// Periodically publish server-side RX (stdin) throughput to the browser.
setInterval(() => {
  if (wssEvents.clients.size === 0) return;
  const mbps = computeRxMbps(Date.now());
  broadcastEvents(JSON.stringify({ type: "rx", mbps }));
}, 250);

// Stdin: chunk list + total length (avoid Buffer.concat every chunk)
const chunks = [];
let totalLen = 0;

function consumeFront(n) {
  let left = n;
  while (left > 0 && chunks.length > 0) {
    const c = chunks[0];
    if (c.length <= left) {
      left -= c.length;
      totalLen -= c.length;
      chunks.shift();
    } else {
      chunks[0] = c.subarray(left);
      totalLen -= left;
      left = 0;
    }
  }
}

process.stdin.resume();
process.stdin.on("data", (chunk) => {
  // IMPORTANT: stdin must stay as raw bytes. Do not call setEncoding() for this stream.
  // If chunk becomes a string, the raw RGB24 byte stream is already corrupted.
  if (!Buffer.isBuffer(chunk)) {
    console.error(
      "stdin chunk is not a Buffer. Make sure stdin is binary (do not use setEncoding)."
    );
    return;
  }
  const buf = chunk;
  if (buf.length === 0) return;
  recordRx(Date.now(), buf.length);
  chunks.push(buf);
  totalLen += buf.length;

  while (totalLen >= frameSizeBytes) {
    const frame = Buffer.alloc(frameSizeBytes);
    let off = 0;
    let toCopy = frameSizeBytes;
    for (let i = 0; i < chunks.length && toCopy > 0; i++) {
      const take = Math.min(toCopy, chunks[i].length);
      chunks[i].copy(frame, off, 0, take);
      off += take;
      toCopy -= take;
    }
    consumeFront(frameSizeBytes);
    broadcastVideo(frame);
  }
});
process.stdin.on("end", () => {
  console.error("stdin ended");
  process.exit(0);
});

// TCP: detection stream (NDJSON) -> /events only
const detectionServer = net.createServer((socket) => {
  let buf = "";
  socket.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    const lines = buf.split("\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        JSON.parse(trimmed);
        broadcastEvents(trimmed);
      } catch (_) {
        // ignore malformed lines
      }
    }
  });
  socket.on("end", () => { });
});
detectionServer.listen(detectionPort, () => {
  console.error(`Detection stream listening on TCP ${detectionPort}`);
});

server.listen(8080, () => {
  console.error(
    `http://localhost:8080  WS /video (binary) + /events (text)  stream=${streamW}x${streamH}  infer=${inferW}x${inferH}  detection TCP=${detectionPort}`
  );
});
