/*
 * Demo WASM: read raw RGB24 frames from stdin, run YOLO inference,
 * output one NDJSON line per frame to stdout: {"frame":N,"detections":[...]}
 * No drawing, no video encoding. Server draws overlay from this stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wasi_nn_backend.h"
#include "wasi_nn_types.h"
#include "kafu.h"

#define YOLO_INPUT_SIZE 640
#define YOLO_NUM_CLASSES 80
#define YOLO_NUM_FEATURES 84
#define YOLO_PAD_VALUE 114
#define YOLO_BBOX_COORDS 4
#define YOLO_RGB_CHANNELS 3

#define DEFAULT_CONF_THRESHOLD 0.25f
#define DEFAULT_IOU_THRESHOLD 0.45f
#define DEFAULT_MAX_DETECTIONS 100

// Keep this buffer modest to reduce WASM linear memory size.
// NOTE: For YOLO11n (8400 detections * 84 features * fp32), output is ~2.8 MiB.
#define OUTPUT_BUFFER_SIZE (8 * 1024 * 1024)

#define YOLO_INPUT_TENSOR_ELEMENTS (1 * YOLO_RGB_CHANNELS * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE)
static float g_input_tensor_buf[YOLO_INPUT_TENSOR_ELEMENTS];
static uint8_t g_output_buffer[OUTPUT_BUFFER_SIZE];

typedef struct {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
} Detection;

static void die(const char *msg) {
  perror(msg);
  exit(1);
}

static double timespec_diff_sec(const struct timespec *start, const struct timespec *end) {
  return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1e9;
}

/* -------------------------------------------------------------------------- */
/* Model loading (cloud)                                                      */
/* -------------------------------------------------------------------------- */
KAFU_DEST(load_graph_from_onnx_file, "cloud")
KAFU_EXPORT(load_graph_from_onnx_file)
int load_graph_from_onnx_file(const char *filename, graph *out_graph) {
  if (!filename || !out_graph)
    return -1;
  fprintf(stderr, "Loading model from %s\n", filename);
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return -1;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return -1;
  }
  long size_long = ftell(fp);
  if (size_long <= 0 || (unsigned long)size_long > UINT32_MAX) {
    fclose(fp);
    return -1;
  }
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return -1;
  }
  uint32_t model_size = (uint32_t)size_long;
  uint8_t *model_buf = (uint8_t *)malloc(model_size);
  if (!model_buf) {
    fclose(fp);
    return -1;
  }
  if (fread(model_buf, 1, model_size, fp) != model_size) {
    fclose(fp);
    free(model_buf);
    return -1;
  }
  fclose(fp);

  graph_builder model = {.buf = model_buf, .size = model_size};
  wasi_nn_error err = wasi_nn_load(&model, 1, onnx, gpu, out_graph);
  free(model_buf);
  if (err == WASI_NN_ERROR_NAME(success)) {
    fprintf(stderr, "Model loaded (bytes=%u)\n", model_size);
  } else {
    fprintf(stderr, "Model load failed (wasi_nn_error=%d)\n", (int)err);
  }
  return (err != WASI_NN_ERROR_NAME(success)) ? (int)err : 0;
}

KAFU_DEST(init_execution_context, "cloud")
KAFU_EXPORT(init_execution_context)
int init_execution_context(graph g, graph_execution_context *out_exec_ctx) {
  if (!out_exec_ctx)
    return -1;
  fprintf(stderr, "Initializing execution context\n");
  wasi_nn_error err = wasi_nn_init_execution_context(g, out_exec_ctx);
  if (err == WASI_NN_ERROR_NAME(success)) {
    fprintf(stderr, "execution context initialized\n");
  } else {
    fprintf(stderr, "init execution context failed (wasi_nn_error=%d)\n", (int)err);
  }
  return (err != WASI_NN_ERROR_NAME(success)) ? (int)err : 0;
}

/* -------------------------------------------------------------------------- */
/* Preprocessing                                                              */
/* -------------------------------------------------------------------------- */
static uint8_t bilinear_interpolate(uint8_t *rgb, int W, int H, float src_x, float src_y,
                                    int channel) {
  int x0 = (int)src_x;
  int y0 = (int)src_y;
  int x1 = (x0 + 1 < W) ? x0 + 1 : x0;
  int y1 = (y0 + 1 < H) ? y0 + 1 : y0;
  float fx = src_x - x0;
  float fy = src_y - y0;
  x0 = (x0 < 0) ? 0 : ((x0 >= W) ? W - 1 : x0);
  y0 = (y0 < 0) ? 0 : ((y0 >= H) ? H - 1 : y0);
  x1 = (x1 < 0) ? 0 : ((x1 >= W) ? W - 1 : x1);
  y1 = (y1 < 0) ? 0 : ((y1 >= H) ? H - 1 : y1);
  float p00 = rgb[(y0 * W + x0) * YOLO_RGB_CHANNELS + channel];
  float p01 = rgb[(y0 * W + x1) * YOLO_RGB_CHANNELS + channel];
  float p10 = rgb[(y1 * W + x0) * YOLO_RGB_CHANNELS + channel];
  float p11 = rgb[(y1 * W + x1) * YOLO_RGB_CHANNELS + channel];
  float p0 = p00 * (1.0f - fx) + p01 * fx;
  float p1 = p10 * (1.0f - fx) + p11 * fx;
  return (uint8_t)((p0 * (1.0f - fy) + p1 * fy) + 0.5f);
}

static void resize_image_bilinear(uint8_t *src_rgb, int src_w, int src_h, uint8_t *dst_rgb,
                                  int dst_w, int dst_h) {
  float scale_x = (dst_w > 1) ? (float)(src_w - 1) / (dst_w - 1) : 0.0f;
  float scale_y = (dst_h > 1) ? (float)(src_h - 1) / (dst_h - 1) : 0.0f;
  for (int y = 0; y < dst_h; y++) {
    for (int x = 0; x < dst_w; x++) {
      float src_x = x * scale_x;
      float src_y = y * scale_y;
      for (int c = 0; c < YOLO_RGB_CHANNELS; c++)
        dst_rgb[(y * dst_w + x) * YOLO_RGB_CHANNELS + c] =
            bilinear_interpolate(src_rgb, src_w, src_h, src_x, src_y, c);
    }
  }
}

static void convert_to_chw_normalized(uint8_t *rgb_hwc, float *tensor_chw, int size) {
  for (int c = 0; c < YOLO_RGB_CHANNELS; c++)
    for (int h = 0; h < size; h++)
      for (int w = 0; w < size; w++) {
        int hwc_idx = (h * size + w) * YOLO_RGB_CHANNELS + c;
        tensor_chw[c * size * size + h * size + w] = rgb_hwc[hwc_idx] / 255.0f;
      }
}

static void preprocess_image(uint8_t *rgb, int width, int height, float *output_tensor) {
  float scale = fminf((float)YOLO_INPUT_SIZE / width, (float)YOLO_INPUT_SIZE / height);
  int new_w = (int)(width * scale);
  int new_h = (int)(height * scale);
  size_t padded_size = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * YOLO_RGB_CHANNELS;
  uint8_t *padded_rgb = (uint8_t *)malloc(padded_size);
  if (!padded_rgb)
    die("malloc padded_rgb");
  memset(padded_rgb, YOLO_PAD_VALUE, padded_size);
  resize_image_bilinear(rgb, width, height, padded_rgb, new_w, new_h);
  convert_to_chw_normalized(padded_rgb, output_tensor, YOLO_INPUT_SIZE);
  free(padded_rgb);
}

/* -------------------------------------------------------------------------- */
/* Post-processing (NMS, IoU)                                                 */
/* -------------------------------------------------------------------------- */
static float calculate_iou(const Detection *det1, const Detection *det2) {
  float inter_x1 = fmaxf(det1->x1, det2->x1);
  float inter_y1 = fmaxf(det1->y1, det2->y1);
  float inter_x2 = fminf(det1->x2, det2->x2);
  float inter_y2 = fminf(det1->y2, det2->y2);
  if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1)
    return 0.0f;
  float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  float area1 = (det1->x2 - det1->x1) * (det1->y2 - det1->y1);
  float area2 = (det2->x2 - det2->x1) * (det2->y2 - det2->y1);
  float union_area = area1 + area2 - inter_area;
  return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

static void sort_detections_by_confidence(Detection *dets, int num_dets) {
  for (int i = 0; i < num_dets - 1; i++)
    for (int j = i + 1; j < num_dets; j++)
      if (dets[j].confidence > dets[i].confidence) {
        Detection temp = dets[i];
        dets[i] = dets[j];
        dets[j] = temp;
      }
}

static int apply_nms(Detection *dets, int num_dets, float iou_threshold, Detection *output,
                     int max_output) {
  if (num_dets == 0)
    return 0;
  sort_detections_by_confidence(dets, num_dets);
  int *suppressed = (int *)calloc(num_dets, sizeof(int));
  if (!suppressed)
    die("malloc suppressed");
  int output_count = 0;
  for (int i = 0; i < num_dets && output_count < max_output; i++) {
    if (suppressed[i])
      continue;
    output[output_count++] = dets[i];
    for (int j = i + 1; j < num_dets; j++) {
      if (suppressed[j] || dets[i].class_id != dets[j].class_id)
        continue;
      if (calculate_iou(&dets[i], &dets[j]) > iou_threshold)
        suppressed[j] = 1;
    }
  }
  free(suppressed);
  return output_count;
}

static void clip_detections(Detection *dets, int num_dets, int width, int height) {
  for (int i = 0; i < num_dets; i++) {
    dets[i].x1 = fmaxf(0.0f, fminf(dets[i].x1, (float)(width - 1)));
    dets[i].y1 = fmaxf(0.0f, fminf(dets[i].y1, (float)(height - 1)));
    dets[i].x2 = fmaxf(0.0f, fminf(dets[i].x2, (float)(width - 1)));
    dets[i].y2 = fmaxf(0.0f, fminf(dets[i].y2, (float)(height - 1)));
  }
}

static int postprocess(float *output, uint32_t output_size_bytes, int width, int height,
                       float scale, float conf_threshold, float iou_threshold, Detection *dets,
                       int max_dets) {
  int total_elements = (int)(output_size_bytes / sizeof(float));
  int num_detections = total_elements / YOLO_NUM_FEATURES;
  if (num_detections <= 0)
    return 0;

  // NOTE: The model output is typically laid out as [features][detections] (feature-major).
  // Avoid transposing into a large temporary buffer to reduce memory footprint and dirty pages.
  Detection *candidates = (Detection *)malloc((size_t)num_detections * sizeof(Detection));
  if (!candidates)
    die("malloc candidates");
  int valid_count = 0;
  for (int i = 0; i < num_detections; i++) {
    float max_score = 0.0f;
    int class_id = 0;
    for (int j = 0; j < YOLO_NUM_CLASSES; j++) {
      float score = output[(YOLO_BBOX_COORDS + j) * num_detections + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }
    if (max_score > conf_threshold) {
      float cx = output[0 * num_detections + i];
      float cy = output[1 * num_detections + i];
      float w = output[2 * num_detections + i];
      float h = output[3 * num_detections + i];
      candidates[valid_count].x1 = (cx - w / 2.0f) / scale;
      candidates[valid_count].y1 = (cy - h / 2.0f) / scale;
      candidates[valid_count].x2 = (cx + w / 2.0f) / scale;
      candidates[valid_count].y2 = (cy + h / 2.0f) / scale;
      candidates[valid_count].confidence = max_score;
      candidates[valid_count].class_id = class_id;
      valid_count++;
    }
  }
  int num_dets = apply_nms(candidates, valid_count, iou_threshold, dets, max_dets);
  clip_detections(dets, num_dets, width, height);
  free(candidates);
  return num_dets;
}

/* -------------------------------------------------------------------------- */
/* KAFU: edge reads frame, cloud runs inference and outputs JSON to stdout   */
/* -------------------------------------------------------------------------- */
KAFU_DEST(read_frame_from_stdin, "edge")
KAFU_EXPORT(read_frame_from_stdin)
size_t read_frame_from_stdin(uint8_t *rgb, size_t frame_bytes) {
  // NOTE:
  // Reading from a pipe/FIFO can legitimately return short reads even in blocking mode.
  // We must accumulate until a full frame is received, or EOF/error occurs.
  fprintf(stderr, "Reading a frame of %zu bytes\n", frame_bytes);

  size_t total = 0;
  while (total < frame_bytes) {
    size_t n = fread(rgb + total, 1, frame_bytes - total, stdin);
    if (n == 0) {
      if (feof(stdin)) {
        fprintf(stderr, "read_frame_from_stdin: EOF total=%zu expected=%zu\n", total, frame_bytes);
        return total;
      }
      if (ferror(stdin)) {
        die("fread stdin");
      }
      // Should not happen, but avoid infinite loop.
      fprintf(stderr, "read_frame_from_stdin: fread returned 0 without eof/error\n");
      return total;
    }
    total += n;
  }
  return total;
}

KAFU_DEST(run_yolo_inference, "cloud")
KAFU_EXPORT(run_yolo_inference)
int run_yolo_inference(uint8_t *rgb, int width, int height, graph_execution_context exec_ctx,
                       float conf_threshold, float iou_threshold, Detection *dets, int max_dets) {
  fprintf(stderr, "Running YOLO inference: w=%d h=%d conf=%.3f iou=%.3f\n", width, height,
          conf_threshold, iou_threshold);
  const size_t tensor_size = (size_t)YOLO_INPUT_TENSOR_ELEMENTS * sizeof(float);
  float scale = fminf((float)YOLO_INPUT_SIZE / width, (float)YOLO_INPUT_SIZE / height);
  preprocess_image(rgb, width, height, g_input_tensor_buf);

  uint32_t dim_data[] = {1, YOLO_RGB_CHANNELS, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE};
  tensor_dimensions dimensions = {.buf = (uint32_t *)&dim_data, .size = 4};
  tensor_data data = {.buf = (uint8_t *)g_input_tensor_buf, .size = (uint32_t)tensor_size};
  tensor input_tensor = {
      .dimensions = dimensions,
      .type = WASI_NN_TYPE_NAME(fp32),
      .data = data,
  };

  wasi_nn_error err = wasi_nn_set_input(exec_ctx, 0, &input_tensor);
  if (err != WASI_NN_ERROR_NAME(success))
    die("set input");

  err = wasi_nn_compute(exec_ctx);
  if (err != WASI_NN_ERROR_NAME(success))
    die("compute");

  uint32_t output_buffer_max_size = OUTPUT_BUFFER_SIZE;
  uint8_t *output_buffer = g_output_buffer;
  uint32_t output_buffer_size;
  err = wasi_nn_get_output(exec_ctx, 0, output_buffer, output_buffer_max_size, &output_buffer_size);
  if (err != WASI_NN_ERROR_NAME(success))
    die("get output");

  float *output_f32 = (float *)output_buffer;
  int num_dets = postprocess(output_f32, output_buffer_size, width, height, scale, conf_threshold,
                             iou_threshold, dets, max_dets);
  return num_dets;
}

/* Output one NDJSON line to stdout: {"frame":N,"detections":[{"x1",...},...]} */
KAFU_DEST(output_detections_to_stdout, "cloud")
KAFU_EXPORT(output_detections_to_stdout)
void output_detections_to_stdout(size_t frame_id, const Detection *dets, int num_dets) {
  if (frame_id == 0 || (frame_id % 120) == 0) {
    fprintf(stderr, "Outputing detections: frame=%zu dets=%d\n", frame_id, num_dets);
  }
  fprintf(stdout, "{\"frame\":%zu,\"detections\":[", frame_id);
  for (int i = 0; i < num_dets; i++) {
    if (i > 0)
      fputc(',', stdout);
    fprintf(stdout, "{\"x1\":%d,\"y1\":%d,\"x2\":%d,\"y2\":%d,\"class\":%d}", (int)dets[i].x1,
            (int)dets[i].y1, (int)dets[i].x2, (int)dets[i].y2, dets[i].class_id);
  }
  fprintf(stdout, "]}\n");
  fflush(stdout);
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */
#define FRAME_LOG_INTERVAL 100
#define FPS_REPORT_INTERVAL_SEC 1.0
#define FPS_WINDOW_SEC 5.0
#define FPS_RING_CAPACITY 4096

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if (argc < 4) {
    fprintf(stderr, "Usage: %s <width> <height> <model_path> [conf] [iou]\n", argv[0]);
    return 2;
  }
  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  const char *model_path = argv[3];
  float conf_threshold = (argc >= 5) ? (float)atof(argv[4]) : DEFAULT_CONF_THRESHOLD;
  float iou_threshold = (argc >= 6) ? (float)atof(argv[5]) : DEFAULT_IOU_THRESHOLD;

  if (width <= 0 || height <= 0) {
    fprintf(stderr, "Invalid width or height\n");
    return 1;
  }

  graph g;
  if (load_graph_from_onnx_file(model_path, &g) != 0) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
  }
  graph_execution_context exec_ctx;
  if (init_execution_context(g, &exec_ctx) != 0) {
    fprintf(stderr, "Failed to init execution context\n");
    return 1;
  }

  const size_t frame_bytes = (size_t)width * (size_t)height * YOLO_RGB_CHANNELS;
  uint8_t *rgb = (uint8_t *)malloc(frame_bytes);
  if (!rgb)
    die("malloc rgb");
  Detection *dets = (Detection *)malloc(DEFAULT_MAX_DETECTIONS * sizeof(Detection));
  if (!dets)
    die("malloc dets");

  size_t frame_id = 0;
  int fps_started = 0;
  struct timespec fps_last_report_ts;
  size_t fps_total_frames = 0;
  struct timespec fps_ring[FPS_RING_CAPACITY];
  size_t fps_ring_head = 0;
  size_t fps_ring_count = 0;

  while (1) {
    size_t got = read_frame_from_stdin(rgb, frame_bytes);
    if (got == 0 || got != frame_bytes)
      break;

    if (!fps_started) {
      clock_gettime(CLOCK_MONOTONIC, &fps_last_report_ts);
      fps_started = 1;
    }

    int num_dets = run_yolo_inference(rgb, width, height, exec_ctx, conf_threshold, iou_threshold,
                                      dets, DEFAULT_MAX_DETECTIONS);
    output_detections_to_stdout(frame_id, dets, num_dets);
    frame_id++;
    fps_total_frames++;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    size_t write_idx = (fps_ring_head + fps_ring_count) % FPS_RING_CAPACITY;
    fps_ring[write_idx] = now;
    if (fps_ring_count == FPS_RING_CAPACITY)
      fps_ring_head = (fps_ring_head + 1) % FPS_RING_CAPACITY;
    else
      fps_ring_count++;
    while (fps_ring_count > 0 &&
           timespec_diff_sec(&fps_ring[fps_ring_head], &now) > FPS_WINDOW_SEC) {
      fps_ring_head = (fps_ring_head + 1) % FPS_RING_CAPACITY;
      fps_ring_count--;
    }
    double since_report_sec = timespec_diff_sec(&fps_last_report_ts, &now);
    if (since_report_sec >= FPS_REPORT_INTERVAL_SEC) {
      double window_span_sec =
          (fps_ring_count > 0)
              ? fmin(FPS_WINDOW_SEC, timespec_diff_sec(&fps_ring[fps_ring_head], &now))
              : 0.0;
      double fps_5s = (window_span_sec > 0.0) ? ((double)fps_ring_count / window_span_sec) : 0.0;
      fprintf(stderr, "FPS: %.2f Total frames=%zu\n", fps_5s, fps_total_frames);
      fps_last_report_ts = now;
    }
    if ((frame_id % FRAME_LOG_INTERVAL) == 0)
      fprintf(stderr, "Processed frame %zu\n", frame_id);
  }

  free(rgb);
  free(dets);
  return 0;
}
