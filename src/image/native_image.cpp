// SPDX-License-Identifier: Apache-2.0
#if defined(__APPLE__) && defined(GARNET_WITH_METAL)
#include "native_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <jpeglib.h>
#include <memory>
#include <png.h>
#include <setjmp.h>
#include <stdexcept>
namespace Garnet::Image {
namespace {
using Clock = std::chrono::steady_clock;
struct RGB {
  int w, h;
  std::vector<unsigned char> data;
};
void Dimensions(uint64_t w, uint64_t h) {
  if (!w || !h || w > 65535 || h > 65535 || w * h > 67108864)
    throw std::runtime_error("Native image dimensions exceed admission");
}
int Orientation(const unsigned char *p, size_t n) {
  if (n < 8)
    return 1;
  const bool le = p[0] == 'I' && p[1] == 'I';
  if (!le && !(p[0] == 'M' && p[1] == 'M'))
    return 1;
  auto u16 = [&](size_t o) -> uint16_t {
    return le ? uint16_t(p[o] | p[o + 1] << 8) : uint16_t(p[o] << 8 | p[o + 1]);
  };
  auto u32 = [&](size_t o) -> uint32_t {
    return le ? uint32_t(p[o]) | uint32_t(p[o + 1]) << 8 |
                    uint32_t(p[o + 2]) << 16 | uint32_t(p[o + 3]) << 24
              : uint32_t(p[o]) << 24 | uint32_t(p[o + 1]) << 16 |
                    uint32_t(p[o + 2]) << 8 | uint32_t(p[o + 3]);
  };
  if (u16(2) != 42)
    return 1;
  size_t off = u32(4);
  if (off > n - 2)
    return 1;
  size_t count = u16(off);
  off += 2;
  if (count > (n - off) / 12)
    return 1;
  for (size_t i = 0; i < count; ++i) {
    size_t a = off + i * 12;
    if (u16(a) == 274 && u16(a + 2) == 3 && u32(a + 4) == 1) {
      int v = u16(a + 8);
      return v >= 1 && v <= 8 ? v : 1;
    }
  }
  return 1;
}
RGB Orient(RGB in, int o) {
  if (o < 2 || o > 8)
    return in;
  const bool swap = o >= 5;
  RGB out{swap ? in.h : in.w, swap ? in.w : in.h, {}};
  out.data.resize(in.data.size());
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      int dx = x, dy = y;
      switch (o) {
      case 2:
        dx = in.w - 1 - x;
        break;
      case 3:
        dx = in.w - 1 - x;
        dy = in.h - 1 - y;
        break;
      case 4:
        dy = in.h - 1 - y;
        break;
      case 5:
        dx = y;
        dy = x;
        break;
      case 6:
        dx = in.h - 1 - y;
        dy = x;
        break;
      case 7:
        dx = in.h - 1 - y;
        dy = in.w - 1 - x;
        break;
      case 8:
        dx = y;
        dy = in.w - 1 - x;
        break;
      }
      std::memcpy(out.data.data() + (size_t(dy) * out.w + dx) * 3,
                  in.data.data() + (size_t(y) * in.w + x) * 3, 3);
    }
  return out;
}
struct JpegError {
  jpeg_error_mgr base;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
  unsigned char *rgb = nullptr;
  unsigned char *row = nullptr;
};
void JpegFail(j_common_ptr c) {
  auto *e = reinterpret_cast<JpegError *>(c->err);
  (*c->err->format_message)(c, e->message);
  longjmp(e->jump, 1);
}
void JpegMessage(j_common_ptr c, int level) {
  if (level < 0)
    JpegFail(c);
}
RGB DecodeJpeg(const std::vector<unsigned char> &bytes) {
  struct State {
    jpeg_decompress_struct decoder{};
    JpegError error{};
  };
  auto state = std::make_unique<State>();
  auto &c = state->decoder;
  auto &e = state->error;
  c.err = jpeg_std_error(&e.base);
  e.base.error_exit = JpegFail;
  e.base.emit_message = JpegMessage;
  if (setjmp(e.jump)) {
    std::free(e.rgb);
    std::free(e.row);
    jpeg_destroy_decompress(&c);
    throw std::runtime_error(std::string("Native JPEG decode: ") + e.message);
  }
  jpeg_create_decompress(&c);
  jpeg_mem_src(&c, bytes.data(), bytes.size());
  jpeg_save_markers(&c, JPEG_APP0 + 1, 65535);
  jpeg_read_header(&c, TRUE);
  if (!c.image_width || !c.image_height ||
      uint64_t(c.image_width) * c.image_height > 67108864 ||
      c.image_width > 65535 || c.image_height > 65535) {
    jpeg_destroy_decompress(&c);
    throw std::runtime_error("Native JPEG dimensions exceed admission");
  }
  int orientation = 1;
  for (auto *m = c.marker_list; m; m = m->next)
    if (m->marker == JPEG_APP0 + 1 && m->data_length >= 6 &&
        !std::memcmp(m->data, "Exif\0\0", 6))
      orientation = Orientation(m->data + 6, m->data_length - 6);
  bool cmyk = c.jpeg_color_space == JCS_CMYK || c.jpeg_color_space == JCS_YCCK;
  c.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
  c.dct_method = JDCT_ISLOW;
  c.do_fancy_upsampling = TRUE;
  jpeg_start_decompress(&c);
  int w = c.output_width, h = c.output_height;
  size_t size = size_t(w) * h * 3;
  e.rgb = static_cast<unsigned char *>(std::malloc(size));
  e.row = static_cast<unsigned char *>(
      std::malloc(size_t(w) * c.output_components));
  if (!e.rgb || !e.row) {
    std::free(e.rgb);
    std::free(e.row);
    jpeg_destroy_decompress(&c);
    throw std::bad_alloc();
  }
  while (c.output_scanline < c.output_height) {
    size_t y = c.output_scanline;
    JSAMPROW row = e.row;
    jpeg_read_scanlines(&c, &row, 1);
    auto *dst = e.rgb + y * w * 3;
    if (!cmyk)
      std::memcpy(dst, e.row, size_t(w) * 3);
    else
      for (int x = 0; x < w; ++x)
        for (int k = 0; k < 3; ++k)
          dst[x * 3 + k] = static_cast<unsigned char>(
              (int(e.row[x * 4 + k]) * e.row[x * 4 + 3] + 127) / 255);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  std::free(e.row);
  std::unique_ptr<unsigned char, decltype(&std::free)> owned(e.rgb, &std::free);
  return Orient(
      RGB{w, h, std::vector<unsigned char>(owned.get(), owned.get() + size)},
      orientation);
}
struct PngReader {
  const unsigned char *data;
  size_t size, offset;
  unsigned char *rgb = nullptr;
  png_bytep *rows = nullptr;
  char message[256]{};
};
void PngRead(png_structp p, png_bytep dst, png_size_t n) {
  auto *s = static_cast<PngReader *>(png_get_io_ptr(p));
  if (n > s->size - s->offset)
    png_error(p, "truncated PNG");
  std::memcpy(dst, s->data + s->offset, n);
  s->offset += n;
}
void PngFail(png_structp p, png_const_charp msg) {
  auto *s = static_cast<PngReader *>(png_get_error_ptr(p));
  std::snprintf(s->message, sizeof(s->message), "%s", msg);
  png_longjmp(p, 1);
}
void PngWarn(png_structp, png_const_charp) {}
RGB DecodePng(const std::vector<unsigned char> &bytes) {
  auto state = std::make_unique<PngReader>();
  state->data = bytes.data();
  state->size = bytes.size();
  state->offset = 0;
  auto &s = *state;
  png_structp p =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, &s, PngFail, PngWarn);
  if (!p)
    throw std::bad_alloc();
  png_infop info = png_create_info_struct(p);
  if (!info) {
    png_destroy_read_struct(&p, nullptr, nullptr);
    throw std::bad_alloc();
  }
  if (setjmp(png_jmpbuf(p))) {
    std::free(s.rgb);
    std::free(s.rows);
    png_destroy_read_struct(&p, &info, nullptr);
    throw std::runtime_error(std::string("Native PNG decode: ") + s.message);
  }
  png_set_read_fn(p, &s, PngRead);
  png_read_info(p, info);
  auto w = png_get_image_width(p, info), h = png_get_image_height(p, info);
  if (!w || !h || w > 65535 || h > 65535 || uint64_t(w) * h > 67108864)
    png_error(p, "dimensions exceed admission");
  int orientation = 1;
  int type = png_get_color_type(p, info), depth = png_get_bit_depth(p, info);
  bool gray16 = type == PNG_COLOR_TYPE_GRAY && depth == 16;
  if (depth == 16 && !gray16)
    png_set_strip_16(p);
  if (type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(p);
  if (type == PNG_COLOR_TYPE_GRAY && depth < 8)
    png_set_expand_gray_1_2_4_to_8(p);
  if (png_get_valid(p, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(p);
  if (type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(p);
  png_set_strip_alpha(p);
  png_set_interlace_handling(p);
  png_read_update_info(p, info);
  if (png_get_rowbytes(p, info) != size_t(w) * (gray16 ? 6 : 3) ||
      png_get_channels(p, info) != 3)
    png_error(p, "unsupported RGB layout");
  size_t rowBytes = size_t(w) * (gray16 ? 6 : 3), size = rowBytes * h;
  s.rgb = static_cast<unsigned char *>(std::malloc(size));
  s.rows = static_cast<png_bytep *>(std::malloc(size_t(h) * sizeof(png_bytep)));
  if (!s.rgb || !s.rows)
    png_error(p, "allocation failed");
  for (size_t y = 0; y < h; ++y)
    s.rows[y] = s.rgb + y * rowBytes;
  png_read_image(p, s.rows);
  png_read_end(p, info);
#ifdef PNG_eXIf_SUPPORTED
  png_uint_32 length = 0;
  png_bytep exif = nullptr;
  if (png_get_eXIf_1(p, info, &length, &exif))
    orientation = Orientation(exif, length);
#endif
  png_destroy_read_struct(&p, &info, nullptr);
  std::free(s.rows);
  std::unique_ptr<unsigned char, decltype(&std::free)> owned(s.rgb, &std::free);
  if (gray16) {
    RGB result{int(w), int(h), {}};
    result.data.resize(size_t(w) * h * 3);
    for (size_t i = 0; i < result.data.size(); ++i)
      result.data[i] = static_cast<unsigned char>(std::min(
          255, int(owned.get()[i * 2]) * 256 + owned.get()[i * 2 + 1]));
    return Orient(std::move(result), orientation);
  }
  return Orient(
      RGB{int(w), int(h),
          std::vector<unsigned char>(owned.get(), owned.get() + size)},
      orientation);
}
RGB Decode(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("Cannot open native image");
  auto size = f.tellg();
  if (size <= 0 || size > 268435456)
    throw std::runtime_error("Invalid native image byte length");
  std::vector<unsigned char> bytes(size);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!f)
    throw std::runtime_error("Cannot read native image");
  if (bytes.size() >= 2 && bytes[0] == 255 && bytes[1] == 216)
    return DecodeJpeg(bytes);
  if (bytes.size() >= 8 && !png_sig_cmp(bytes.data(), 0, 8))
    return DecodePng(bytes);
  throw std::runtime_error("Native image input must be JPEG or PNG");
}
struct Coefficients {
  int start;
  std::vector<int32_t> values;
};
double Cubic(double x) {
  x = std::abs(x);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return (((x - 5) * x + 8) * x - 4) * -.5;
  return 0;
}
std::vector<Coefficients> Coeffs(int input, int output) {
  std::vector<Coefficients> result;
  result.reserve(output);
  double scale = double(input) / output, filter = std::max(1., scale),
         support = 2 * filter;
  for (int i = 0; i < output; ++i) {
    double center = (i + .5) * scale;
    int first = std::max(0, int(center - support + .5)),
        last = std::min(input, int(center + support + .5));
    std::vector<double> weights;
    double sum = 0;
    for (int x = first; x < last; ++x) {
      double v = Cubic((x - center + .5) / filter);
      weights.push_back(v);
      sum += v;
    }
    Coefficients c{first, {}};
    for (double v : weights) {
      double q = (v / sum) * 4194304.;
      c.values.push_back(int32_t(q < 0 ? q - .5 : q + .5));
    }
    result.push_back(std::move(c));
  }
  return result;
}
unsigned char Clip(int64_t v) {
  v >>= 22;
  return static_cast<unsigned char>(
      std::max<int64_t>(0, std::min<int64_t>(255, v)));
}
RGB Resize(const RGB &in, int width, int height) {
  if (in.w == width && in.h == height)
    return in;
  RGB horizontal{width, in.h, {}};
  const RGB *source = &in;
  if (width != in.w) {
    auto cx = Coeffs(in.w, width);
    horizontal.data.resize(size_t(width) * in.h * 3);
    for (int y = 0; y < in.h; ++y)
      for (int x = 0; x < width; ++x)
        for (int c = 0; c < 3; ++c) {
          int64_t sum = 2097152;
          const auto &k = cx[x];
          for (size_t j = 0; j < k.values.size(); ++j)
            sum += int64_t(in.data[(size_t(y) * in.w + k.start + j) * 3 + c]) *
                   k.values[j];
          horizontal.data[(size_t(y) * width + x) * 3 + c] = Clip(sum);
        }
    source = &horizontal;
  }
  if (height == in.h)
    return *source;
  auto cy = Coeffs(in.h, height);
  RGB out{width, height, {}};
  out.data.resize(size_t(width) * height * 3);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 2097152;
        const auto &k = cy[y];
        for (size_t j = 0; j < k.values.size(); ++j)
          sum +=
              int64_t(source->data[(size_t(k.start + j) * width + x) * 3 + c]) *
              k.values[j];
        out.data[(size_t(y) * width + x) * 3 + c] = Clip(sum);
      }
  return out;
}
int EvenRound(double v) {
  int lo = int(std::floor(v));
  double f = v - lo;
  return f > .5 || (f == .5 && (lo & 1)) ? lo + 1 : lo;
}
} // namespace
NativeImageResult PrepareNativeImage(const std::string &path,
                                     const NativeImageConfig &c) {
  if (c.patch < 1 || c.patch > 128 || c.merge < 1 || c.merge > 8 ||
      c.temporal < 1 || c.temporal > 8 || c.pixels < 4096 || c.pixels > 1048576)
    throw std::runtime_error("Invalid native image profile");
  for (int k = 0; k < 3; ++k)
    if (!std::isfinite(c.mean[k]) || !std::isfinite(c.stddev[k]) ||
        c.stddev[k] <= 0)
      throw std::runtime_error("Invalid native image normalization");
  auto start = Clock::now();
  auto image = Decode(path);
  auto decoded = Clock::now();
  int factor = c.patch * c.merge;
  int h = std::max(factor, EvenRound(double(image.h) / factor) * factor),
      w = std::max(factor, EvenRound(double(image.w) / factor) * factor);
  if (int64_t(h) * w > c.pixels) {
    double b = std::sqrt(double(image.h) * image.w / c.pixels);
    h = std::max(factor, int(std::floor(image.h / b / factor)) * factor);
    w = std::max(factor, int(std::floor(image.w / b / factor)) * factor);
  } else if (int64_t(h) * w < c.pixels) {
    double b = std::sqrt(double(c.pixels) / (double(image.h) * image.w));
    h = int(std::ceil(image.h * b / factor)) * factor;
    w = int(std::ceil(image.w * b / factor)) * factor;
  }
  Dimensions(w, h);
  auto resized = Resize(image, w, h);
  auto scaled = Clock::now();
  NativeImageResult out;
  out.gridH = h / c.patch;
  out.gridW = w / c.patch;
  out.features = 3 * c.temporal * c.patch * c.patch;
  uint64_t count = uint64_t(out.gridH) * out.gridW * out.features;
  if (count > 67108864)
    throw std::runtime_error("Native image tensor exceeds admission");
  out.patches.reserve(count);
  for (int by = 0; by < out.gridH / c.merge; ++by)
    for (int bx = 0; bx < out.gridW / c.merge; ++bx)
      for (int iy = 0; iy < c.merge; ++iy)
        for (int ix = 0; ix < c.merge; ++ix)
          for (int channel = 0; channel < 3; ++channel)
            for (int t = 0; t < c.temporal; ++t)
              for (int y = 0; y < c.patch; ++y)
                for (int x = 0; x < c.patch; ++x) {
                  int row = (by * c.merge + iy) * c.patch + y,
                      col = (bx * c.merge + ix) * c.patch + x;
                  float v =
                      float(
                          resized.data[(size_t(row) * w + col) * 3 + channel]) /
                      255.f;
                  v = v - c.mean[channel];
                  v = v / c.stddev[channel];
                  out.patches.push_back(v);
                }
  auto packed = Clock::now();
  out.decodeMs =
      std::chrono::duration<double, std::milli>(decoded - start).count();
  out.resizeMs =
      std::chrono::duration<double, std::milli>(scaled - decoded).count();
  out.packMs =
      std::chrono::duration<double, std::milli>(packed - scaled).count();
  return out;
}
} // namespace Garnet::Image

#else
#include "native_image.h"
#include <stdexcept>
namespace Garnet::Image {
NativeImageResult PrepareNativeImage(const std::string &, const NativeImageConfig &) {
  throw std::runtime_error("Native JPEG/PNG preparation requires an Apple Metal build");
}
}
#endif
