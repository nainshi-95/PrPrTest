#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct MotionVector {
    int x = 0;  // 1/16-pel units
    int y = 0;  // 1/16-pel units
    int64_t error = std::numeric_limits<int64_t>::max();
};

template <typename T>
class Array2D {
public:
    Array2D() = default;
    Array2D(int w, int h, const T& value = T()) { allocate(w, h, value); }

    void allocate(int w, int h, const T& value = T()) {
        width_ = w;
        height_ = h;
        data_.assign(static_cast<size_t>(w) * static_cast<size_t>(h), value);
    }

    int w() const { return width_; }
    int h() const { return height_; }

    T& get(int x, int y) { return data_[static_cast<size_t>(y) * width_ + x]; }
    const T& get(int x, int y) const { return data_[static_cast<size_t>(y) * width_ + x]; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<T> data_;
};

struct Image {
    int w = 0;
    int h = 0;
    std::vector<int> pix; // stored as int for safe math

    Image() = default;
    Image(int width, int height) : w(width), h(height), pix(static_cast<size_t>(width) * height, 0) {}

    int& at(int x, int y) { return pix[static_cast<size_t>(y) * w + x]; }
    int at(int x, int y) const { return pix[static_cast<size_t>(y) * w + x]; }
};

constexpr int BASELINE_BIT_DEPTH = 10;
constexpr int MOTION_VECTOR_FACTOR = 16;
constexpr int NTAPS = 6;
constexpr int HALF_NTAPS = (NTAPS - 1) / 2;

static const int16_t INTERP[16][NTAPS] = {
    { 0,  0, 64,  0,  0,  0 },
    { 1, -3, 63,  4, -2,  1 },
    { 1, -5, 62,  8, -3,  1 },
    { 1, -6, 60, 13, -4,  0 },
    { 1, -7, 57, 19, -5, -1 },
    { 1, -8, 54, 24, -6, -1 },
    { 1, -9, 50, 29, -6, -1 },
    { 1, -9, 46, 35, -7, -2 },
    { 1, -10, 42, 42, -10, 1 },
    { -2, -7, 35, 46, -9, 1 },
    { -1, -6, 29, 50, -9, 1 },
    { -1, -6, 24, 54, -8, 1 },
    { -1, -5, 19, 57, -7, 1 },
    { 0, -4, 13, 60, -6, 1 },
    { 1, -3,  8, 62, -5, 1 },
    { 1, -2,  4, 63, -3, 1 },
};

inline int clamp_int(int v, int lo, int hi) {
    return std::min(std::max(v, lo), hi);
}

Image pad_with_border(const Image& src, int margin) {
    Image out(src.w + 2 * margin, src.h + 2 * margin);
    for (int y = 0; y < out.h; ++y) {
        int sy = clamp_int(y - margin, 0, src.h - 1);
        for (int x = 0; x < out.w; ++x) {
            int sx = clamp_int(x - margin, 0, src.w - 1);
            out.at(x, y) = src.at(sx, sy);
        }
    }
    return out;
}

Image subsample_luma(const Image& input, int factor = 2) {
    if (input.w % factor != 0 || input.h % factor != 0) {
        throw std::runtime_error("Input size must be divisible by subsampling factor");
    }
    Image out(input.w / factor, input.h / factor);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            const int x0 = x * factor;
            const int y0 = y * factor;
            out.at(x, y) = (input.at(x0, y0) + input.at(x0 + 1, y0) +
                            input.at(x0, y0 + 1) + input.at(x0 + 1, y0 + 1) + 2) >> 2;
        }
    }
    return out;
}

int filtered_sample_6tap(const Image& buf, int base_x, int base_y, int dx, int dy, int max_value) {
    const int dxFrac = dx & 15;
    const int dyFrac = dy & 15;
    const int dxInt = dx >> 4;
    const int dyInt = dy >> 4;
    const auto* xFilter = INTERP[dxFrac];
    const auto* yFilter = INTERP[dyFrac];

    const int ox = base_x + dxInt;
    const int oy = base_y + dyInt;

    if (dxFrac == 0 && dyFrac == 0) {
        return buf.at(ox, oy);
    }
    if (dxFrac == 0) {
        int sum = 1 << 5;
        for (int k = 0; k < NTAPS; ++k) {
            sum += buf.at(ox, oy + k - HALF_NTAPS) * yFilter[k];
        }
        return clamp_int(sum >> 6, 0, max_value);
    }
    if (dyFrac == 0) {
        int sum = 1 << 5;
        for (int k = 0; k < NTAPS; ++k) {
            sum += buf.at(ox + k - HALF_NTAPS, oy) * xFilter[k];
        }
        return clamp_int(sum >> 6, 0, max_value);
    }

    int temp[NTAPS];
    for (int ky = 0; ky < NTAPS; ++ky) {
        int sum = 0;
        const int yy = oy + ky - HALF_NTAPS;
        for (int kx = 0; kx < NTAPS; ++kx) {
            sum += buf.at(ox + kx - HALF_NTAPS, yy) * xFilter[kx];
        }
        temp[ky] = sum;
    }
    int sum = 1 << 11;
    for (int k = 0; k < NTAPS; ++k) {
        sum += temp[k] * yFilter[k];
    }
    return clamp_int(sum >> 12, 0, max_value);
}

int64_t motion_error_luma(const Image& orig, const Image& buffer, int x, int y, int dx, int dy,
                          int bs, int64_t besterror, int max_value) {
    int64_t error = 0;
    for (int y1 = 0; y1 < bs; ++y1) {
        for (int x1 = 0; x1 < bs; ++x1) {
            const int pred = filtered_sample_6tap(buffer, x + x1, y + y1, dx, dy, max_value);
            const int diff = orig.at(x + x1, y + y1) - pred;
            error += static_cast<int64_t>(diff) * diff;
        }
        if (error > besterror) {
            return error;
        }
    }
    return error;
}

void motion_estimation_luma(Array2D<MotionVector>& mvs, const Image& orig, const Image& buffer,
                            int blockSize, int bitdepth,
                            const Array2D<MotionVector>* previous = nullptr,
                            int factor = 1, bool doubleRes = false) {
    int range = doubleRes ? 0 : 5;
    const int stepSize = blockSize;
    const int origWidth = orig.w;
    const int origHeight = orig.h;
    const int maxValue = (1 << bitdepth) - 1;

    const double offset = 5.0 / (1 << (2 * BASELINE_BIT_DEPTH - 16)) * (1 << (2 * bitdepth - 16));
    const double scale  = 50.0 / (1 << (2 * BASELINE_BIT_DEPTH - 16)) * (1 << (2 * bitdepth - 16));

    for (int blockY = 0; blockY + blockSize <= origHeight; blockY += stepSize) {
        for (int blockX = 0; blockX + blockSize <= origWidth; blockX += stepSize) {
            MotionVector best;

            if (previous == nullptr) {
                range = 8;
            } else {
                for (int py = -1; py <= 1; ++py) {
                    const int testy = blockY / (2 * blockSize) + py;
                    for (int px = -1; px <= 1; ++px) {
                        const int testx = blockX / (2 * blockSize) + px;
                        if (testx >= 0 && testx < previous->w() && testy >= 0 && testy < previous->h()) {
                            MotionVector old = previous->get(testx, testy);
                            const int64_t error = motion_error_luma(
                                orig, buffer, blockX, blockY,
                                old.x * factor, old.y * factor,
                                blockSize, best.error, maxValue);
                            if (error < best.error) {
                                best = {old.x * factor, old.y * factor, error};
                            }
                        }
                    }
                }
                const int64_t zero_error = motion_error_luma(orig, buffer, blockX, blockY, 0, 0, blockSize, best.error, maxValue);
                if (zero_error < best.error) {
                    best = {0, 0, zero_error};
                }
            }

            MotionVector prevBest = best;
            for (int y2 = prevBest.y / MOTION_VECTOR_FACTOR - range; y2 <= prevBest.y / MOTION_VECTOR_FACTOR + range; ++y2) {
                for (int x2 = prevBest.x / MOTION_VECTOR_FACTOR - range; x2 <= prevBest.x / MOTION_VECTOR_FACTOR + range; ++x2) {
                    const int64_t error = motion_error_luma(
                        orig, buffer, blockX, blockY,
                        x2 * MOTION_VECTOR_FACTOR, y2 * MOTION_VECTOR_FACTOR,
                        blockSize, best.error, maxValue);
                    if (error < best.error) {
                        best = {x2 * MOTION_VECTOR_FACTOR, y2 * MOTION_VECTOR_FACTOR, error};
                    }
                }
            }

            if (doubleRes) {
                prevBest = best;
                int doubleRange = 12;
                for (int y2 = prevBest.y - doubleRange; y2 <= prevBest.y + doubleRange; y2 += 4) {
                    for (int x2 = prevBest.x - doubleRange; x2 <= prevBest.x + doubleRange; x2 += 4) {
                        const int64_t error = motion_error_luma(orig, buffer, blockX, blockY, x2, y2, blockSize, best.error, maxValue);
                        if (error < best.error) {
                            best = {x2, y2, error};
                        }
                    }
                }
                prevBest = best;
                doubleRange = 3;
                for (int y2 = prevBest.y - doubleRange; y2 <= prevBest.y + doubleRange; ++y2) {
                    for (int x2 = prevBest.x - doubleRange; x2 <= prevBest.x + doubleRange; ++x2) {
                        const int64_t error = motion_error_luma(orig, buffer, blockX, blockY, x2, y2, blockSize, best.error, maxValue);
                        if (error < best.error) {
                            best = {x2, y2, error};
                        }
                    }
                }
            }

            if (blockY > 0) {
                const MotionVector above = mvs.get(blockX / stepSize, (blockY - stepSize) / stepSize);
                const int64_t error = motion_error_luma(orig, buffer, blockX, blockY, above.x, above.y, blockSize, best.error, maxValue);
                if (error < best.error) {
                    best = {above.x, above.y, error};
                }
            }
            if (blockX > 0) {
                const MotionVector left = mvs.get((blockX - stepSize) / stepSize, blockY / stepSize);
                const int64_t error = motion_error_luma(orig, buffer, blockX, blockY, left.x, left.y, blockSize, best.error, maxValue);
                if (error < best.error) {
                    best = {left.x, left.y, error};
                }
            }

            double avg = 0.0;
            for (int y1 = 0; y1 < blockSize; ++y1) {
                for (int x1 = 0; x1 < blockSize; ++x1) {
                    avg += orig.at(blockX + x1, blockY + y1);
                }
            }
            avg /= static_cast<double>(blockSize * blockSize);

            double variance = 0.0;
            for (int y1 = 0; y1 < blockSize; ++y1) {
                for (int x1 = 0; x1 < blockSize; ++x1) {
                    const double d = orig.at(blockX + x1, blockY + y1) - avg;
                    variance += d * d;
                }
            }
            best.error = static_cast<int64_t>(20.0 * ((best.error + offset) / (variance + offset)) +
                                              (best.error / static_cast<double>(blockSize * blockSize)) / scale);

            mvs.get(blockX / stepSize, blockY / stepSize) = best;
        }
    }
}

Array2D<MotionVector> motion_estimation(const Image& orgPic, const Image& buffer, int bitdepth) {
    if (orgPic.w != buffer.w || orgPic.h != buffer.h) {
        throw std::runtime_error("target and reference must have the same shape");
    }
    if (orgPic.w % 16 != 0 || orgPic.h % 16 != 0) {
        throw std::runtime_error("height and width must be multiples of 16");
    }

    const Image orgPad = pad_with_border(orgPic, 4);
    const Image bufPad = pad_with_border(buffer, 4);

    const Image origSub2 = subsample_luma(orgPad, 2);
    const Image origSub4 = subsample_luma(origSub2, 2);
    const Image bufSub2  = subsample_luma(bufPad, 2);
    const Image bufSub4  = subsample_luma(bufSub2, 2);

    Array2D<MotionVector> mv0(orgPic.w / 16, orgPic.h / 16);
    Array2D<MotionVector> mv1(orgPic.w / 16, orgPic.h / 16);
    Array2D<MotionVector> mv2(orgPic.w / 16, orgPic.h / 16);
    Array2D<MotionVector> mv(orgPic.w / 8, orgPic.h / 8);

    motion_estimation_luma(mv0, origSub4, bufSub4, 16, bitdepth);
    motion_estimation_luma(mv1, origSub2, bufSub2, 16, bitdepth, &mv0, 2);
    motion_estimation_luma(mv2, orgPad,    bufPad,  16, bitdepth, &mv1, 2);
    motion_estimation_luma(mv,  orgPad,    bufPad,   8, bitdepth, &mv2, 1, true);

    return mv;
}

Image apply_motion(const Array2D<MotionVector>& mvs, const Image& ref, int bitdepth) {
    if (ref.w % 8 != 0 || ref.h % 8 != 0) {
        throw std::runtime_error("reference size must be multiples of 8");
    }
    if (mvs.w() != ref.w / 8 || mvs.h() != ref.h / 8) {
        throw std::runtime_error("motion shape must be (H/8, W/8, 2)");
    }

    const Image bufPad = pad_with_border(ref, 4);
    Image out(ref.w, ref.h);
    const int maxValue = (1 << bitdepth) - 1;

    for (int by = 0; by < mvs.h(); ++by) {
        for (int bx = 0; bx < mvs.w(); ++bx) {
            const MotionVector mv = mvs.get(bx, by);
            const int x0 = bx * 8 + 4;
            const int y0 = by * 8 + 4;
            for (int iy = 0; iy < 8; ++iy) {
                for (int ix = 0; ix < 8; ++ix) {
                    out.at(bx * 8 + ix, by * 8 + iy) = filtered_sample_6tap(bufPad, x0 + ix, y0 + iy, mv.x, mv.y, maxValue);
                }
            }
        }
    }
    return out;
}

template <typename T>
Image numpy_to_image(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    auto b = arr.unchecked<2>();
    Image img(static_cast<int>(b.shape(1)), static_cast<int>(b.shape(0)));
    for (ssize_t y = 0; y < b.shape(0); ++y) {
        for (ssize_t x = 0; x < b.shape(1); ++x) {
            img.at(static_cast<int>(x), static_cast<int>(y)) = static_cast<int>(b(y, x));
        }
    }
    return img;
}

template <typename T>
py::array_t<T> image_to_numpy(const Image& img) {
    py::array_t<T> out({img.h, img.w});
    auto b = out.template mutable_unchecked<2>();
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            b(y, x) = static_cast<T>(img.at(x, y));
        }
    }
    return out;
}

std::pair<Image, int> parse_image_and_bitdepth(const py::array& arr, int bitdepth) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("input must be a 2D numpy array");
    }

    Image img;
    if (py::isinstance<py::array_t<uint8_t>>(arr)) {
        img = numpy_to_image<uint8_t>(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 8;
    } else if (py::isinstance<py::array_t<uint16_t>>(arr)) {
        img = numpy_to_image<uint16_t>(py::array_t<uint16_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else if (py::isinstance<py::array_t<int16_t>>(arr)) {
        img = numpy_to_image<int16_t>(py::array_t<int16_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else if (py::isinstance<py::array_t<int32_t>>(arr)) {
        img = numpy_to_image<int32_t>(py::array_t<int32_t, py::array::c_style | py::array::forcecast>(arr));
        if (bitdepth <= 0) bitdepth = 10;
    } else {
        throw std::runtime_error("supported dtypes: uint8, uint16, int16, int32");
    }
    return {img, bitdepth};
}

py::array_t<int16_t> motion_to_numpy(const Array2D<MotionVector>& mvs) {
    py::array_t<int16_t> out({mvs.h(), mvs.w(), 2});
    auto b = out.mutable_unchecked<3>();
    for (int y = 0; y < mvs.h(); ++y) {
        for (int x = 0; x < mvs.w(); ++x) {
            const auto& mv = mvs.get(x, y);
            b(y, x, 0) = static_cast<int16_t>(mv.x);
            b(y, x, 1) = static_cast<int16_t>(mv.y);
        }
    }
    return out;
}

Array2D<MotionVector> numpy_to_motion(const py::array_t<int16_t, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 3 || arr.shape(2) != 2) {
        throw std::runtime_error("motion must have shape (H/8, W/8, 2)");
    }
    auto b = arr.unchecked<3>();
    Array2D<MotionVector> mvs(static_cast<int>(b.shape(1)), static_cast<int>(b.shape(0)));
    for (ssize_t y = 0; y < b.shape(0); ++y) {
        for (ssize_t x = 0; x < b.shape(1); ++x) {
            mvs.get(static_cast<int>(x), static_cast<int>(y)) = {b(y, x, 0), b(y, x, 1), 0};
        }
    }
    return mvs;
}

py::array estimate_motion_py(const py::array& target, const py::array& reference, int bitdepth = -1) {
    auto [timg, bd0] = parse_image_and_bitdepth(target, bitdepth);
    auto [rimg, bd1] = parse_image_and_bitdepth(reference, bitdepth);
    const int bd = std::max(bd0, bd1);
    const auto mvs = motion_estimation(timg, rimg, bd);
    return motion_to_numpy(mvs);
}

py::array apply_motion_py(const py::array_t<int16_t, py::array::c_style | py::array::forcecast>& motion,
                          const py::array& reference,
                          int bitdepth = -1) {
    auto [rimg, bd] = parse_image_and_bitdepth(reference, bitdepth);
    const auto mvs = numpy_to_motion(motion);
    const Image out = apply_motion(mvs, rimg, bd);

    if (py::isinstance<py::array_t<uint8_t>>(reference)) {
        return image_to_numpy<uint8_t>(out);
    }
    if (py::isinstance<py::array_t<uint16_t>>(reference)) {
        return image_to_numpy<uint16_t>(out);
    }
    if (py::isinstance<py::array_t<int16_t>>(reference)) {
        return image_to_numpy<int16_t>(out);
    }
    return image_to_numpy<int32_t>(out);
}

} // namespace

PYBIND11_MODULE(tf_motion, m) {
    m.doc() = "Standalone pybind11 wrapper for VTM-style temporal filter motion estimation/apply-motion (luma only)";

    m.def("estimate_motion", &estimate_motion_py,
          py::arg("target"), py::arg("reference"), py::arg("bitdepth") = -1,
          R"pbdoc(
Estimate block motion from reference -> target.

Args:
    target:    2D numpy array, shape (H, W), H/W multiples of 16.
    reference: 2D numpy array, same shape/dtype as target.
    bitdepth:  Optional explicit bit depth. Default: infer from dtype.

Returns:
    int16 numpy array of shape (H/8, W/8, 2).
    Motion vectors are stored in 1/16-pel units: [..., 0] = mv_x, [..., 1] = mv_y.
)pbdoc");

    m.def("apply_motion", &apply_motion_py,
          py::arg("motion"), py::arg("reference"), py::arg("bitdepth") = -1,
          R"pbdoc(
Apply motion vectors to a reference image and return compensated prediction.

Args:
    motion:    int16 numpy array, shape (H/8, W/8, 2), in 1/16-pel units.
    reference: 2D numpy array, shape (H, W).
    bitdepth:  Optional explicit bit depth. Default: infer from dtype.

Returns:
    Compensated image with same shape/dtype as reference.
)pbdoc");
}














from setuptools import setup, Extension
import sys

try:
    import pybind11
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required. Install it first with: pip install pybind11"
    ) from e

ext_modules = [
    Extension(
        "tf_motion",
        ["tf_motion_pybind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="tf_motion",
    version="0.1.0",
    ext_modules=ext_modules,
)










import numpy as np
import tf_motion

H, W = 64, 64
ref = np.zeros((H, W), dtype=np.uint16)
ref[16:48, 16:48] = 700

target = np.zeros_like(ref)
target[18:50, 20:52] = 700  # roughly (+4, +2) pixel shift from ref to target

mv = tf_motion.estimate_motion(target, ref, bitdepth=10)
pred = tf_motion.apply_motion(mv, ref, bitdepth=10)

print("mv shape:", mv.shape)
print("center mv (1/16 pel):", mv[H//16//1, W//16//1])
print("SAD:", np.abs(pred.astype(np.int32) - target.astype(np.int32)).sum())







