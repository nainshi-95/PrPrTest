#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

inline int clamp_int(int v, int lo, int hi)
{
    return std::min(std::max(v, lo), hi);
}

struct MotionVector
{
    int x = 0;
    int y = 0;
    int64_t error = INT64_MAX;
};

struct Image
{
    int w = 0;
    int h = 0;
    std::vector<int> pix;

    Image() {}
    Image(int width, int height) : w(width), h(height), pix(width * height, 0) {}

    int& at(int x, int y)
    {
        return pix[y * w + x];
    }

    int at(int x, int y) const
    {
        return pix[y * w + x];
    }

    int at_clamped(int x, int y) const
    {
        x = clamp_int(x, 0, w - 1);
        y = clamp_int(y, 0, h - 1);
        return pix[y * w + x];
    }
};

template<typename T>
Image numpy_to_image(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr)
{
    auto b = arr.template unchecked<2>();

    int h = b.shape(0);
    int w = b.shape(1);

    Image img(w, h);

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.at(x,y) = (int)b(y,x);

    return img;
}

template<typename T>
py::array_t<T> image_to_numpy(const Image& img)
{
    py::array_t<T> out({img.h, img.w});
    auto b = out.template mutable_unchecked<2>();

    for (int y = 0; y < img.h; y++)
        for (int x = 0; x < img.w; x++)
            b(y,x) = (T)img.at(x,y);

    return out;
}

constexpr int NTAPS = 6;
constexpr int HALF_NTAPS = 2;

static const int16_t INTERP[16][6] =
{
 {0,0,64,0,0,0},
 {1,-3,63,4,-2,1},
 {1,-5,62,8,-3,1},
 {1,-6,60,13,-4,0},
 {1,-7,57,19,-5,-1},
 {1,-8,54,24,-6,-1},
 {1,-9,50,29,-6,-1},
 {1,-9,46,35,-7,-2},
 {1,-10,42,42,-10,1},
 {-2,-7,35,46,-9,1},
 {-1,-6,29,50,-9,1},
 {-1,-6,24,54,-8,1},
 {-1,-5,19,57,-7,1},
 {0,-4,13,60,-6,1},
 {1,-3,8,62,-5,1},
 {1,-2,4,63,-3,1}
};

int filtered_sample_6tap(
    const Image& buf,
    int base_x,
    int base_y,
    int dx,
    int dy,
    int max_value)
{
    int dxFrac = dx & 15;
    int dyFrac = dy & 15;

    int dxInt = dx >> 4;
    int dyInt = dy >> 4;

    const int* xFilter = INTERP[dxFrac];
    const int* yFilter = INTERP[dyFrac];

    int ox = base_x + dxInt;
    int oy = base_y + dyInt;

    if (dxFrac == 0 && dyFrac == 0)
        return buf.at_clamped(ox,oy);

    if (dxFrac == 0)
    {
        int sum = 1<<5;

        for(int k=0;k<NTAPS;k++)
            sum += buf.at_clamped(ox, oy + k - HALF_NTAPS) * yFilter[k];

        return clamp_int(sum>>6,0,max_value);
    }

    if (dyFrac == 0)
    {
        int sum = 1<<5;

        for(int k=0;k<NTAPS;k++)
            sum += buf.at_clamped(ox + k - HALF_NTAPS, oy) * xFilter[k];

        return clamp_int(sum>>6,0,max_value);
    }

    int temp[NTAPS];

    for(int ky=0; ky<NTAPS; ky++)
    {
        int sum = 0;
        int yy = oy + ky - HALF_NTAPS;

        for(int kx=0;kx<NTAPS;kx++)
            sum += buf.at_clamped(ox + kx - HALF_NTAPS, yy) * xFilter[kx];

        temp[ky] = sum;
    }

    int sum = 1<<11;

    for(int k=0;k<NTAPS;k++)
        sum += temp[k] * yFilter[k];

    return clamp_int(sum>>12,0,max_value);
}

int64_t motion_error_luma(
    const Image& orig,
    const Image& buffer,
    int x,
    int y,
    int dx,
    int dy,
    int bs,
    int64_t besterror,
    int max_value)
{
    int64_t error = 0;

    for(int yy=0; yy<bs; yy++)
    {
        for(int xx=0; xx<bs; xx++)
        {
            int pred = filtered_sample_6tap(
                buffer,
                x + xx,
                y + yy,
                dx,
                dy,
                max_value);

            int diff = orig.at(x+xx, y+yy) - pred;

            error += diff*diff;
        }

        if(error > besterror)
            return error;
    }

    return error;
}

py::array estimate_motion_py(
    const py::array& target,
    const py::array& reference,
    int bitdepth)
{
    auto [tbuf,bd0] = std::make_pair(
        numpy_to_image<uint16_t>(target.cast<py::array_t<uint16_t>>()),
        bitdepth);

    auto [rbuf,bd1] = std::make_pair(
        numpy_to_image<uint16_t>(reference.cast<py::array_t<uint16_t>>()),
        bitdepth);

    int H = tbuf.h;
    int W = tbuf.w;

    if(W%16 || H%16)
        throw std::runtime_error("input must be multiple of 16");

    int mvW = W/8;
    int mvH = H/8;

    py::array_t<int16_t> out({mvH,mvW,2});
    auto mv = out.mutable_unchecked<3>();

    int maxv = (1<<bitdepth)-1;

    for(int by=0; by<mvH; by++)
    {
        for(int bx=0; bx<mvW; bx++)
        {
            MotionVector best;

            int x = bx*8;
            int y = by*8;

            for(int dy=-64; dy<=64; dy+=16)
            for(int dx=-64; dx<=64; dx+=16)
            {
                int64_t err = motion_error_luma(
                    tbuf,
                    rbuf,
                    x,
                    y,
                    dx,
                    dy,
                    8,
                    best.error,
                    maxv);

                if(err < best.error)
                {
                    best.x = dx;
                    best.y = dy;
                    best.error = err;
                }
            }

            mv(by,bx,0) = best.x;
            mv(by,bx,1) = best.y;
        }
    }

    return out;
}

py::array apply_motion_py(
    const py::array_t<int16_t>& motion,
    const py::array& reference,
    int bitdepth)
{
    auto rimg = numpy_to_image<uint16_t>(
        reference.cast<py::array_t<uint16_t>>());

    int H = rimg.h;
    int W = rimg.w;

    auto mv = motion.unchecked<3>();

    Image out(W,H);

    int maxv = (1<<bitdepth)-1;

    for(int by=0; by<mv.shape(0); by++)
    {
        for(int bx=0; bx<mv.shape(1); bx++)
        {
            int dx = mv(by,bx,0);
            int dy = mv(by,bx,1);

            int x0 = bx*8;
            int y0 = by*8;

            for(int yy=0; yy<8; yy++)
            for(int xx=0; xx<8; xx++)
            {
                out.at(x0+xx,y0+yy) =
                    filtered_sample_6tap(
                        rimg,
                        x0+xx,
                        y0+yy,
                        dx,
                        dy,
                        maxv);
            }
        }
    }

    return image_to_numpy<uint16_t>(out);
}

PYBIND11_MODULE(tf_motion, m)
{
    m.def("estimate_motion", &estimate_motion_py);
    m.def("apply_motion", &apply_motion_py);
}





def pad_to_multiple(img, m=16, border=64):
    import numpy as np

    img = np.pad(img, ((border,border),(border,border)), mode="edge")

    H,W = img.shape

    H2 = ((H + m - 1)//m)*m
    W2 = ((W + m - 1)//m)*m

    img = np.pad(img, ((0,H2-H),(0,W2-W)), mode="edge")

    return img





