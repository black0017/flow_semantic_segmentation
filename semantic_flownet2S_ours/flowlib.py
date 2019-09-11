""" Basic optical flow helpers. Flow visualization, I/O and metrics. """
from __future__ import division
import sys
import re
import numpy as np
import cv2


# Visualization
def flow_to_image(flow, maxval=-1):
    """ Convert flow (HxWx2 numpy array) to uint8 image according to Middlebury
    colorwheel. Maxval is the normalization scalar. If unset the maximum
    vector length is used. """
    UNKNOWN_FLOW_THRESH = 1e7
    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()

    maxu = -998.
    maxv = -999.
    minu = 999.
    minv = 999.

    # maxrad = -1
    # maxrad = 30.
    maxrad = maxval

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    # print "max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f" %
    # (maxrad, minu,maxu, minv, maxv)
    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxrad, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(
        255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(
        0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(
        255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(
        255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(
        255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


# Flow I/O
def read_flow_kitti(path):
    """ Read filepath (string), return (flow: np.float64, valid: bool). """
    img3d = read_PNG_u16(path)
    flow = (img3d[:, :, 0:2].astype('float64') - 2**15) / 64.0
    valid = img3d[:, :, 2].astype('bool')
    flow[~valid, :] = 0#  instead of ------> 10**9

    return flow, valid
# produces similar results !!!!!!!!!!!
def read_gen_flow(png_path): #for flows only
    flo_file = cv2.imread(png_path,cv2.IMREAD_UNCHANGED)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    #normalize flow !!!!!
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    valid = ~invalid.astype(bool)
    return(flo_img),(valid)


def read_flow_vkitti(path):
    """ Read png file and return (flow, valid). """
    img3d = read_PNG_u16(path)
    h, w, __ = img3d.shape
    valid = img3d[:, :, 2] != 0
    flow = 2.0 * img3d[:, :, :2].astype(np.float64) / float(2**16 - 1) - 1
    flow[:, :, 0] *= w - 1
    flow[:, :, 1] *= h - 1
    flow[~valid, :] = 10**9

    return flow, valid


def read_flow_freiburg(path):
    """ Read pfm file and return (flow, valid). """
    flow, scale = read_PFM(path)
    flow = flow[:, :, :2]
    flow *= scale
    valid = np.ones(flow.shape[:2]).astype('bool')
    return flow, valid


def read_flow_flo(filename):
    """ Read flo file and return flow array. """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if magic != 202021.25:
        print('Magic number incorrect. Invalid .flo file.')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print("Reading %d x %d flo file" % (h, w))
        # data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # data2d = np.resize(data2d, (h, w, 2))
        # Numpy bullshit adendum
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (int(h), int(w), 2))
    f.close()
    return data2d


def write_flow_flo(flow, filename):
    """Write optical flow in Middlebury .flo format.
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None. """

    if flow.dtype != np.float32:
        print('Conversion to float32, possible error.')
        flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.astype(np.float32).tofile(f)
    f.close()


def write_kitti_png(path, flow, valid=None):
    temp = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float64)
    temp[:, :, :2] = flow.astype(np.float64) * 64.0 + 2**15
    if valid is not None:
        temp[:, :, 2] = valid
    temp = temp.astype('uint16')
    write_PNG_u16(path, temp)


def write_vkitti_png(path, flow, valid=None):
    h, w, _ = flow.shape
    temp = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float64)
    a = 2**16 - 1
    temp[:, :, 0] = (flow[:, :, 0].astype(np.float64) * a)
    temp[:, :, 1] = (flow[:, :, 1].astype(np.float64) * a)
    temp[:, :, 0] = (temp[:, :, 0] + a * (w - 1)) / float(2 * (w - 1))
    temp[:, :, 1] = (temp[:, :, 1] + a * (h - 1)) / float(2 * (h - 1))
    if valid is not None:
        temp[:, :, 2] = valid
    """ Rounding is needed because type casting acts like np.ceil. Pixels that
    end just below the correct value due to arithmetic errors get erroneously
    squashed to the integer below. """
    temp = np.round(temp).astype('uint16')
    write_PNG_u16(path, temp)


# Metrics
def calc_outliers(flow, ground_truth, valid=False):
    """ Take a flow array and ground truth values, calculate the outlier
    percentage as defined by KITTI optical flow benchmark.
    Return (outlier percentage, total valid). """
    if valid is False:
        valid = np.ones(flow.shape[:2], dtype=np.bool)

    errMag = np.linalg.norm(flow[valid, :2] - ground_truth[valid, :2],
                            axis=1)
    inliers = np.logical_or(errMag <= 3,
                            errMag <= np.linalg.norm(ground_truth[valid, :],
                                                     axis=1) * 0.05)
    return (np.sum(valid) - np.sum(inliers))/np.sum(valid)


def calc_aee(flow, ground_truth, valid):
    """ Take a flow array and ground truth values, calculate the average
    endpoint error. Return (mean_aee, total valid pixels). """

    #print(flow.shape)
    #print(ground_truth.shape)
    #print(valid.shape)
    #print(np.max(valid))
    if valid is False:
        valid = np.ones(flow.shape[:2], dtype=np.bool)

    errors = np.linalg.norm(flow[valid, :2] - ground_truth[valid, :2], axis=1)
    print(np.mean(errors))
    return np.mean(errors)

def calc_aee_cdm(flow, ground_truth, valid , cdm):
    """ Take a flow array and ground truth values, calculate the average
    endpoint error. Return (mean_aee, total valid pixels). """
    if valid is False:
        valid = np.ones(flow.shape[:2], dtype=np.bool)

    _flow = flow[valid, :2]
    _gt   = ground_truth[valid, :2]
    _cdm  = cdm[valid,:2]
    temp  = ( _flow - _gt ) + ( np.multiply( (_flow - _gt) , _cdm ))  
    errors = np.linalg.norm( temp ,  axis=1 )
    print(np.mean(errors))
    return np.mean(errors)


# File I/O
def read_PFM(path):
    infile = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = infile.readline().rstrip()
    if header.decode('ASCII') == 'PF':
        color = True
    elif header.decode('ASCII') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                         infile.readline().decode('ASCII'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(infile.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(infile, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_PFM(path, image, scale=1):
    outfile = open(path, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and \
            image.shape[2] == 1:
        color = False
    else:
        erstring = 'Image must have H x W x 3, H x W x 1 or H x W dimensions.'
        raise Exception(erstring)

    outfile.write('PF\n' if color else 'Pf\n')
    outfile.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    outfile.write('%f\n' % scale)

    image.tofile(outfile)


def read_PNG_u16(path):
    """ Reads a PNG file as is. """
    img3d = cv2.imread(path, -1)
    # bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img3d[..., ::-1]


def write_PNG_u16(path, flow):
    """ Does not check if input flow is multichannel. """
    ret = cv2.imwrite(path, flow[..., ::-1])
    if not ret:
        print('Flow not written')


