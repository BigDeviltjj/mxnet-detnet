import numpy as np
def generate_anchors(base_size,ratios,scales):
    base_anchor = np.array([1,1,base_size,base_size]) - 1
    ratio_anchor = []
    for r in ratios:
        ratio_anchor.append(_get_ratio_anchor(base_anchor,r))
    ret_anchor = []
    for s in scales:
        for ra in ratio_anchor:
            ret_anchor.append(_get_scale_anchor(ra,s))

    return np.vstack(ret_anchor)
def _get_whctr(anchor):
    w = anchor[2] - anchor[0]+1
    h = anchor[3] - anchor[1]+1
    xctr = anchor[0] + (w-1)/2
    yctr = anchor[1] + (h-1)/2
    return w,h,xctr,yctr

def _get_anchor(w,h,xctr,yctr):
    xmin = xctr - (w-1)/2
    ymin = yctr - (h-1)/2
    xmax = xctr + (w-1)/2
    ymax = yctr + (h-1)/2
    return np.hstack([xmin,ymin,xmax,ymax])
def _get_ratio_anchor(anchor, r):
    w,h,xctr,yctr = _get_whctr(anchor)
    ws = np.round(np.sqrt(w*h/r))
    hs = np.round(ws * r)
    return _get_anchor(ws,hs,xctr,yctr)

def _get_scale_anchor(anchor, s):
    w,h,xctr,yctr = _get_whctr(anchor)
    ws = w * s
    hs = h * s
    return _get_anchor(ws,hs,xctr,yctr)

