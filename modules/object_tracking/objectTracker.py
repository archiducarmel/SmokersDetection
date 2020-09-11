from .deepsort.deep_sort import DeepSort

def p1p2Toxywh(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def getTracker(type, weights_path, min_confidence, use_cuda, nn_budget, n_init, max_iou_distance, max_dist, max_age):
    if type == "DeepSort":
        tracker = DeepSort(weights_path,
                           min_confidence=min_confidence,
                           use_cuda=use_cuda,
                           nn_budget=nn_budget,
                           n_init=n_init,
                           max_iou_distance=max_iou_distance,
                           max_dist=max_dist,
                           max_age=max_age)
    else:
        tracker = None

    return tracker