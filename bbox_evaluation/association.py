import numpy as np

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def associate(bboxes1, bboxes2, iou_threshold):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        matches = np.empty((0,2),dtype=int)
        unmatched_bboxes1 = np.arange(len(bboxes1))
        unmatched_bboxes2 = np.arange(len(bboxes2))
        return matches, unmatched_bboxes1, unmatched_bboxes2
    iou_matrix = iou_batch(bboxes1, bboxes2)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_bboxes1= []
    for d, det in enumerate(bboxes1):
        if(d not in matched_indices[:,0]):
            unmatched_bboxes1.append(d)
    unmatched_bboxes2 = []
    for t, trk in enumerate(bboxes2):
        if(t not in matched_indices[:,1]):
            unmatched_bboxes2.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_bboxes1.append(m[0])
            unmatched_bboxes2.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, unmatched_bboxes1, unmatched_bboxes2

def iou_batch_3d(bboxes1, bboxes2):
    """
    Computes IoU between two 3D bounding boxes in the form [x1, y1, z1, x2, y2, z2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)  # Expanding dimensions for broadcasting
    bboxes1 = np.expand_dims(bboxes1, 1)  # Expanding dimensions for broadcasting
    
    # Calculate intersection coordinates
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    zz1 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    xx2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    yy2 = np.minimum(bboxes1[..., 4], bboxes2[..., 4])
    zz2 = np.minimum(bboxes1[..., 5], bboxes2[..., 5])
    
    # Calculate intersection dimensions
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    d = np.maximum(0., zz2 - zz1)
    
    # Calculate intersection volume
    whd = w * h * d
    
    # Calculate volumes of individual boxes
    vol1 = (bboxes1[..., 3] - bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (bboxes1[..., 5] - bboxes1[..., 2])
    vol2 = (bboxes2[..., 3] - bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (bboxes2[..., 5] - bboxes2[..., 2])
    
    # Calculate IoU
    iou = whd / (vol1 + vol2 - whd)
    
    return iou

def associate_3d(bboxes1, bboxes2, iou_threshold):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        matches = np.empty((0, 2), dtype=int)
        unmatched_bboxes1 = np.arange(len(bboxes1))
        unmatched_bboxes2 = np.arange(len(bboxes2))
        return matches, unmatched_bboxes1, unmatched_bboxes2, []

    iou_matrix = iou_batch_3d(bboxes1, bboxes2)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_bboxes1 = []
    for d, det in enumerate(bboxes1):
        if d not in matched_indices[:, 0]:
            unmatched_bboxes1.append(d)
    
    unmatched_bboxes2 = []
    for t, trk in enumerate(bboxes2):
        if t not in matched_indices[:, 1]:
            unmatched_bboxes2.append(t)

    # filter out matches with low IOU
    matches = []
    iou_values = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_bboxes1.append(m[0])
            unmatched_bboxes2.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            iou_values.append(iou_matrix[m[0], m[1]])
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, unmatched_bboxes1, unmatched_bboxes2, iou_values