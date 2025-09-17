# detect.py
import os
import io
import cv2
import numpy as np
import requests
import imutils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# Helper: save cv2 image (BGR) to path
def _save_bgr_image(img_bgr, out_path):
    cv2.imwrite(out_path, img_bgr)
    return out_path

# Analyze from a local image path
def analyze_image_from_path(path):
    # read image (color)
    img = cv2.imread(path)
    if img is None:
        return {"error": f"Unable to read image at {path}"}

    basename = os.path.splitext(os.path.basename(path))[0]
    out_prefix = os.path.join(os.path.dirname(path), basename)

    return _analyze_core(img, path, out_prefix)

# Analyze by fetching from URL bytes (requests). Returns similar report
def analyze_image_from_bytes(url, upload_folder):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Could not fetch image from URL: {e}"}

    # read image bytes into numpy array
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Downloaded content is not a valid image."}

    # determine a safe filename
    basename = "url_image"
    out_prefix = os.path.join(upload_folder, basename)
    # ensure unique by appending number if exists
    i = 0
    while os.path.exists(f"{out_prefix}_{i}.jpg"):
        i += 1
    out_prefix = f"{out_prefix}_{i}"
    orig_path = f"{out_prefix}.jpg"
    cv2.imwrite(orig_path, img)
    return _analyze_core(img, orig_path, out_prefix)

# Core analysis (returns dict)
def _analyze_core(img_bgr, orig_path, out_prefix):
    report = {"input": orig_path}
    h, w = img_bgr.shape[:2]
    report["dimensions"] = {"width": int(w), "height": int(h)}

    # 1) Compression-difference (an ELA-like signal) via encode / decode + structural similarity
    try:
        # encode to JPEG at quality 90 (simulate recompression)
        success, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise RuntimeError("Encoding failed")
        recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # grayscale versions for SSIM/MSE
        gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_recomp = cv2.cvtColor(recompressed, cv2.COLOR_BGR2GRAY)

        # compute structural similarity and mse
        s, diff = ssim(gray_orig, gray_recomp, full=True)
        mse_val = mean_squared_error(gray_orig, gray_recomp)

        diff_img = (diff * 255).astype("uint8")
        # amplify difference for visualization: apply color map
        diff_vis = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
        ela_path = out_prefix + "_diff.jpg"
        _save_bgr_image(diff_vis, ela_path)

        report["compression_similarity"] = {"ssim": float(s), "mse": float(mse_val)}
        report["compression_diff_image"] = ela_path

        # heuristic: low ssim (<<1) and high mse suggests visible changes after recompress
        if s < 0.95 or mse_val > 200:
            report.setdefault("flags", []).append("Compression-difference moderate/strong")
        else:
            report.setdefault("notes", []).append("Compression-difference low")
    except Exception as e:
        report["compression_error"] = str(e)

    # 2) ORB copy-move detection (internal duplicates)
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # resize for speed if large
        max_dim = 1200
        scale = 1.0
        if max(gray.shape) > max_dim:
            scale = max_dim / float(max(gray.shape))
            gray_small = imutils.resize(gray, width=int(gray.shape[1]*scale))
            img_vis_small = imutils.resize(img_bgr, width=int(img_bgr.shape[1]*scale))
        else:
            gray_small = gray.copy()
            img_vis_small = img_bgr.copy()

        orb = cv2.ORB_create(2000)
        kp, des = orb.detectAndCompute(gray_small, None)
        if des is None or len(kp) < 8:
            report["orb"] = {"matches": 0, "message": "Not enough keypoints for copy-move detection"}
        else:
            # match descriptors to themselves (exclude trivial self matches)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn_matches = bf.knnMatch(des, des, k=2)

            good = []
            for pair in knn_matches:
                if len(pair) != 2:
                    continue
                m, n = pair
                # avoid identical keypoint
                if m.queryIdx == m.trainIdx:
                    continue
                # ratio test
                if m.distance < 0.75 * n.distance:
                    pt1 = kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    # exclude very close matches (likely same feature neighborhood)
                    if np.linalg.norm(np.array(pt1) - np.array(pt2)) > 25:
                        good.append((m, pt1, pt2))

            # draw lines for top matches
            vis = img_vis_small.copy()
            for idx, (m, pt1, pt2) in enumerate(good[:300]):
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))
                cv2.line(vis, p1, p2, (0, 0, 255), 1)

            orb_out = out_prefix + "_orb.jpg"
            _save_bgr_image(vis, orb_out)

            matches_count = len(good)
            report["orb"] = {"matches": int(matches_count), "visual": orb_out}
            # heuristic thresholds
            if matches_count > 80:
                report.setdefault("flags", []).append("Many internal keypoint matches (possible copy-move)")
            else:
                report.setdefault("notes", []).append("No strong copy-move signals")

    except Exception as e:
        report["orb_error"] = str(e)

    # 3) Laplacian variance (blur / smoothing detection)
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        report["laplacian_variance"] = float(var)
        if var < 100:
            report.setdefault("flags", []).append("Low laplacian variance (image may be smoothed/edited)")
        else:
            report.setdefault("notes", []).append("Normal texture/edge variance")
    except Exception as e:
        report["laplacian_error"] = str(e)

    # 4) Simple aggregated score heuristic (0-100)
    score = 50  # start neutral
    # SSIM adjustment
    ssim_val = report.get("compression_similarity", {}).get("ssim", 1.0)
    if ssim_val < 0.9:
        score -= 20
    elif ssim_val < 0.97:
        score -= 5
    else:
        score += 10

    # ORB matches
    orb_matches = report.get("orb", {}).get("matches", 0)
    if orb_matches > 120:
        score -= 25
    elif orb_matches > 60:
        score -= 10
    else:
        score += 10

    # Laplacian
    if report.get("laplacian_variance", 0) < 100:
        score -= 15
    else:
        score += 5

    # clamp
    score = max(0, min(100, score))
    report["final_score"] = int(score)

    # human-readable verdict
    if score >= 70:
        report["verdict"] = "Likely original (no strong automated tamper signals detected)"
    elif score >= 40:
        report["verdict"] = "Suspicious — manual inspection recommended"
    else:
        report["verdict"] = "Likely tampered (several automated indicators)"

    return report
