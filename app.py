from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, abort, flash, jsonify
from werkzeug.utils import secure_filename
import sqlite3
import os
import shutil
import uuid
import re
import io
from pathlib import Path

import fitz
from PIL import Image, ImageOps
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "showroom.db"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_PDF = {"pdf"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}


def allowed(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def clean_image(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    mask = gray < 248
    ys, xs = np.where(mask)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
        if (x1 > 0.04*w or y1 > 0.04*h or (w-1-x2) > 0.04*w or (h-1-y2) > 0.04*h):
            pad_x = int(w * 0.01); pad_y = int(h * 0.01)
            img = img.crop((max(0,x1-pad_x), max(0,y1-pad_y), min(w,x2+1+pad_x), min(h,y2+1+pad_y)))
    img.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    img.save(path, "JPEG", quality=92, optimize=True)


def extract_pdf(pdf_path, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for page in doc:
        images = page.get_images(full=True)
        if images:
            for image in images:
                data = doc.extract_image(image[0])
                ext = data.get("ext", "jpg")
                count += 1
                raw = output_folder / f"_raw_{count}.{ext}"
                out = output_folder / f"car_photo_{count}.jpg"
                raw.write_bytes(data["image"])
                try:
                    im = Image.open(raw).convert("RGB")
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
                    im.save(out, "JPEG", quality=92, optimize=True)
                finally:
                    raw.unlink(missing_ok=True)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
            count += 1
            out = output_folder / f"car_photo_{count}.jpg"
            pix.save(str(out))
            clean_image(out)
    doc.close()
    return count


def image_list(folder):
    return sorted(folder.glob("car_photo_*.jpg"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))


def _pct_box_to_px(r, iw, ih):
    x = max(0, min(float(r.get("x", 0)), 100))
    y = max(0, min(float(r.get("y", 0)), 100))
    w = max(0, min(float(r.get("w", 0)), 100 - x))
    h = max(0, min(float(r.get("h", 0)), 100 - y))
    x1 = int(x * iw / 100); y1 = int(y * ih / 100)
    x2 = max(x1 + 2, int((x + w) * iw / 100)); y2 = max(y1 + 2, int((y + h) * ih / 100))
    return x1, y1, min(iw, x2), min(ih, y2)


def _backup_originals(folder):
    for p in image_list(folder):
        backup = p.with_name(".original_" + p.name)
        if not backup.exists():
            shutil.copy2(p, backup)


def _detect_logo_box(template_bgr, image_bgr):
    """Find a selected logo crop in another photo, even when its position/scale changes."""
    th, tw = template_bgr.shape[:2]
    ih, iw = image_bgr.shape[:2]
    if th < 8 or tw < 8 or ih < th or iw < tw:
        return None, 0.0

    # First try ORB feature matching for movement/scale changes.
    try:
        orb = cv2.ORB_create(nfeatures=700)
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), None)
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:60]
            good = [m for m in matches if m.distance < 65]
            if len(good) >= 4:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 6.0)
                if H is not None and inliers is not None and int(inliers.sum()) >= 4:
                    corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
                    mapped = cv2.perspectiveTransform(corners, H).reshape(-1,2)
                    x1 = max(0, int(mapped[:,0].min())); y1 = max(0, int(mapped[:,1].min()))
                    x2 = min(iw, int(mapped[:,0].max())); y2 = min(ih, int(mapped[:,1].max()))
                    bw, bh = x2-x1, y2-y1
                    if bw >= 8 and bh >= 8 and bw < iw*0.6 and bh < ih*0.6:
                        score = int(inliers.sum()) / max(1, len(good))
                        return (x1,y1,x2,y2), score
    except Exception:
        pass

    # Fallback: multi-scale normalized template matching.
    gray_t = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    best = None
    for scale in np.linspace(0.55, 1.65, 23):
        nw, nh = int(tw*scale), int(th*scale)
        if nw < 8 or nh < 8 or nw >= iw or nh >= ih:
            continue
        resized = cv2.resize(gray_t, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        result = cv2.matchTemplate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if best is None or score > best[0]:
            best = (score, loc[0], loc[1], nw, nh)
    if best and best[0] >= 0.58:
        score,x,y,w,h = best
        return (x,y,x+w,y+h), float(score)
    return None, 0.0


def _place_logo(result_bgr, logo_path, box):
    if not logo_path or not Path(logo_path).exists():
        return result_bgr
    x1,y1,x2,y2 = box
    target_w, target_h = max(2,x2-x1), max(2,y2-y1)
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    base = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    lw, lh = logo.size
    px = x1 + max(0, (target_w-lw)//2); py = y1 + max(0, (target_h-lh)//2)
    base.alpha_composite(logo, (px, py))
    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def apply_regions(image_path, regions, logo_path=None):
    img = cv2.imread(str(image_path))
    if img is None: return False
    ih, iw = img.shape[:2]
    mask = np.zeros((ih, iw), dtype=np.uint8); boxes=[]
    for r in regions:
        box=_pct_box_to_px(r,iw,ih)
        x1,y1,x2,y2=box; mask[y1:y2,x1:x2]=255; boxes.append(box)
    if not boxes: return False
    result=cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)
    for box in boxes: result=_place_logo(result,logo_path,box)
    cv2.imwrite(str(image_path),result,[int(cv2.IMWRITE_JPEG_QUALITY),92]); return True


def apply_batch_auto(folder, regions, logo_path=None):
    images=image_list(folder)
    if not images: return 0,0
    _backup_originals(folder)
    first_original=folder/('.original_'+images[0].name)
    src=cv2.imread(str(first_original))
    ih,iw=src.shape[:2]
    templates=[]
    for r in regions:
        x1,y1,x2,y2=_pct_box_to_px(r,iw,ih)
        crop=src[y1:y2,x1:x2].copy()
        if crop.size: templates.append((crop,(x1,y1,x2,y2)))
    processed=0; detected=0
    for idx,p in enumerate(images):
        original=folder/('.original_'+p.name)
        img=cv2.imread(str(original))
        if img is None: continue
        ih2,iw2=img.shape[:2]; boxes=[]
        if idx==0:
            boxes=[b for _,b in templates]
        else:
            for tmpl, fallback in templates:
                box,score=_detect_logo_box(tmpl,img)
                if box is not None:
                    boxes.append(box); detected+=1
                else:
                    # If the logo is still near the same normalized position, use the original selection.
                    x1,y1,x2,y2=fallback
                    sx,sy=iw2/iw,ih2/ih
                    boxes.append((int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)))
        if not boxes: continue
        mask=np.zeros((ih2,iw2),dtype=np.uint8)
        for x1,y1,x2,y2 in boxes:
            x1=max(0,min(x1,iw2-1)); y1=max(0,min(y1,ih2-1)); x2=max(x1+1,min(x2,iw2)); y2=max(y1+1,min(y2,ih2)); mask[y1:y2,x1:x2]=255
        result=cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)
        for box in boxes: result=_place_logo(result,logo_path,box)
        cv2.imwrite(str(p),result,[int(cv2.IMWRITE_JPEG_QUALITY),92]); processed+=1
    return processed, detected


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    pdf = request.files.get("pdf")
    if not pdf or not pdf.filename or not allowed(pdf.filename, ALLOWED_PDF):
        flash("Please select a valid PDF.")
        return redirect(url_for("home"))
    token = uuid.uuid4().hex[:12]
    folder = UPLOAD_DIR / token
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / secure_filename(pdf.filename)
    pdf.save(pdf_path)
    try:
        extract_pdf(pdf_path, folder)
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        flash(f"PDF processing failed: {e}")
        return redirect(url_for("home"))
    return redirect(url_for("photos", folder=token))


@app.route("/photos/<folder>")
def photos(folder):
    path = UPLOAD_DIR / secure_filename(folder)
    if not path.is_dir(): abort(404)
    images = [p.name for p in image_list(path)]
    return render_template("gallery.html", folder=folder, images=images)


@app.route("/logo/<folder>", methods=["POST"])
def upload_logo(folder):
    path = UPLOAD_DIR / secure_filename(folder)
    if not path.is_dir(): abort(404)
    logo = request.files.get("logo")
    if not logo or not logo.filename or not allowed(logo.filename, ALLOWED_IMAGE):
        return jsonify(ok=False, error="Please choose JPG, PNG or WEBP logo."), 400
    logo_path = path / "showroom_logo.png"
    logo.save(logo_path)
    return jsonify(ok=True, logo_url=url_for("image", folder=folder, filename=logo_path.name))


@app.route("/cleanup-batch/<folder>", methods=["POST"])
def cleanup_batch(folder):
    folder = secure_filename(folder)
    path = UPLOAD_DIR / folder
    if not path.is_dir(): abort(404)
    try:
        regions = request.form.get("regions", "")
        regions = __import__("json").loads(regions)
        if not regions or len(regions) > 20: raise ValueError
    except Exception:
        return jsonify(ok=False, error="Invalid logo area selection."), 400
    logo = request.files.get("logo")
    logo_path = None
    if logo and logo.filename:
        if not allowed(logo.filename, ALLOWED_IMAGE):
            return jsonify(ok=False, error="Invalid logo file."), 400
        logo_path = path / "showroom_logo.png"
        logo.save(logo_path)
    else:
        existing = path / "showroom_logo.png"
        if existing.exists(): logo_path = existing
    images = image_list(path)
    if not images: return jsonify(ok=False, error="No photos found."), 400
    processed, detected = apply_batch_auto(path, regions, logo_path)
    return jsonify(ok=True, processed=processed, detected=detected, total=len(images), logo_applied=bool(logo_path), automatic=True)


@app.route("/cleanup/<folder>", methods=["POST"])
def cleanup_single(folder):
    path = UPLOAD_DIR / secure_filename(folder)
    if not path.is_dir(): abort(404)
    filename = secure_filename(request.form.get("filename", ""))
    image_path = path / filename
    if not filename.startswith("car_photo_") or image_path.parent != path or not image_path.exists(): abort(400)
    try:
        regions = [{"x": float(request.form["x"]), "y": float(request.form["y"]), "w": float(request.form["w"]), "h": float(request.form["h"])}]
    except Exception: abort(400)
    logo_path = path / "showroom_logo.png"
    return jsonify(ok=apply_regions(image_path, regions, logo_path if logo_path.exists() else None))


@app.route("/image/<folder>/<filename>")
def image(folder, filename):
    return send_from_directory(UPLOAD_DIR / secure_filename(folder), secure_filename(filename))


@app.route("/download/<folder>/<filename>")
def download(folder, filename):
    return send_from_directory(UPLOAD_DIR / secure_filename(folder), secure_filename(filename), as_attachment=True)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
