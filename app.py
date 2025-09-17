# app.py
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from detect import analyze_image_from_path, analyze_image_from_bytes

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        # either file upload or image URL
        url = request.form.get("image_url", "").strip()
        file = request.files.get("pan_image", None)

        if url:
            # analyze image bytes fetched by requests inside analyze function
            report = analyze_image_from_bytes(url, app.config["UPLOAD_FOLDER"])
            if "error" in report:
                error = report["error"]
                return render_template("index.html", error=error)
            return render_template("result.html", report=report)
        elif file:
            if file.filename == "":
                error = "No selected file"
                return render_template("index.html", error=error)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                report = analyze_image_from_path(save_path)
                return render_template("result.html", report=report)
            else:
                error = "File type not allowed (png/jpg/jpeg)"
                return render_template("index.html", error=error)
        else:
            error = "Please upload a file or provide an image URL."
            return render_template("index.html", error=error)
    return render_template("index.html", error=error)

if __name__ == "__main__":
    app.run(debug=True)
