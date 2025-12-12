from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
from diet_analyzer import analyze_image

app = Flask(__name__)
app.secret_key = "fast-food-diet-analyzer-secret"  # change if you want


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file field in the request.", "warning")
            return redirect(url_for("index"))

        file = request.files["image"]
        if file.filename == "":
            flash("Please select an image before clicking Analyze.", "warning")
            return redirect(url_for("index"))

        try:
            pil_img = Image.open(file.stream).convert("RGB")
        except Exception:
            flash("Could not read the uploaded file as an image.", "danger")
            return redirect(url_for("index"))

        # Run your full pipeline (YOLO + Mask R-CNN + CNN + calories)
        results = analyze_image(pil_img, return_images=True)

        if not results["per_item"]:
            flash("No supported food items detected. Try another image.", "info")

    return render_template("index.html", results=results)


if __name__ == "__main__":
    # For development. In production, use gunicorn/uwsgi.
    app.run(host="0.0.0.0", port=5000, debug=True)
