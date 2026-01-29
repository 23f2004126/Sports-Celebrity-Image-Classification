from flask import Flask, request, jsonify, send_from_directory
import util
import os

# IMPORTANT: tell Flask where UI lives
app = Flask(
    __name__,
    static_folder="../UI",
    static_url_path=""
)

# ---------- ROUTES ----------

@app.route("/")
def home():
    # serve app.html from UI folder
    return send_from_directory("../UI", "app.html")


@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.form.get("image_data")
    result = util.classify_image(image_data)
    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# ---------- MAIN ----------
if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()

    # Render provides PORT dynamically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
