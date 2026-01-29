from flask import Flask, request, jsonify, render_template
import util
import os

# Tell Flask where UI lives (DO NOT MOVE UI FOLDER)
app = Flask(
    __name__,
    template_folder="../UI",
    static_folder="../UI"
)


@app.route('/')
def home():
    # Serve UI
    return render_template("app.html")


@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form['image_data']
    result = util.classify_image(image_data)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()

    # REQUIRED for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
