from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    return render_template("base.html", SCRIPT_ROOT=request.script_root)


@app.route("/predict", methods=['POST'])  # Make sure to accept POST requests
def predict():
    text = request.json.get("message")  # Use request.json directly
    if text:  # Check if text is valid
        response = get_response(text)
        message = {"answer": response}
        return jsonify(message)
    else:
        return jsonify({"error": "Invalid request"}), 400  # Return an error response

if __name__ == "__main__":
    app.run(debug=True)