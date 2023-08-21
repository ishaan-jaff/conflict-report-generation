from flask import Flask, request, jsonify
import argparse
from generate_report import main


app = Flask(__name__)

@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        # Parse the request data
        request_data = request.get_json()
        print("got request on flask app with args: ", request_data)

        upper_left = request_data.get("upper_left")
        lower_right = request_data.get("lower_right")
        dates = request_data.get("dates")

        # Create argparse namespace
        args = argparse.Namespace(
            upper_left=upper_left,
            lower_right=lower_right,
            dates=dates
        )

        # Call the main function from the provided code
        main(args)

        response = {"message": "Report generation successful."}
        return jsonify(response), 200

    except Exception as e:
        response = {"error": str(e)}
        return jsonify(response), 500

if __name__ == "__main__":
    app.run(debug=True)
