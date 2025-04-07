from flask import Flask
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Simple name server")
parser.add_argument("--name", type=str, required=True, help="Name to return from the server")
args = parser.parse_args()

# Flask app
app = Flask(__name__)
NAME = args.name

@app.route('/name', methods=['GET'])
def get_name():
    return NAME, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
