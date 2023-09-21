"""Flask API
"""
import flask


api = flask.Flask(__name__)


@api.route("/api/testing", methods=["GET"])
def testing():
    return {"message": "It's working!"}, 200
