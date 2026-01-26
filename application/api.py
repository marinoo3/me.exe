from flask import Blueprint, jsonify, abort, current_app, request
from typing import cast

from . import AppContext

# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
api = Blueprint('api', __name__)


@api.route('ask', methods=['POST'])
def ask():
    query: str = request.json.get('query')
    if not query:
        abort(400, "Missing `query` in request body")
        
    return jsonify({
        'response': query
    })