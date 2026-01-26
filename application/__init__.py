from flask import Flask  




class AppContext(Flask):
    pass


def create_app():

    # Create Flask app
    app = Flask(__name__)

    # Load configuration
    with app.app_context():
        pass

    # Init backend ajax 
    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')


    return app