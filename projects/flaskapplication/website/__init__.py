from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path

db = SQLAlchemy()
DB_NAME = "database.db"
def create_app():
    #Creating new App
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "Simple Secret Key"

    # # Adding database to the application
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    # Adding routes
    from .routes import routes
    app.register_blueprint(routes, url_prefix="/")

    # Create Database
    create_database(app)
    return app

def create_database(app):
    print(app.app_context().app.extensions)
    print(db.metadata.tables)
    if not path.exists('website/' + DB_NAME):
        with app.app_context():
            db.create_all()
        print("Database Created")
        # db.create_all(app=app)
        # print("Database Created")

