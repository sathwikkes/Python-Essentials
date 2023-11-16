from website import create_app
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path






app = create_app()
if __name__=='__main__':
    app.run(debug=True)