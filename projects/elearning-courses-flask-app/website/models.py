from . import db

#Your Model Here
class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20))
    author = db.Column(db.String(50))
    overview = db.Column(db.String(200))
    image = db.Column(db.String(200))
    url = db.Column(db.String(100))