from . import db
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20))
    description = db.Column(db.String(100))
    snippet=db.Column(db.String(500))
    url= db.Column(db.String(100))