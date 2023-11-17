from flask import Blueprint, render_template
from .models import Result
from flask import request
from .api_keys import google_api_key, search_engine_id
import requests
routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    query = request.args.get('query')
    if query == "":
        return render_template('base.html')
    if query is None:
        return render_template('base.html')
    start = 1
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={search_engine_id}&q={query}&start={start}"
    data = requests.get(url).json()
    search_items = data.get("items")
    results = []
    print(search_items)
    for i, search_item in enumerate(search_items, start=1):
        try:
            long_description = search_item["pagemap"]["metatags"][0]["og:description"]
        except KeyError:
            long_description = "N/A"
        print(long_description)
        results.append(Result(title=search_item.get("title"),description=long_description, snippet=search_item.get("snippet"),url=search_item.get("link")))
    return render_template('response_view.html', results=results)