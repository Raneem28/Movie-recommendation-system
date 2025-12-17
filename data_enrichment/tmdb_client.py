import requests

API_KEY = "f403e7212241c522f963249f2e5a16c3"
SESSION = requests.Session()

def search_movie(title, year=None):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": title}
    if year:
        params['year'] = int(year)
    # Use session
    try:
        res = SESSION.get(url, params=params, timeout=5).json()
        return res["results"][0] if res.get("results") else None
    except Exception:
        return None

def search_movies_list(title, limit=5):
    """Return list of top movie matches from TMDB"""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": title}
    try:
        res = SESSION.get(url, params=params, timeout=5).json()
        return res.get("results", [])[:limit]
    except Exception:
        return []

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY, "append_to_response": "credits"}
    try:
        return SESSION.get(url, params=params, timeout=5).json()
    except Exception:
        return {}
def extract_metadata(movie_json):
    runtime = movie_json.get("runtime", 0)

    # Safely extract cast (handle missing credits)
    cast = []
    if "credits" in movie_json and "cast" in movie_json["credits"]:
        cast = [
            actor["name"]
            for actor in movie_json["credits"]["cast"][:5]
        ]

    genres = [g["name"] for g in movie_json.get("genres", [])]

    return runtime, cast, genres
