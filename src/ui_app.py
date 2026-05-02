from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "ml-latest-small"
OUTPUT_DIR = ROOT_DIR / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RATING_THRESHOLD = 4.0


@dataclass(frozen=True)
class DataStore:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    rules: pd.DataFrame
    users: pd.DataFrame
    liked_by_user: dict[int, set[int]]
    movie_titles: dict[int, str]
    movie_rating_counts: dict[int, int]
    summary: dict[str, object]


class CustomRecommendationRequest(BaseModel):
    name: str = ""
    liked_movie_ids: list[int] = Field(min_length=1, max_length=200)
    top_n: int = Field(default=10, ge=1, le=30)


def parse_id_set(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    text = str(value).strip()
    if not text:
        return frozenset()
    return frozenset(int(part) for part in text.split(",") if part.strip())


def as_builtin(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_recommendations(
    store: DataStore,
    liked: set[int],
    top_n: int,
    user_label: str,
) -> dict[str, object]:
    recommendations: dict[int, dict[str, object]] = {}
    triggered_rules = 0

    for rule in store.rules.itertuples(index=False):
        antecedent = rule.antecedent_set
        if not antecedent or not antecedent.issubset(liked):
            continue
        triggered_rules += 1
        for movie_id in rule.consequent_set:
            if movie_id in liked:
                continue
            current = recommendations.get(movie_id)
            if current is None or rule.score > current["score"]:
                recommendations[movie_id] = {
                    "movieId": int(movie_id),
                    "title": store.movie_titles.get(int(movie_id), str(movie_id)),
                    "score": float(rule.score),
                    "confidence": float(rule.confidence),
                    "lift": float(rule.lift),
                    "support": float(rule.support),
                    "reason": rule.antecedents,
                    "rule": f"{rule.antecedents} => {rule.consequents}",
                }

    ranked = sorted(
        recommendations.values(),
        key=lambda item: (item["score"], item["confidence"], item["lift"]),
        reverse=True,
    )[:top_n]

    return {
        "user_label": user_label,
        "top_n": top_n,
        "liked_count": len(liked),
        "triggered_rules": triggered_rules,
        "recommendations": ranked,
    }


@lru_cache(maxsize=1)
def load_store() -> DataStore:
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    rules_path = OUTPUT_DIR / "association_rules.csv"
    summary_path = OUTPUT_DIR / "summary.json"

    missing = [
        str(path.relative_to(ROOT_DIR))
        for path in [ratings_path, movies_path, rules_path, summary_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required files: "
            + ", ".join(missing)
            + ". Run src/movie_apriori_recommender.py first."
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    rules = pd.read_csv(
        rules_path,
        dtype={"antecedent_ids": str, "consequent_ids": str},
    )
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    movie_titles = movies.set_index("movieId")["title"].to_dict()

    rules["antecedent_set"] = rules["antecedent_ids"].apply(parse_id_set)
    rules["consequent_set"] = rules["consequent_ids"].apply(parse_id_set)
    rules["score"] = rules["confidence"] * rules["lift"]

    liked = ratings.loc[ratings["rating"] >= RATING_THRESHOLD, ["userId", "movieId"]]
    liked_by_user = (
        liked.groupby("userId")["movieId"].apply(lambda values: set(map(int, values))).to_dict()
    )
    movie_rating_counts = ratings.groupby("movieId").size().astype(int).to_dict()

    rating_stats = ratings.groupby("userId").agg(
        ratings_count=("movieId", "count"),
        avg_rating=("rating", "mean"),
        last_rating=("rated_at", "max"),
    )
    liked_stats = liked.groupby("userId").size().rename("liked_count")
    users = (
        rating_stats.join(liked_stats, how="left")
        .fillna({"liked_count": 0})
        .reset_index()
        .sort_values("userId")
    )
    users["liked_count"] = users["liked_count"].astype(int)
    users["last_rating"] = users["last_rating"].dt.strftime("%Y-%m-%d")

    return DataStore(
        ratings=ratings,
        movies=movies,
        rules=rules,
        users=users,
        liked_by_user=liked_by_user,
        movie_titles={int(k): v for k, v in movie_titles.items()},
        movie_rating_counts={int(k): int(v) for k, v in movie_rating_counts.items()},
        summary=summary,
    )


app = FastAPI(title="MovieLens Apriori Recommender")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/summary")
def api_summary() -> dict[str, object]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "dataset": store.summary.get("dataset", {}),
        "parameters": store.summary.get("parameters", {}),
        "apriori": store.summary.get("apriori", {}),
        "user_based_cf": store.summary.get("user_based_cf", {}),
        "files": {
            "rules": "outputs/association_rules.csv",
            "itemsets": "outputs/frequent_itemsets.csv",
            "summary": "outputs/summary.json",
            "rating_chart": "outputs/rating_distribution.png",
            "genre_chart": "outputs/top_genres.png",
        },
    }


@app.get("/api/users")
def api_users(q: str = "", limit: int = Query(default=610, ge=1, le=610)) -> list[dict[str, object]]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    users = store.users.copy()
    if q.strip():
        users = users[users["userId"].astype(str).str.contains(q.strip(), regex=False)]
    users = users.head(limit)
    records = users.to_dict(orient="records")
    return [{key: as_builtin(value) for key, value in record.items()} for record in records]


@app.get("/api/movies")
def api_movies(q: str = "", limit: int = Query(default=30, ge=1, le=100)) -> list[dict[str, object]]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    movies = store.movies.copy()
    if q.strip():
        needle = q.strip().lower()
        title_lower = movies["title"].str.lower()
        movies = movies[title_lower.str.contains(needle, regex=False)].copy()
        if not movies.empty:
            matched_title = movies["title"].str.lower()
            starts_title = matched_title.str.startswith(needle)
            starts_word = matched_title.apply(
                lambda title: any(word.startswith(needle) for word in title.split())
            )
            movies["match_rank"] = np.select(
                [starts_title, starts_word],
                [0, 1],
                default=2,
            )
            movies["rating_count"] = movies["movieId"].map(store.movie_rating_counts).fillna(0)
            movies = movies.sort_values(
                ["match_rank", "rating_count", "title"],
                ascending=[True, False, True],
            ).drop(columns=["match_rank", "rating_count"])
    movies = movies.head(limit)
    return movies[["movieId", "title", "genres"]].to_dict(orient="records")


@app.get("/api/user/{user_id}")
def api_user(user_id: int) -> dict[str, object]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    user_rows = store.users.loc[store.users["userId"] == user_id]
    if user_rows.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    liked_ids = store.liked_by_user.get(user_id, set())
    user_ratings = store.ratings.loc[
        (store.ratings["userId"] == user_id) & (store.ratings["movieId"].isin(liked_ids))
    ].merge(store.movies, on="movieId", how="left")
    liked_movies = (
        user_ratings.sort_values(["rating", "rated_at"], ascending=[False, False])
        [["movieId", "title", "genres", "rating", "rated_at"]]
        .head(50)
        .copy()
    )
    liked_movies["rated_at"] = liked_movies["rated_at"].dt.strftime("%Y-%m-%d")

    profile = user_rows.iloc[0].to_dict()
    profile = {key: as_builtin(value) for key, value in profile.items()}
    return {
        "profile": profile,
        "liked_movies": liked_movies.to_dict(orient="records"),
    }


@app.get("/api/recommendations/{user_id}")
def api_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=30),
) -> dict[str, object]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user_id not in set(store.users["userId"]):
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    liked = store.liked_by_user.get(user_id, set())
    result = build_recommendations(store, liked, top_n, f"User {user_id}")
    result["userId"] = user_id
    return result


@app.post("/api/custom-recommendations")
def api_custom_recommendations(payload: CustomRecommendationRequest) -> dict[str, object]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    movie_ids = list(dict.fromkeys(int(movie_id) for movie_id in payload.liked_movie_ids))
    known_movie_ids = set(store.movie_titles)
    invalid = [movie_id for movie_id in movie_ids if movie_id not in known_movie_ids]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Movie ID không tồn tại trong dataset: {invalid[:10]}",
        )

    liked = set(movie_ids)
    selected_movies = [
        {
            "movieId": movie_id,
            "title": store.movie_titles[movie_id],
        }
        for movie_id in movie_ids
    ]
    user_label = payload.name.strip() or "User mới"
    result = build_recommendations(store, liked, payload.top_n, user_label)
    result["selected_movies"] = selected_movies
    return result


@app.get("/api/rules")
def api_rules(limit: int = Query(default=25, ge=1, le=100)) -> list[dict[str, object]]:
    try:
        store = load_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "score",
    ]
    records = store.rules.sort_values(
        ["score", "confidence", "lift"], ascending=False
    ).head(limit)[columns].to_dict(orient="records")
    return [{key: as_builtin(value) for key, value in record.items()} for record in records]
