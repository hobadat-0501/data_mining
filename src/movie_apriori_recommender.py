from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Rule:
    antecedents: FrozenSet[int]
    consequents: FrozenSet[int]
    support_count: int
    support: float
    antecedent_support: float
    consequent_support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Movie recommendation with Apriori on MovieLens latest-small."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("ml-latest-small"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-user", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--cf-neighbors", type=int, default=40)
    parser.add_argument("--skip-cf", action="store_true")
    return parser.parse_args()


def load_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_files = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]
    missing = [name for name in required_files if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {', '.join(missing)}"
        )

    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    links = pd.read_csv(data_dir / "links.csv")
    return ratings, movies, tags, links


def temporal_train_test_split(
    ratings: pd.DataFrame, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"])

    for _, group in ordered.groupby("userId", sort=False):
        n_test = max(1, int(math.ceil(len(group) * test_ratio)))
        if n_test >= len(group):
            n_test = len(group) - 1
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def build_transactions(
    ratings: pd.DataFrame, rating_threshold: float
) -> dict[int, FrozenSet[int]]:
    liked = ratings.loc[ratings["rating"] >= rating_threshold, ["userId", "movieId"]]
    grouped = liked.groupby("userId")["movieId"].apply(lambda values: frozenset(values))
    return grouped.to_dict()


def dataset_stats(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    links: pd.DataFrame,
    transactions: dict[int, FrozenSet[int]],
    rating_threshold: float,
) -> dict[str, object]:
    liked_count = sum(len(items) for items in transactions.values())
    total_possible_liked = ratings["userId"].nunique() * movies["movieId"].nunique()
    liked_density = liked_count / total_possible_liked if total_possible_liked else 0.0

    per_user_liked = pd.Series([len(items) for items in transactions.values()])
    genres = (
        movies["genres"]
        .str.split("|")
        .explode()
        .replace("(no genres listed)", np.nan)
        .dropna()
    )

    return {
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "movies": int(movies["movieId"].nunique()),
        "tags": int(len(tags)),
        "links": int(len(links)),
        "rating_threshold": rating_threshold,
        "users_with_liked_movies": int(len(transactions)),
        "liked_user_movie_pairs": int(liked_count),
        "avg_liked_movies_per_user": float(per_user_liked.mean()),
        "median_liked_movies_per_user": float(per_user_liked.median()),
        "max_liked_movies_per_user": int(per_user_liked.max()),
        "liked_matrix_sparsity": float(1 - liked_density),
        "genre_count": int(genres.nunique()),
        "top_genres": genres.value_counts().head(10).to_dict(),
    }


def make_plots(
    ratings: pd.DataFrame, movies: pd.DataFrame, output_dir: Path
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=ratings, x="rating", color="#4C78A8")
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=160)
    plt.close()

    genre_counts = (
        movies["genres"]
        .str.split("|")
        .explode()
        .replace("(no genres listed)", np.nan)
        .dropna()
        .value_counts()
        .head(12)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, color="#F58518")
    plt.title("Top movie genres")
    plt.xlabel("Number of movies")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(output_dir / "top_genres.png", dpi=160)
    plt.close()


def generate_candidates(
    previous_frequent: set[FrozenSet[int]], k: int
) -> set[FrozenSet[int]]:
    previous_tuples = sorted(tuple(sorted(itemset)) for itemset in previous_frequent)
    previous_lookup = set(previous_frequent)
    candidates: set[FrozenSet[int]] = set()

    for i, left in enumerate(previous_tuples):
        for right in previous_tuples[i + 1 :]:
            if left[: k - 2] != right[: k - 2]:
                break
            candidate_tuple = tuple(sorted(set(left) | set(right)))
            if len(candidate_tuple) != k:
                continue
            candidate = frozenset(candidate_tuple)
            subsets_ok = all(
                frozenset(subset) in previous_lookup
                for subset in itertools.combinations(candidate_tuple, k - 1)
            )
            if subsets_ok:
                candidates.add(candidate)

    return candidates


def count_candidates(
    transactions: list[set[int]],
    candidates: set[FrozenSet[int]],
) -> Counter[FrozenSet[int]]:
    counts: Counter[FrozenSet[int]] = Counter()
    candidate_list = list(candidates)

    for transaction in transactions:
        for candidate in candidate_list:
            if candidate.issubset(transaction):
                counts[candidate] += 1

    return counts


def run_apriori(
    transactions_by_user: dict[int, FrozenSet[int]],
    min_support: float,
    max_len: int,
) -> tuple[pd.DataFrame, dict[FrozenSet[int], int], int]:
    if not 0 < min_support <= 1:
        raise ValueError("--min-support must be in (0, 1]")
    if max_len < 1:
        raise ValueError("--max-len must be >= 1")

    transaction_sets = [set(items) for items in transactions_by_user.values() if items]
    transaction_count = len(transaction_sets)
    min_support_count = max(1, int(math.ceil(transaction_count * min_support)))

    item_counts: Counter[FrozenSet[int]] = Counter()
    for transaction in transaction_sets:
        for movie_id in transaction:
            item_counts[frozenset([movie_id])] += 1

    current = {
        itemset: count
        for itemset, count in item_counts.items()
        if count >= min_support_count
    }

    support_lookup: dict[FrozenSet[int], int] = {}
    records: list[dict[str, object]] = []

    for k in range(1, max_len + 1):
        if not current:
            break

        for itemset, count in current.items():
            support_lookup[itemset] = count
            records.append(
                {
                    "itemsets": itemset,
                    "support_count": count,
                    "support": count / transaction_count,
                    "length": len(itemset),
                }
            )

        if k == max_len:
            break

        candidates = generate_candidates(set(current.keys()), k + 1)
        if not candidates:
            break
        counted = count_candidates(transaction_sets, candidates)
        current = {
            itemset: count
            for itemset, count in counted.items()
            if count >= min_support_count
        }

    itemsets_df = pd.DataFrame(records)
    if not itemsets_df.empty:
        itemsets_df = itemsets_df.sort_values(
            ["length", "support", "support_count"], ascending=[True, False, False]
        ).reset_index(drop=True)
    return itemsets_df, support_lookup, min_support_count


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    support_lookup: dict[FrozenSet[int], int],
    transaction_count: int,
    min_confidence: float,
    min_lift: float,
) -> list[Rule]:
    rules: list[Rule] = []
    if frequent_itemsets.empty:
        return rules

    for itemset in frequent_itemsets["itemsets"]:
        if len(itemset) < 2:
            continue
        itemset_support_count = support_lookup[itemset]
        itemset_support = itemset_support_count / transaction_count
        items = tuple(sorted(itemset))

        for size in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_count = support_lookup.get(antecedent)
                consequent_count = support_lookup.get(consequent)
                if not antecedent_count or not consequent_count:
                    continue

                antecedent_support = antecedent_count / transaction_count
                consequent_support = consequent_count / transaction_count
                confidence = itemset_support / antecedent_support
                if confidence < min_confidence:
                    continue
                lift = confidence / consequent_support if consequent_support else 0.0
                if lift <= min_lift:
                    continue
                leverage = itemset_support - antecedent_support * consequent_support
                conviction = (
                    math.inf
                    if confidence >= 1
                    else (1 - consequent_support) / (1 - confidence)
                )
                rules.append(
                    Rule(
                        antecedents=antecedent,
                        consequents=consequent,
                        support_count=itemset_support_count,
                        support=itemset_support,
                        antecedent_support=antecedent_support,
                        consequent_support=consequent_support,
                        confidence=confidence,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                    )
                )

    rules.sort(key=lambda rule: (rule.lift, rule.confidence, rule.support), reverse=True)
    return rules


def format_itemset(itemset: FrozenSet[int], movie_titles: dict[int, str]) -> str:
    return " | ".join(movie_titles.get(movie_id, str(movie_id)) for movie_id in sorted(itemset))


def itemsets_to_frame(
    itemsets: pd.DataFrame, movie_titles: dict[int, str]
) -> pd.DataFrame:
    if itemsets.empty:
        return pd.DataFrame()

    exported = itemsets.copy()
    exported["movie_ids"] = exported["itemsets"].apply(
        lambda itemset: ",".join(str(movie_id) for movie_id in sorted(itemset))
    )
    exported["titles"] = exported["itemsets"].apply(
        lambda itemset: format_itemset(itemset, movie_titles)
    )
    return exported.drop(columns=["itemsets"])[
        ["length", "support_count", "support", "movie_ids", "titles"]
    ]


def rules_to_frame(rules: list[Rule], movie_titles: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule in rules:
        rows.append(
            {
                "antecedent_ids": ",".join(str(i) for i in sorted(rule.antecedents)),
                "consequent_ids": ",".join(str(i) for i in sorted(rule.consequents)),
                "antecedents": format_itemset(rule.antecedents, movie_titles),
                "consequents": format_itemset(rule.consequents, movie_titles),
                "support_count": rule.support_count,
                "support": rule.support,
                "antecedent_support": rule.antecedent_support,
                "consequent_support": rule.consequent_support,
                "confidence": rule.confidence,
                "lift": rule.lift,
                "leverage": rule.leverage,
                "conviction": rule.conviction,
            }
        )
    return pd.DataFrame(rows)


def recommend_for_user(
    user_id: int,
    transactions_by_user: dict[int, FrozenSet[int]],
    rules: list[Rule],
    movie_titles: dict[int, str],
    top_n: int,
) -> pd.DataFrame:
    liked = set(transactions_by_user.get(user_id, frozenset()))
    if not liked:
        return pd.DataFrame(
            columns=[
                "userId",
                "movieId",
                "title",
                "score",
                "confidence",
                "lift",
                "reason",
            ]
        )

    recommendations: dict[int, dict[str, object]] = {}
    for rule in rules:
        if not rule.antecedents.issubset(liked):
            continue
        for movie_id in rule.consequents:
            if movie_id in liked:
                continue
            score = rule.confidence * rule.lift
            current = recommendations.get(movie_id)
            if current is None or score > current["score"]:
                recommendations[movie_id] = {
                    "userId": user_id,
                    "movieId": movie_id,
                    "title": movie_titles.get(movie_id, str(movie_id)),
                    "score": score,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                    "reason": format_itemset(rule.antecedents, movie_titles),
                }

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "userId",
                "movieId",
                "title",
                "score",
                "confidence",
                "lift",
                "reason",
            ]
        )

    result = pd.DataFrame(recommendations.values())
    return result.sort_values(
        ["score", "confidence", "lift"], ascending=False
    ).head(top_n).reset_index(drop=True)


def build_test_liked_by_user(
    test_ratings: pd.DataFrame, rating_threshold: float
) -> dict[int, set[int]]:
    liked = test_ratings.loc[
        test_ratings["rating"] >= rating_threshold, ["userId", "movieId"]
    ]
    return liked.groupby("userId")["movieId"].apply(lambda values: set(values)).to_dict()


def score_recommendations(
    recommendations_by_user: dict[int, list[int]],
    test_liked_by_user: dict[int, set[int]],
    evaluable_users: list[int],
    top_n: int,
) -> dict[str, float]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    hit_users = 0
    users_with_recommendations = 0

    for user_id in evaluable_users:
        actual = test_liked_by_user.get(user_id, set())
        if not actual:
            continue
        recommended = recommendations_by_user.get(user_id, [])[:top_n]
        if recommended:
            users_with_recommendations += 1
        hits = len(set(recommended) & actual)
        if hits:
            hit_users += 1
        precision_values.append(hits / top_n)
        recall_values.append(hits / len(actual))

    user_count = len(precision_values)
    return {
        "users_evaluated": int(user_count),
        "users_with_recommendations": int(users_with_recommendations),
        "precision_at_n": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_at_n": float(np.mean(recall_values)) if recall_values else 0.0,
        "hit_rate_at_n": float(hit_users / user_count) if user_count else 0.0,
        "coverage": float(users_with_recommendations / user_count) if user_count else 0.0,
    }


def evaluate_apriori(
    transactions_by_user: dict[int, FrozenSet[int]],
    rules: list[Rule],
    movie_titles: dict[int, str],
    test_liked_by_user: dict[int, set[int]],
    top_n: int,
) -> dict[str, float]:
    evaluable_users = [
        user_id
        for user_id in transactions_by_user
        if test_liked_by_user.get(user_id)
    ]
    recommendations_by_user: dict[int, list[int]] = {}
    for user_id in evaluable_users:
        recs = recommend_for_user(user_id, transactions_by_user, rules, movie_titles, top_n)
        recommendations_by_user[user_id] = recs["movieId"].astype(int).tolist()

    return score_recommendations(
        recommendations_by_user, test_liked_by_user, evaluable_users, top_n
    )


def evaluate_user_based_cf(
    train_ratings: pd.DataFrame,
    test_liked_by_user: dict[int, set[int]],
    top_n: int,
    neighbors: int,
) -> dict[str, float]:
    user_ids = sorted(train_ratings["userId"].unique())
    movie_ids = sorted(train_ratings["movieId"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

    matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
    for row in train_ratings.itertuples(index=False):
        matrix[user_to_idx[row.userId], movie_to_idx[row.movieId]] = row.rating

    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, 0.0)

    if neighbors > 0 and neighbors < similarities.shape[1]:
        filtered = np.zeros_like(similarities)
        top_neighbor_idx = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
        rows = np.arange(similarities.shape[0])[:, None]
        filtered[rows, top_neighbor_idx] = similarities[rows, top_neighbor_idx]
        similarities = filtered

    rated_mask = matrix > 0
    denom = np.abs(similarities) @ rated_mask.astype(np.float32)
    scores = (similarities @ matrix) / np.maximum(denom, 1e-9)
    scores[rated_mask] = -np.inf

    evaluable_users = [
        user_id
        for user_id in user_ids
        if test_liked_by_user.get(user_id)
    ]
    recommendations_by_user: dict[int, list[int]] = {}
    for user_id in evaluable_users:
        user_scores = scores[user_to_idx[user_id]]
        finite = np.isfinite(user_scores)
        if not finite.any():
            recommendations_by_user[user_id] = []
            continue
        candidate_count = min(top_n, int(finite.sum()))
        top_idx = np.argpartition(user_scores, -candidate_count)[-candidate_count:]
        top_idx = top_idx[np.argsort(user_scores[top_idx])[::-1]]
        recommendations_by_user[user_id] = [idx_to_movie[int(idx)] for idx in top_idx]

    return score_recommendations(
        recommendations_by_user, test_liked_by_user, evaluable_users, top_n
    )


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isinf(value):
        return "Infinity"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, tags, links = load_movielens(args.data_dir)
    movie_titles = movies.set_index("movieId")["title"].to_dict()

    train_ratings, test_ratings = temporal_train_test_split(ratings, args.test_ratio)
    train_transactions = build_transactions(train_ratings, args.rating_threshold)
    all_transactions = build_transactions(ratings, args.rating_threshold)
    stats = dataset_stats(ratings, movies, tags, links, all_transactions, args.rating_threshold)

    make_plots(ratings, movies, args.output_dir)

    itemsets, support_lookup, min_support_count = run_apriori(
        train_transactions, args.min_support, args.max_len
    )
    transaction_count = len(train_transactions)
    rules = generate_rules(
        itemsets,
        support_lookup,
        transaction_count,
        args.min_confidence,
        args.min_lift,
    )

    itemsets_export = itemsets_to_frame(itemsets, movie_titles)
    rules_export = rules_to_frame(rules, movie_titles)
    sample_recommendations = recommend_for_user(
        args.sample_user, train_transactions, rules, movie_titles, args.top_n
    )

    itemsets_export.to_csv(args.output_dir / "frequent_itemsets.csv", index=False)
    rules_export.to_csv(args.output_dir / "association_rules.csv", index=False)
    sample_recommendations.to_csv(
        args.output_dir / f"recommendations_user_{args.sample_user}.csv",
        index=False,
    )

    test_liked_by_user = build_test_liked_by_user(test_ratings, args.rating_threshold)
    apriori_metrics = evaluate_apriori(
        train_transactions, rules, movie_titles, test_liked_by_user, args.top_n
    )

    cf_metrics: dict[str, float] | None = None
    if not args.skip_cf:
        cf_metrics = evaluate_user_based_cf(
            train_ratings, test_liked_by_user, args.top_n, args.cf_neighbors
        )

    itemsets_by_length = (
        itemsets_export.groupby("length").size().astype(int).to_dict()
        if not itemsets_export.empty
        else {}
    )
    runtime_seconds = time.perf_counter() - start
    summary = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "parameters": {
            "rating_threshold": args.rating_threshold,
            "min_support": args.min_support,
            "min_support_count": min_support_count,
            "min_confidence": args.min_confidence,
            "min_lift": args.min_lift,
            "max_len": args.max_len,
            "top_n": args.top_n,
            "test_ratio": args.test_ratio,
            "cf_neighbors": None if args.skip_cf else args.cf_neighbors,
        },
        "dataset": stats,
        "train": {
            "ratings": int(len(train_ratings)),
            "transactions": int(len(train_transactions)),
        },
        "test": {
            "ratings": int(len(test_ratings)),
            "users_with_liked_movies": int(len(test_liked_by_user)),
        },
        "apriori": {
            "frequent_itemsets": int(len(itemsets_export)),
            "frequent_itemsets_by_length": itemsets_by_length,
            "association_rules": int(len(rules_export)),
            "metrics": apriori_metrics,
        },
        "user_based_cf": None if cf_metrics is None else {"metrics": cf_metrics},
        "sample_user": {
            "userId": args.sample_user,
            "recommendation_file": f"recommendations_user_{args.sample_user}.csv",
            "recommendation_count": int(len(sample_recommendations)),
        },
        "runtime_seconds": runtime_seconds,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=json_default)

    print("Movie Apriori recommender finished")
    print(f"Dataset: {args.data_dir}")
    print(f"Transactions used for Apriori: {transaction_count}")
    print(f"Frequent itemsets: {len(itemsets_export)}")
    print(f"Association rules: {len(rules_export)}")
    print(
        "Apriori Precision@{n}: {precision:.4f}, Recall@{n}: {recall:.4f}, "
        "Coverage: {coverage:.4f}".format(
            n=args.top_n,
            precision=apriori_metrics["precision_at_n"],
            recall=apriori_metrics["recall_at_n"],
            coverage=apriori_metrics["coverage"],
        )
    )
    if cf_metrics is not None:
        print(
            "User-CF Precision@{n}: {precision:.4f}, Recall@{n}: {recall:.4f}, "
            "Coverage: {coverage:.4f}".format(
                n=args.top_n,
                precision=cf_metrics["precision_at_n"],
                recall=cf_metrics["recall_at_n"],
                coverage=cf_metrics["coverage"],
            )
        )
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
