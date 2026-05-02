# Kết quả thực nghiệm trên `ml-latest-small`

File PDF mô tả cùng bài toán đề xuất phim bằng Apriori, nhưng dùng MovieLens 100K dạng `u.data/u.item`. Lần chạy này dùng dataset local `ml-latest-small`.

## Cấu hình

- Rating yêu thích: `rating >= 4.0`
- Chia train/test: theo thời gian từng user, `test_ratio = 0.2`
- `min_support = 0.05`, tương đương ít nhất 31 transaction trong tập train
- `min_confidence = 0.5`
- `min_lift = 1.0`
- `max_len = 4`
- `top_n = 10`

## Thống kê dữ liệu

- Ratings: 100.836
- Users: 610
- Movies: 9.742
- Tags: 3.683
- Users có ít nhất một phim yêu thích: 609
- Cặp user-phim yêu thích: 48.580
- Trung bình phim yêu thích mỗi user: 79,77
- Sparsity ma trận liked user-movie: 99,18%

## Kết quả Apriori

- Transactions train: 607
- Frequent itemsets: 9.367
- 1-itemsets: 275
- 2-itemsets: 2.047
- 3-itemsets: 4.093
- 4-itemsets: 2.952
- Association rules mạnh: 29.345

Một số phim phổ biến nhất trong frequent 1-itemsets:

- `Shawshank Redemption, The (1994)`: support 39,37%
- `Forrest Gump (1994)`: support 37,40%
- `Pulp Fiction (1994)`: support 35,58%
- `Matrix, The (1999)`: support 34,10%
- `Silence of the Lambs, The (1991)`: support 33,44%

## Đánh giá

Apriori:

- Precision@10: 6,10%
- Recall@10: 5,60%
- HitRate@10: 36,78%
- Coverage: 96,44%

User-based Collaborative Filtering baseline:

- Precision@10: 0,51%
- Recall@10: 0,70%
- HitRate@10: 4,73%
- Coverage: 100%

## File kết quả

- `outputs/frequent_itemsets.csv`
- `outputs/association_rules.csv`
- `outputs/recommendations_user_1.csv`
- `outputs/summary.json`
- `outputs/rating_distribution.png`
- `outputs/top_genres.png`
