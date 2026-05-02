# Tóm tắt mã nguồn dự án đề xuất phim bằng Apriori

## 1. Mục tiêu dự án

Dự án xây dựng hệ thống đề xuất phim cá nhân hóa dựa trên thuật toán Apriori và luật kết hợp. Hệ thống sử dụng dataset MovieLens `ml-latest-small`, trong đó mỗi người dùng được xem là một transaction, còn các phim được người dùng đánh giá từ 4 sao trở lên được xem là các item yêu thích.

Từ dữ liệu rating, chương trình khai phá các tập phim thường được yêu thích cùng nhau, sinh luật kết hợp dạng:

```text
Nếu user thích phim A, B => có khả năng user cũng thích phim C
```

Sau đó hệ thống dùng các luật này để gợi ý Top-N phim cho user có sẵn trong dataset hoặc user mới do người dùng nhập thủ công trên giao diện.

## 2. Cấu trúc mã nguồn

```text
.
├── src/
│   ├── movie_apriori_recommender.py
│   ├── ui_app.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── README.md
├── PROJECT_RESULTS.md
├── CODE_SUMMARY_REPORT.md
├── requirements.txt
└── .gitignore
```

Ý nghĩa các file chính:

- `src/movie_apriori_recommender.py`: xử lý dữ liệu, chạy Apriori, sinh luật kết hợp, đánh giá mô hình và xuất kết quả.
- `src/ui_app.py`: backend FastAPI phục vụ giao diện web và các API đề xuất phim.
- `src/static/index.html`: giao diện người dùng.
- `src/static/styles.css`: định dạng giao diện.
- `src/static/app.js`: xử lý tương tác frontend, gọi API, tìm phim, chọn phim và hiển thị kết quả.
- `README.md`: hướng dẫn cài đặt, chạy pipeline và chạy giao diện.
- `PROJECT_RESULTS.md`: tóm tắt kết quả thực nghiệm.
- `requirements.txt`: danh sách thư viện cần cài.
- `.gitignore`: bỏ qua dataset, output, file PDF và cache khi đẩy code lên GitHub.

## 3. Dataset sử dụng

Dataset local nằm trong thư mục:

```text
ml-latest-small/
```

Gồm các file:

- `ratings.csv`: thông tin user đánh giá phim.
- `movies.csv`: thông tin phim và thể loại.
- `tags.csv`: tag do user gắn cho phim.
- `links.csv`: liên kết movieId với IMDB/TMDB.

Trong code, hai file quan trọng nhất là:

- `ratings.csv`: dùng để biết user nào thích phim nào.
- `movies.csv`: dùng để lấy tên phim và thể loại.

Điều kiện xác định một phim là user yêu thích:

```text
rating >= 4.0
```

## 4. Pipeline xử lý trong `movie_apriori_recommender.py`

File này là phần lõi của dự án. Chương trình thực hiện các bước chính sau.

### 4.1. Đọc tham số dòng lệnh

Hàm:

```python
parse_args()
```

Cho phép cấu hình các tham số:

- `--data-dir`: thư mục dataset.
- `--output-dir`: thư mục lưu kết quả.
- `--rating-threshold`: ngưỡng rating để xem là yêu thích.
- `--min-support`: support tối thiểu.
- `--min-confidence`: confidence tối thiểu.
- `--min-lift`: lift tối thiểu.
- `--max-len`: độ dài tối đa của itemset.
- `--top-n`: số phim đề xuất.
- `--test-ratio`: tỷ lệ chia test.

### 4.2. Load dữ liệu

Hàm:

```python
load_movielens(data_dir)
```

Đọc 4 file CSV:

```text
ratings.csv, movies.csv, tags.csv, links.csv
```

Nếu thiếu file nào, chương trình báo lỗi để người dùng kiểm tra lại dataset.

### 4.3. Chia train/test theo thời gian

Hàm:

```python
temporal_train_test_split(ratings, test_ratio)
```

Dữ liệu rating của từng user được sắp xếp theo `timestamp`. Các rating cũ hơn đưa vào train, các rating mới hơn đưa vào test.

Cách chia này hợp lý hơn chia ngẫu nhiên vì mô phỏng thực tế:

```text
Dùng lịch sử xem phim trước đây để dự đoán phim user thích trong tương lai
```

### 4.4. Xây dựng transaction

Hàm:

```python
build_transactions(ratings, rating_threshold)
```

Mỗi user được chuyển thành một transaction:

```text
User 1 = {movieId 1, movieId 50, movieId 260, ...}
```

Chỉ các phim có rating từ `rating_threshold` trở lên mới được đưa vào transaction.

### 4.5. Thống kê và trực quan hóa dữ liệu

Các hàm:

```python
dataset_stats(...)
make_plots(...)
```

Sinh ra các thông tin:

- Số ratings.
- Số users.
- Số movies.
- Số tags.
- Số user có phim yêu thích.
- Trung bình số phim yêu thích mỗi user.
- Độ thưa của ma trận user-movie.
- Top thể loại phim.

Đồng thời xuất hai biểu đồ:

```text
outputs/rating_distribution.png
outputs/top_genres.png
```

## 5. Thuật toán Apriori trong code

Code tự cài đặt Apriori, không phụ thuộc `mlxtend`.

### 5.1. Sinh candidate itemsets

Hàm:

```python
generate_candidates(previous_frequent, k)
```

Từ các frequent itemset kích thước `k-1`, chương trình sinh các candidate itemset kích thước `k`.

Ví dụ:

```text
{A, B}, {A, C} => {A, B, C}
```

Sau đó áp dụng tính chất Apriori:

```text
Nếu một itemset là frequent thì mọi tập con của nó cũng phải frequent
```

Nếu candidate có tập con không frequent thì bị loại bỏ sớm.

### 5.2. Đếm support cho candidate

Hàm:

```python
count_candidates(transactions, candidates)
```

Với mỗi transaction, chương trình kiểm tra candidate nào là tập con của transaction đó. Nếu có, tăng bộ đếm support.

### 5.3. Chạy Apriori

Hàm:

```python
run_apriori(transactions_by_user, min_support, max_len)
```

Các bước:

1. Đếm frequent 1-itemset.
2. Sinh candidate 2-itemset, 3-itemset, ...
3. Đếm support.
4. Giữ lại itemset có support đủ lớn.
5. Dừng khi không còn candidate hoặc đạt `max_len`.

Kết quả trả về:

- `frequent_itemsets`: bảng các tập phim phổ biến.
- `support_lookup`: dictionary lưu support count.
- `min_support_count`: số transaction tối thiểu cần đạt.

## 6. Sinh luật kết hợp

Hàm:

```python
generate_rules(...)
```

Từ mỗi frequent itemset có độ dài từ 2 trở lên, chương trình sinh luật:

```text
antecedents => consequents
```

Ví dụ:

```text
Matrix, The (1999) => Star Wars: Episode V (1980)
```

Các chỉ số được tính:

### Support

```text
support(A => B) = số transaction chứa cả A và B / tổng số transaction
```

Support cho biết mức độ phổ biến của luật trong toàn bộ dữ liệu.

### Confidence

```text
confidence(A => B) = support(A ∪ B) / support(A)
```

Confidence cho biết trong số user thích A, có bao nhiêu phần trăm cũng thích B.

### Lift

```text
lift(A => B) = confidence(A => B) / support(B)
```

Lift cho biết mối liên hệ giữa A và B có mạnh hơn ngẫu nhiên hay không.

- `lift > 1`: A và B có tương quan dương.
- `lift = 1`: A và B gần như độc lập.
- `lift < 1`: A và B có tương quan âm.

Trong dự án chỉ giữ các luật có:

```text
confidence >= min_confidence
lift > min_lift
```

## 7. Cơ chế đề xuất phim

Hàm:

```python
recommend_for_user(user_id, transactions_by_user, rules, movie_titles, top_n)
```

Với một user, hệ thống lấy tập phim user đã thích:

```text
liked = {các phim user rating >= 4}
```

Sau đó duyệt qua từng luật:

```text
A => B
```

Nếu:

```text
A là tập con của liked
B chưa nằm trong liked
```

thì phim trong B được đưa vào danh sách đề xuất.

Điểm xếp hạng đề xuất:

```text
score = confidence × lift
```

Ý nghĩa:

- `confidence` cao: nhiều user có sở thích giống vậy cũng thích phim này.
- `lift` cao: mối liên hệ không chỉ do phim phổ biến ngẫu nhiên.
- `score` cao: phim được ưu tiên đề xuất cao hơn.

Lưu ý: `score` không phải rating 5 sao và không có max cố định.

## 8. Đánh giá mô hình

### 8.1. Đánh giá Apriori

Hàm:

```python
evaluate_apriori(...)
```

Chương trình tạo đề xuất cho từng user trong tập test, sau đó so sánh với các phim user thật sự thích trong test.

Các chỉ số:

- `Precision@N`: trong N phim đề xuất, bao nhiêu phim user thật sự thích.
- `Recall@N`: trong các phim user thật sự thích, hệ thống tìm lại được bao nhiêu phim.
- `HitRate@N`: tỷ lệ user có ít nhất một đề xuất đúng.
- `Coverage`: tỷ lệ user nhận được ít nhất một đề xuất.

### 8.2. Baseline User-based Collaborative Filtering

Hàm:

```python
evaluate_user_based_cf(...)
```

Code cũng xây dựng baseline User-based Collaborative Filtering để so sánh.

Cách làm:

1. Tạo ma trận user-movie.
2. Tính cosine similarity giữa các user.
3. Lấy các user giống nhau nhất.
4. Dự đoán phim user có thể thích dựa trên rating của user tương tự.

Baseline này giúp đối chiếu hiệu quả của Apriori.

## 9. Output sinh ra

Sau khi chạy:

```powershell
python .\src\movie_apriori_recommender.py --data-dir .\ml-latest-small --output-dir .\outputs --sample-user 1
```

Chương trình sinh các file:

```text
outputs/frequent_itemsets.csv
outputs/association_rules.csv
outputs/recommendations_user_1.csv
outputs/summary.json
outputs/rating_distribution.png
outputs/top_genres.png
```

Ý nghĩa:

- `frequent_itemsets.csv`: các tập phim thường được thích cùng nhau.
- `association_rules.csv`: các luật kết hợp.
- `recommendations_user_1.csv`: đề xuất mẫu cho user 1.
- `summary.json`: thống kê, tham số và kết quả đánh giá.
- `rating_distribution.png`: biểu đồ phân phối rating.
- `top_genres.png`: biểu đồ top thể loại phim.

## 10. Backend giao diện trong `ui_app.py`

File `ui_app.py` xây dựng backend bằng FastAPI.

Chạy server:

```powershell
python -m uvicorn src.ui_app:app --host 127.0.0.1 --port 8000
```

Sau đó mở:

```text
http://127.0.0.1:8000
```

### 10.1. DataStore

Class:

```python
DataStore
```

Lưu các dữ liệu đã load:

- `ratings`
- `movies`
- `rules`
- `users`
- `liked_by_user`
- `movie_titles`
- `movie_rating_counts`
- `summary`

Hàm:

```python
load_store()
```

Đọc dữ liệu một lần và cache lại bằng `lru_cache`, giúp API chạy nhanh hơn.

### 10.2. API tổng quan

Endpoint:

```text
GET /api/summary
```

Trả về:

- Thống kê dataset.
- Tham số Apriori.
- Kết quả đánh giá.
- Đường dẫn các file output.

### 10.3. API danh sách user

Endpoint:

```text
GET /api/users
```

Trả về danh sách user có trong dataset, gồm:

- `userId`
- số rating
- số phim liked
- rating trung bình
- ngày rating gần nhất

### 10.4. API tìm phim

Endpoint:

```text
GET /api/movies?q=matrix&limit=12
```

Trả về danh sách phim khớp với từ khóa. Code ưu tiên:

1. Phim có tên bắt đầu bằng từ khóa.
2. Phim có từ trong tiêu đề bắt đầu bằng từ khóa.
3. Phim chứa từ khóa ở vị trí bất kỳ.
4. Nếu cùng mức khớp, phim có nhiều rating hơn được xếp trước.

Nhờ đó khi nhập `mat` hoặc `matrix`, phim phổ biến như `Matrix, The (1999)` được ưu tiên hiển thị.

### 10.5. API thông tin một user

Endpoint:

```text
GET /api/user/{user_id}
```

Trả về:

- profile rating của user.
- danh sách tối đa 50 phim user đã thích.

### 10.6. API đề xuất cho user có sẵn

Endpoint:

```text
GET /api/recommendations/{user_id}?top_n=10
```

Trả về Top-N phim đề xuất cho user có sẵn trong dataset.

### 10.7. API đề xuất cho user mới

Endpoint:

```text
POST /api/custom-recommendations
```

Body mẫu:

```json
{
  "name": "Nguyen Van A",
  "liked_movie_ids": [2571, 1196, 260],
  "top_n": 10
}
```

API này cho phép nhập user mới bằng danh sách phim yêu thích. Hệ thống không cần userId có sẵn trong dataset.

## 11. Giao diện người dùng

Giao diện gồm 4 phần chính.

### 11.1. Tổng quan dataset

Hiển thị nhanh:

- số ratings
- số users
- số movies
- số luật kết hợp
- Precision@10

### 11.2. Đề xuất cho user có sẵn

Người dùng nhập:

- `User ID`
- `Top-N`

Sau đó bấm:

```text
Đề xuất phim
```

Kết quả hiển thị:

- tên phim đề xuất
- lý do đề xuất
- confidence
- lift
- support
- score

### 11.3. Nhập user mới

Người dùng có thể:

1. Nhập tên user.
2. Gõ tên phim vào ô tìm kiếm.
3. Hệ thống tự hiện một số gợi ý phim.
4. Bấm `Thêm` để đưa phim vào danh sách user thích.
5. Chọn `Top-N`.
6. Bấm `Kiểm tra hệ thống`.

Sau đó hệ thống sinh đề xuất dựa trên các phim đã chọn.

### 11.4. Luật kết hợp và biểu đồ

Giao diện hiển thị:

- các luật kết hợp nổi bật, sắp xếp theo `score`.
- biểu đồ phân phối rating.
- biểu đồ top thể loại phim.

## 12. Frontend JavaScript trong `app.js`

File `app.js` đảm nhiệm:

- Gọi API backend bằng `fetch`.
- Render thống kê dataset.
- Render thông tin user.
- Render danh sách đề xuất.
- Tìm phim tự động khi người dùng gõ chữ.
- Quản lý danh sách phim đã chọn cho user mới.
- Gửi request kiểm tra user mới.

Các hàm quan trọng:

- `fetchJson(url)`: gọi API GET.
- `postJson(url, payload)`: gọi API POST.
- `renderSummary(summary)`: hiển thị thống kê.
- `renderUser(userData)`: hiển thị thông tin user.
- `renderRecommendations(data)`: hiển thị phim đề xuất.
- `searchMovies()`: tìm phim theo từ khóa.
- `renderMovieSearchResults(movies)`: hiển thị gợi ý phim.
- `addCustomMovie(...)`: thêm phim vào user mới.
- `loadCustomRecommendations()`: gửi danh sách phim user mới thích lên backend.

## 13. CSS trong `styles.css`

File CSS xây dựng giao diện dạng dashboard:

- Sidebar bên trái.
- Khu vực thống kê.
- Panel chọn user.
- Panel nhập user mới.
- Danh sách phim đề xuất.
- Bảng phim user đã thích.
- Danh sách luật kết hợp.
- Biểu đồ.

CSS cũng có responsive design để giao diện dùng được trên màn hình nhỏ.

## 14. Kết quả thực nghiệm hiện tại

Với cấu hình:

```text
rating_threshold = 4.0
min_support = 0.05
min_confidence = 0.5
min_lift = 1.0
max_len = 4
top_n = 10
```

Kết quả trên dataset local:

- Ratings: 100.836
- Users: 610
- Movies: 9.742
- Users có ít nhất một phim yêu thích: 609
- Frequent itemsets: 9.367
- Association rules: 29.345
- Apriori Precision@10: 6,10%
- Apriori Recall@10: 5,60%
- Apriori Coverage: 96,44%

## 15. Ưu điểm của hệ thống

- Có khả năng giải thích đề xuất rõ ràng bằng luật kết hợp.
- User có thể biết vì sao một phim được đề xuất.
- Có thể đề xuất cho user mới nếu user nhập một số phim yêu thích.
- Không cần huấn luyện mô hình phức tạp.
- Giao diện web dễ kiểm thử trực tiếp.

## 16. Hạn chế

- Apriori có thể chậm nếu dataset rất lớn.
- Chỉ dùng thông tin rating, chưa khai thác sâu nội dung phim.
- Dataset `ml-latest-small` không có thông tin tuổi, giới tính, nghề nghiệp.
- User mới phải chọn ít nhất một vài phim yêu thích thì hệ thống mới có cơ sở đề xuất.
- Score không phải rating tuyệt đối, chỉ dùng để xếp hạng trong hệ thống.

## 17. Hướng phát triển

- Kết hợp Apriori với Content-based Filtering dựa trên thể loại phim.
- Thêm poster phim bằng TMDB API.
- Cho phép user nhập rating 1-5 thay vì chỉ chọn phim thích.
- Lưu hồ sơ user mới vào file hoặc database.
- Thay Apriori bằng FP-Growth để tăng tốc trên dữ liệu lớn.
- Triển khai giao diện lên web server hoặc cloud.

## 18. Kết luận

Dự án đã xây dựng hoàn chỉnh một hệ thống đề xuất phim cá nhân hóa dựa trên Apriori. Phần xử lý dữ liệu khai phá frequent itemsets và association rules từ MovieLens, phần backend cung cấp API đề xuất, còn phần frontend cho phép kiểm thử trực quan với cả user có sẵn và user mới.

Điểm nổi bật của hệ thống là khả năng giải thích đề xuất:

```text
Phim được đề xuất vì user đã thích một nhóm phim cụ thể trước đó
```

Điều này giúp mô hình Apriori phù hợp với mục tiêu của bài toán khai phá dữ liệu: không chỉ đưa ra kết quả, mà còn chỉ ra được mẫu tri thức ẩn trong dữ liệu.
