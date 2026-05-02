# Dự án đề xuất phim cá nhân hóa bằng Apriori

Dự án này hiện thực nội dung trong báo cáo `KPDL_Nhom09_DeXuatPhim_Baocao.pdf`: dùng khai phá luật kết hợp Apriori để đề xuất phim cá nhân hóa.

Dataset bạn tải trên Kaggle là MovieLens 100K, nhưng gói dữ liệu trong ảnh và trong thư mục local đang có cấu trúc `ml-latest-small`. Bộ này không dùng các file `u.data`, `u.item` như phần mẫu trong báo cáo, nên code đã được điều chỉnh để đọc trực tiếp các file:

- `ml-latest-small/ratings.csv`
- `ml-latest-small/movies.csv`
- `ml-latest-small/tags.csv`

Dataset không được commit lên GitHub. Sau khi clone code, hãy tải dataset từ Kaggle và đặt thư mục `ml-latest-small` ở thư mục gốc dự án.
- `ml-latest-small/links.csv`

## Ý tưởng xử lý

Mỗi người dùng là một transaction. Các phim có `rating >= 4.0` được xem là phim người dùng thích.

Pipeline:

1. Load dữ liệu và thống kê EDA.
2. Chia train/test theo thời gian đánh giá của từng user.
3. Tạo transaction từ tập train.
4. Chạy Apriori để tìm frequent itemsets.
5. Sinh association rules theo `confidence` và `lift`.
6. Gợi ý Top-N phim cho user dựa trên luật `A => B`.
7. Đánh giá bằng `Precision@N`, `Recall@N`, `HitRate@N`, `Coverage`.
8. So sánh thêm với baseline User-based Collaborative Filtering.

## Cài thư viện

```powershell
python -m pip install -r requirements.txt
```

## Chạy dự án

```powershell
python .\src\movie_apriori_recommender.py --data-dir .\ml-latest-small --output-dir .\outputs --sample-user 1
```

Tham số mặc định:

- `--rating-threshold 4.0`
- `--min-support 0.05`
- `--min-confidence 0.5`
- `--min-lift 1.0`
- `--max-len 4`
- `--top-n 10`
- `--test-ratio 0.2`

Nếu muốn chạy nhanh hơn và bỏ baseline Collaborative Filtering:

```powershell
python .\src\movie_apriori_recommender.py --skip-cf
```

## Chạy giao diện người dùng

Sau khi đã sinh các file trong `outputs`, chạy web UI:

```powershell
python -m uvicorn src.ui_app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

Giao diện cho phép:

- Chọn `userId` và số lượng phim cần gợi ý.
- Nhập user mới bằng cách tìm phim và thêm các phim user thích.
- Xem profile rating của user.
- Xem các phim user đã thích (`rating >= 4.0`).
- Xem Top-N phim đề xuất, điểm score, confidence, lift và lý do gợi ý.
- Xem thống kê dataset, biểu đồ EDA và các luật kết hợp nổi bật.

Quy trình kiểm thử user mới:

1. Mở mục `User mới`.
2. Nhập tên user nếu cần.
3. Gõ tên phim, ví dụ `matrix`, `star wars`, `toy story`; hệ thống sẽ tự hiện các gợi ý gần đúng.
4. Bấm `Thêm` với các phim user thích.
5. Chọn `Top-N`.
6. Bấm `Kiểm tra hệ thống`.

## Kết quả sinh ra

Sau khi chạy, thư mục `outputs` sẽ có:

- `frequent_itemsets.csv`: các tập phim phổ biến được Apriori tìm thấy.
- `association_rules.csv`: luật kết hợp dạng phim đã thích => phim nên gợi ý.
- `recommendations_user_1.csv`: Top-N phim gợi ý cho user mẫu.
- `summary.json`: tham số, thống kê dataset và kết quả đánh giá.
- `rating_distribution.png`: biểu đồ phân phối rating.
- `top_genres.png`: biểu đồ top thể loại phim.

## Ghi chú

Code tự hiện thực Apriori, không cần `mlxtend`. Baseline User-based CF dùng `scikit-learn`.

## Tài liệu báo cáo code

Xem file `CODE_SUMMARY_REPORT.md` để lấy phần tóm tắt toàn bộ mã nguồn, pipeline, thuật toán, API và giao diện cho báo cáo.
