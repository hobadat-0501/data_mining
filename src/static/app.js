const fmt = new Intl.NumberFormat("vi-VN");
const pct = new Intl.NumberFormat("vi-VN", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const state = {
  summary: null,
  customLiked: new Map(),
  searchTimer: null,
  searchRequestId: 0,
};

function text(id, value) {
  document.getElementById(id).textContent = value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function setStatus(message, kind = "") {
  const el = document.getElementById("appStatus");
  el.textContent = message;
  el.className = `status ${kind}`.trim();
}

function renderSummary(summary) {
  state.summary = summary;
  const dataset = summary.dataset || {};
  const apriori = summary.apriori || {};
  const metrics = apriori.metrics || {};

  text("metricRatings", fmt.format(dataset.ratings || 0));
  text("metricUsers", fmt.format(dataset.users || 0));
  text("metricMovies", fmt.format(dataset.movies || 0));
  text("metricRules", fmt.format(apriori.association_rules || 0));
  text("metricPrecision", pct.format(metrics.precision_at_n || 0));
}

function renderUser(userData) {
  const profile = userData.profile;
  text("userRatings", fmt.format(profile.ratings_count || 0));
  text("userLiked", fmt.format(profile.liked_count || 0));
  text("userAverage", Number(profile.avg_rating || 0).toFixed(2));
  text("userLatest", profile.last_rating || "-");

  const rows = userData.liked_movies.map((movie) => `
    <tr>
      <td>${escapeHtml(movie.title)}</td>
      <td>${escapeHtml(movie.genres)}</td>
      <td><strong>${Number(movie.rating).toFixed(1)}</strong></td>
    </tr>
  `);
  document.getElementById("likedMovies").innerHTML = rows.join("") || `
    <tr><td colspan="3">User này chưa có phim rating từ 4 sao trở lên.</td></tr>
  `;
}

function renderRecommendations(data) {
  const label = data.user_label || `User ${data.userId}`;
  text(
    "recommendationMeta",
    `${label}: ${fmt.format(data.triggered_rules)} luật được kích hoạt, ${fmt.format(data.liked_count)} phim đã thích.`
  );

  const cards = data.recommendations.map((item, index) => `
    <article class="rec-card">
      <div>
        <h3>${index + 1}. ${escapeHtml(item.title)}</h3>
        <p>Do user đã thích: ${escapeHtml(item.reason)}</p>
        <div class="badges">
          <span class="badge">confidence ${pct.format(item.confidence)}</span>
          <span class="badge">lift ${Number(item.lift).toFixed(2)}</span>
          <span class="badge">support ${pct.format(item.support)}</span>
        </div>
      </div>
      <div class="score">${Number(item.score).toFixed(2)}</div>
    </article>
  `);
  document.getElementById("recommendations").innerHTML = cards.join("") || `
    <div class="empty">Không tìm thấy đề xuất phù hợp cho user này với các luật hiện tại.</div>
  `;
}

function renderRules(rules) {
  const html = rules.map((rule) => `
    <article class="rule-card">
      <strong>${escapeHtml(rule.antecedents)} => ${escapeHtml(rule.consequents)}</strong>
      <span>
        support ${pct.format(rule.support)} · confidence ${pct.format(rule.confidence)}
        · lift ${Number(rule.lift).toFixed(2)} · score ${Number(rule.score).toFixed(2)}
      </span>
    </article>
  `);
  document.getElementById("rulesList").innerHTML = html.join("") || `
    <div class="empty">Chưa có luật kết hợp để hiển thị.</div>
  `;
}

function renderMovieSearchResults(movies) {
  const html = movies.map((movie) => {
    const selected = state.customLiked.has(Number(movie.movieId));
    return `
      <article class="movie-result">
        <div>
          <strong>${escapeHtml(movie.title)}</strong>
          <span>${escapeHtml(movie.genres)}</span>
        </div>
        <button
          type="button"
          class="small-button"
          data-add-movie="${movie.movieId}"
          ${selected ? "disabled" : ""}
        >
          ${selected ? "Đã thêm" : "Thêm"}
        </button>
      </article>
    `;
  });
  document.getElementById("movieSearchResults").innerHTML = html.join("") || `
    <div class="empty">Không tìm thấy phim phù hợp.</div>
  `;
}

function renderCustomLiked() {
  const movies = [...state.customLiked.values()];
  text("customLikedCount", `${fmt.format(movies.length)} phim`);
  const html = movies.map((movie) => `
    <article class="selected-movie">
      <div>
        <strong>${escapeHtml(movie.title)}</strong>
        <span>${escapeHtml(movie.genres)}</span>
      </div>
      <button type="button" class="icon-button" data-remove-movie="${movie.movieId}" aria-label="Xóa phim">x</button>
    </article>
  `);
  document.getElementById("customLikedMovies").innerHTML = html.join("") || `
    <div class="empty">Chưa chọn phim nào.</div>
  `;
}

async function searchMovies(showEmptyError = true) {
  const query = document.getElementById("movieSearchInput").value.trim();
  if (!query) {
    document.getElementById("movieSearchResults").innerHTML = "";
    if (showEmptyError) {
      setStatus("Nhập tên phim để tìm", "error");
    }
    return;
  }

  const requestId = ++state.searchRequestId;
  setStatus("Đang tìm phim...");
  try {
    const movies = await fetchJson(`/api/movies?q=${encodeURIComponent(query)}&limit=12`);
    if (requestId !== state.searchRequestId) {
      return;
    }
    renderMovieSearchResults(movies);
    setStatus("Sẵn sàng", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function addCustomMovie(movieId, title, genres) {
  state.customLiked.set(Number(movieId), {
    movieId: Number(movieId),
    title,
    genres,
  });
  renderCustomLiked();
}

function removeCustomMovie(movieId) {
  state.customLiked.delete(Number(movieId));
  renderCustomLiked();
  const query = document.getElementById("movieSearchInput").value.trim();
  if (query) {
    searchMovies(false);
  }
}

function scheduleMovieSearch() {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(() => {
    searchMovies(false);
  }, 250);
}

async function loadCustomRecommendations() {
  const likedMovieIds = [...state.customLiked.keys()];
  if (likedMovieIds.length === 0) {
    setStatus("Chọn ít nhất 1 phim", "error");
    return;
  }

  const payload = {
    name: document.getElementById("customName").value.trim(),
    liked_movie_ids: likedMovieIds,
    top_n: Number(document.getElementById("customTopNSelect").value),
  };

  setStatus("Đang kiểm tra user mới...");
  try {
    const recs = await postJson("/api/custom-recommendations", payload);
    renderRecommendations(recs);
    text("userRatings", fmt.format(likedMovieIds.length));
    text("userLiked", fmt.format(likedMovieIds.length));
    text("userAverage", "5.00");
    text("userLatest", "User mới");
    document.getElementById("likedMovies").innerHTML = [...state.customLiked.values()].map((movie) => `
      <tr>
        <td>${escapeHtml(movie.title)}</td>
        <td>${escapeHtml(movie.genres)}</td>
        <td><strong>5.0</strong></td>
      </tr>
    `).join("");
    setStatus("Sẵn sàng", "ready");
    document.getElementById("recommend").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadRecommendations() {
  const userId = Number(document.getElementById("userInput").value);
  const topN = Number(document.getElementById("topNSelect").value);
  if (!userId) {
    setStatus("User ID không hợp lệ", "error");
    return;
  }

  setStatus("Đang tính gợi ý...");
  try {
    const [userData, recs] = await Promise.all([
      fetchJson(`/api/user/${userId}`),
      fetchJson(`/api/recommendations/${userId}?top_n=${topN}`),
    ]);
    renderUser(userData);
    renderRecommendations(recs);
    setStatus("Sẵn sàng", "ready");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function init() {
  try {
    const [summary, rules] = await Promise.all([
      fetchJson("/api/summary"),
      fetchJson("/api/rules?limit=25"),
    ]);
    renderSummary(summary);
    renderRules(rules);
    renderCustomLiked();
    await loadRecommendations();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

document.getElementById("recommendButton").addEventListener("click", loadRecommendations);
document.getElementById("userInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadRecommendations();
  }
});
document.getElementById("movieSearchButton").addEventListener("click", searchMovies);
document.getElementById("movieSearchInput").addEventListener("input", scheduleMovieSearch);
document.getElementById("movieSearchInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    searchMovies();
  }
});
document.getElementById("movieSearchResults").addEventListener("click", (event) => {
  const button = event.target.closest("[data-add-movie]");
  if (!button) {
    return;
  }
  const card = button.closest(".movie-result");
  const title = card.querySelector("strong").textContent;
  const genres = card.querySelector("span").textContent;
  addCustomMovie(button.dataset.addMovie, title, genres);
  button.textContent = "Đã thêm";
  button.disabled = true;
});
document.getElementById("customLikedMovies").addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove-movie]");
  if (button) {
    removeCustomMovie(button.dataset.removeMovie);
  }
});
document.getElementById("customRecommendButton").addEventListener("click", loadCustomRecommendations);

init();
