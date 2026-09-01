from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

PERIODS = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "max": "max",
}


def safe_float(value):
    try:
        value = float(value)
        return None if np.isnan(value) or np.isinf(value) else value
    except (TypeError, ValueError):
        return None


def normalize_data(data):
    if data is None or data.empty:
        return None

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance can return (Close, TICKER), etc.
        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            return None
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    return data


def get_stock_data(symbol, period="1y"):
    period = period if period in PERIODS else "1y"

    data = yf.download(
        symbol,
        period=PERIODS[period],
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_data(data)


# ==========================================================
# Technical indicators
# ==========================================================

def add_indicators(data):
    df = data.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands(20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Stochastic(14)
    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    denom = (highest14 - lowest14).replace(0, np.nan)
    df["StochK"] = 100 * (close - lowest14) / denom
    df["StochD"] = df["StochK"].rolling(3).mean()

    # ROC(12)
    df["ROC"] = close.pct_change(12) * 100

    # Volatility / standard deviation
    df["StdDev20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # Volume moving average
    df["VolumeMA20"] = volume.rolling(20).mean()
    df["VolumeRatio"] = volume / df["VolumeMA20"].replace(0, np.nan)

    return df


# ==========================================================
# DTW
# ==========================================================

def dtw_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n, m = len(a), len(b)
    matrix = np.full((n + 1, m + 1), np.inf)
    matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1],
            )

    return float(matrix[n, m])


def find_similar_pattern(data):
    close = data["Close"].dropna()
    window = 30

    if len(close) < 100:
        return {
            "distance": None,
            "direction": "データ不足",
            "detail": "比較できる過去データが不足しています。",
        }

    current = close.iloc[-window:].values
    current = current / current[0] - 1

    best_distance = np.inf
    best_direction = "不明"

    for end in range(window, len(close) - 10):
        pattern = close.iloc[end - window:end].values
        pattern = pattern / pattern[0] - 1

        distance = dtw_distance(current, pattern)

        if distance < best_distance:
            best_distance = distance
            future_return = close.iloc[end + 10] / close.iloc[end] - 1

            if future_return > 0.02:
                best_direction = "過去類似例は上昇"
            elif future_return < -0.02:
                best_direction = "過去類似例は下降"
            else:
                best_direction = "過去類似例は横ばい"

    return {
        "distance": safe_float(best_distance),
        "direction": best_direction,
        "detail": "直近30営業日の値動きと似た過去パターンを探索しています。",
    }


# ==========================================================
# XGBoost + Walk-Forward
# ==========================================================

FEATURES = [
    "return_1",
    "return_5",
    "return_20",
    "ma20_ratio",
    "ma50_ratio",
    "ma200_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "atr_ratio",
    "stoch_k",
    "stoch_d",
    "roc",
    "volatility",
    "volume_ratio",
]


def create_ml_features(data):
    df = add_indicators(data)

    close = df["Close"]

    df["return_1"] = close.pct_change()
    df["return_5"] = close.pct_change(5)
    df["return_20"] = close.pct_change(20)

    df["ma20_ratio"] = close / df["MA20"] - 1
    df["ma50_ratio"] = close / df["MA50"] - 1
    df["ma200_ratio"] = close / df["MA200"] - 1

    df["rsi"] = df["RSI"] / 100
    df["macd"] = df["MACD"]
    df["macd_signal"] = df["MACD_signal"]
    df["macd_hist"] = df["MACD_hist"]

    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["bb_position"] = (close - df["BB_lower"]) / bb_range

    df["atr_ratio"] = df["ATR"] / close.replace(0, np.nan)
    df["stoch_k"] = df["StochK"] / 100
    df["stoch_d"] = df["StochD"] / 100
    df["roc"] = df["ROC"] / 100
    df["volatility"] = df["StdDev20"]
    df["volume_ratio"] = df["VolumeRatio"]

    return df


def make_model():
    return XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )


def xgboost_prediction(data):
    df = create_ml_features(data)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    train = df[FEATURES + ["target"]].dropna()

    if len(train) < 140:
        return {
            "prediction": "データ不足",
            "probability": None,
            "test_accuracy": None,
        }

    split = int(len(train) * 0.8)
    train_part = train.iloc[:split]
    test_part = train.iloc[split:]

    model = make_model()
    model.fit(train_part[FEATURES], train_part["target"])

    latest = df[FEATURES].dropna().iloc[-1:]
    probability = float(model.predict_proba(latest)[0][1])

    if probability >= 0.55:
        prediction = "上昇シグナル"
    elif probability <= 0.45:
        prediction = "下降シグナル"
    else:
        prediction = "中立"

    accuracy = None
    if len(test_part):
        predictions = model.predict(test_part[FEATURES])
        accuracy = float(
            (predictions == test_part["target"].values).mean()
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "test_accuracy": accuracy,
    }


def walk_forward(data):
    df = create_ml_features(data)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df[FEATURES + ["target"]].dropna()

    if len(df) < 180:
        return {"accuracy": None, "windows": 0}

    results = []
    initial_train = 120
    step = 20

    for end in range(initial_train, len(df), step):
        train = df.iloc[:end]
        test = df.iloc[end:min(end + step, len(df))]

        if len(test) == 0:
            break

        model = make_model()
        model.fit(train[FEATURES], train["target"])
        predictions = model.predict(test[FEATURES])

        results.append(
            float((predictions == test["target"].values).mean())
        )

    return {
        "accuracy": float(np.mean(results)) if results else None,
        "windows": len(results),
    }


# ==========================================================
# Ordinary indicators / directional analysis
# ==========================================================

def indicator_analysis(df):
    last = df.iloc[-1]

    rsi = safe_float(last["RSI"])
    macd = safe_float(last["MACD"])
    macd_signal = safe_float(last["MACD_signal"])
    close = safe_float(last["Close"])

    ma20 = safe_float(last["MA20"])
    ma50 = safe_float(last["MA50"])
    ma200 = safe_float(last["MA200"])

    if rsi is None:
        rsi_label = "データ不足"
    elif rsi >= 70:
        rsi_label = "過熱気味"
    elif rsi <= 30:
        rsi_label = "売られすぎ水準"
    elif rsi >= 50:
        rsi_label = "上昇モメンタム"
    else:
        rsi_label = "弱め"

    if macd is None or macd_signal is None:
        macd_label = "データ不足"
    elif macd > macd_signal:
        macd_label = "上昇"
    else:
        macd_label = "下降"

    ma_votes = 0
    ma_count = 0
    for ma in [ma20, ma50, ma200]:
        if ma is not None:
            ma_count += 1
            ma_votes += 1 if close > ma else -1

    if ma_count == 0:
        trend_label = "データ不足"
    elif ma_votes >= 2:
        trend_label = "上昇トレンド"
    elif ma_votes <= -2:
        trend_label = "下降トレンド"
    else:
        trend_label = "混在"

    return {
        "rsi": rsi,
        "rsi_label": rsi_label,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": safe_float(last["MACD_hist"]),
        "macd_label": macd_label,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "trend_label": trend_label,
        "bollinger": {
            "upper": safe_float(last["BB_upper"]),
            "middle": safe_float(last["BB_mid"]),
            "lower": safe_float(last["BB_lower"]),
        },
        "atr": safe_float(last["ATR"]),
        "stochastic_k": safe_float(last["StochK"]),
        "stochastic_d": safe_float(last["StochD"]),
        "roc": safe_float(last["ROC"]),
        "volatility": safe_float(last["StdDev20"]),
        "volume_ratio": safe_float(last["VolumeRatio"]),
    }


def combined_signal(indicators, xgb, dtw):
    score = 50.0
    reasons = []
    positives = []
    cautions = []

    # XGBoost: strongest component
    probability = xgb.get("probability")
    if probability is not None:
        xgb_component = (probability - 0.5) * 70
        score += xgb_component
        if probability >= 0.55:
            positives.append(f"XGBoostは上昇確率{probability * 100:.1f}%")
        elif probability <= 0.45:
            cautions.append(f"XGBoostは下降確率{(1-probability) * 100:.1f}%")
        else:
            reasons.append("XGBoostは中立圏")

    # RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        if 50 <= rsi < 70:
            score += 6
            positives.append("RSIは上昇モメンタム")
        elif rsi >= 70:
            score -= 3
            cautions.append("RSIは過熱気味")
        elif rsi < 30:
            score += 2
            positives.append("RSIは売られすぎ水準")
        elif rsi < 50:
            score -= 5
            cautions.append("RSIは50を下回る")

    # MACD
    if indicators.get("macd") is not None and indicators.get("macd_signal") is not None:
        if indicators["macd"] > indicators["macd_signal"]:
            score += 7
            positives.append("MACDは上向き")
        else:
            score -= 7
            cautions.append("MACDは下向き")

    # MA trend
    trend = indicators.get("trend_label")
    if trend == "上昇トレンド":
        score += 9
        positives.append("移動平均線から上昇トレンド")
    elif trend == "下降トレンド":
        score -= 9
        cautions.append("移動平均線から下降トレンド")

    # DTW: small weight
    direction = dtw.get("direction")
    if direction == "過去類似例は上昇":
        score += 3
        positives.append("過去の類似パターンは上昇")
    elif direction == "過去類似例は下降":
        score -= 3
        cautions.append("過去の類似パターンは下降")

    score = max(0, min(100, score))

    if score >= 60:
        label = "上昇寄り"
    elif score <= 40:
        label = "下降寄り"
    else:
        label = "中立"

    if positives:
        reasons.extend(positives[:3])
    if cautions:
        reasons.extend(cautions[:2])

    explanation = (
        "。".join(reasons) + "。"
        if reasons
        else "複数の指標から明確な方向性は確認できませんでした。"
    )

    return {
        "label": label,
        "score": round(score, 1),
        "explanation": explanation,
        "positives": positives,
        "cautions": cautions,
    }


# ==========================================================
# Chart + basic information
# ==========================================================

def make_chart_data(df):
    rows = []

    for index, row in df.iterrows():
        rows.append(
            {
                "date": index.strftime("%Y-%m-%d"),
                "open": safe_float(row["Open"]),
                "high": safe_float(row["High"]),
                "low": safe_float(row["Low"]),
                "close": safe_float(row["Close"]),
                "volume": safe_float(row["Volume"]),
                "ma20": safe_float(row["MA20"]),
                "ma50": safe_float(row["MA50"]),
                "ma200": safe_float(row["MA200"]),
                "bb_upper": safe_float(row["BB_upper"]),
                "bb_lower": safe_float(row["BB_lower"]),
                "rsi": safe_float(row["RSI"]),
                "macd": safe_float(row["MACD"]),
                "macd_signal": safe_float(row["MACD_signal"]),
            }
        )

    return rows


def get_basic_info(symbol, data):
    last = data.iloc[-1]
    previous = data.iloc[-2] if len(data) >= 2 else None

    close = safe_float(last["Close"])
    prev_close = safe_float(previous["Close"]) if previous is not None else None

    change = close - prev_close if close is not None and prev_close is not None else None
    change_pct = (
        change / prev_close * 100
        if change is not None and prev_close not in (None, 0)
        else None
    )

    info = {}
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.info or {}
        fields = [
            "longName",
            "shortName",
            "exchange",
            "currency",
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "trailingEps",
            "dividendYield",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
        ]
        for field in fields:
            info[field] = raw.get(field)
    except Exception:
        info = {}

    return {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": safe_float(info.get("marketCap")),
        "per": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "pbr": safe_float(info.get("priceToBook")),
        "eps": safe_float(info.get("trailingEps")),
        "dividend_yield": safe_float(info.get("dividendYield")),
        "week52_high": safe_float(info.get("fiftyTwoWeekHigh")),
        "week52_low": safe_float(info.get("fiftyTwoWeekLow")),
        "price": close,
        "change": safe_float(change),
        "change_pct": safe_float(change_pct),
        "day_high": safe_float(last["High"]),
        "day_low": safe_float(last["Low"]),
        "open": safe_float(last["Open"]),
        "volume": safe_float(last["Volume"]),
    }


def complete_analysis(symbol, data):
    df = add_indicators(data)

    indicators = indicator_analysis(df)
    xgb = xgboost_prediction(data)
    dtw = find_similar_pattern(data)
    walk = walk_forward(data)
    combined = combined_signal(indicators, xgb, dtw)

    return {
        "combined": combined,
        "indicators": indicators,
        "xgboost": xgb,
        "dtw": dtw,
        "walk_forward": walk,
    }


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/search")
def search_api():
    keyword = request.args.get("q", "").strip()

    if not keyword:
        return jsonify([])

    try:
        result = yf.Search(
            keyword,
            max_results=15,
            news_count=0,
            lists_count=0,
            enable_fuzzy_query=True,
            timeout=10,
        )

        stocks = []
        for item in result.quotes:
            quote_type = item.get("quoteType", "")
            if quote_type not in {"EQUITY", "ETF", "MUTUALFUND"}:
                continue

            stocks.append(
                {
                    "symbol": item.get("symbol", ""),
                    "name": (
                        item.get("longname")
                        or item.get("shortname")
                        or item.get("symbol")
                    ),
                    "exchange": item.get("exchange", ""),
                }
            )

        return jsonify(stocks)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/analyze")
def analyze_api():
    symbol = request.args.get("symbol", "").strip()
    period = request.args.get("period", "1y")

    if not symbol:
        return jsonify({"error": "銘柄が指定されていません。"}), 400

    try:
        data = get_stock_data(symbol, period)

        if data is None or data.empty:
            return jsonify({"error": "株価データを取得できませんでした。"}), 404

        analysis = complete_analysis(symbol, data)
        chart = make_chart_data(add_indicators(data))
        basic = get_basic_info(symbol, data)

        return jsonify(
            {
                "symbol": symbol,
                "basic": basic,
                "analysis": analysis,
                "chart": chart,
            }
        )

    except Exception as exc:
        print("========== ANALYSIS ERROR ==========")
        print(type(exc).__name__)
        print(str(exc))
        print("====================================")
        return jsonify(
            {"error": f"{type(exc).__name__}: {exc}"}
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
