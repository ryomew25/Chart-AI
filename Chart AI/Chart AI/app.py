from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)


# ==================================================
# データを整理
# ==================================================

def normalize_data(data):

    if data is None:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    return data


# ==================================================
# RSI
# ==================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(
        window=period,
        min_periods=period
    ).mean()

    average_loss = loss.rolling(
        window=period,
        min_periods=period
    ).mean()

    rs = (
        average_gain
        /
        average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 /
        (1 + rs)
    )

    return rsi


# ==================================================
# MACD
# ==================================================

def calculate_macd(close):

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


# ==================================================
# 数値を安全に変換
# ==================================================

def safe_float(value):

    try:

        number = float(value)

        if np.isnan(number):
            return None

        return number

    except (TypeError, ValueError):

        return None


# ==================================================
# Yahoo Finance 検索
# ==================================================

def search_stocks(keyword):

    keyword = keyword.strip()

    if not keyword:
        return []

    try:

        result = yf.Search(
            keyword,
            max_results=15,
            news_count=0,
            lists_count=0,
            enable_fuzzy_query=True,
            timeout=10
        )

        stocks = []

        for item in result.quotes:

            symbol = item.get(
                "symbol",
                ""
            )

            name = (
                item.get("longname")
                or item.get("shortname")
                or symbol
            )

            exchange = item.get(
                "exchange",
                ""
            )

            quote_type = item.get(
                "quoteType",
                ""
            )

            if quote_type not in {
                "EQUITY",
                "ETF",
                "MUTUALFUND"
            }:
                continue

            stocks.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "type": quote_type
            })

        return stocks

    except Exception as exc:

        print(
            "========== SEARCH ERROR =========="
        )

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print(
            "=================================="
        )

        return []


# ==================================================
# 株価データ取得
# ==================================================

def get_stock_data(
    symbol,
    period="1y"
):

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    data = normalize_data(data)

    if "Close" not in data.columns:
        return None

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required_columns:

        if column not in data.columns:
            return None

    for column in required_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    if data.empty:
        return None

    return data


# ==================================================
# テクニカル分析
# ==================================================

def analyze_direction(data):

    data = data.copy()

    close = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data["MA20"] = close.rolling(
        20,
        min_periods=20
    ).mean()

    data["MA50"] = close.rolling(
        50,
        min_periods=50
    ).mean()

    data["MA200"] = close.rolling(
        200,
        min_periods=200
    ).mean()

    data["RSI"] = calculate_rsi(
        close
    )

    (
        data["MACD"],
        data["SIGNAL"],
        data["HISTOGRAM"]
    ) = calculate_macd(
        close
    )

    latest = data.iloc[-1]

    price = safe_float(
        latest["Close"]
    )

    ma20 = safe_float(
        latest["MA20"]
    )

    ma50 = safe_float(
        latest["MA50"]
    )

    ma200 = safe_float(
        latest["MA200"]
    )

    rsi = safe_float(
        latest["RSI"]
    )

    macd = safe_float(
        latest["MACD"]
    )

    signal = safe_float(
        latest["SIGNAL"]
    )

    support_data = (
        data["Close"]
        .dropna()
        .tail(30)
    )

    support = (
        safe_float(support_data.min())
        if not support_data.empty
        else None
    )

    resistance = (
        safe_float(support_data.max())
        if not support_data.empty
        else None
    )

    if price is None:

        raise ValueError(
            "現在価格を取得できませんでした。"
        )

    # ------------------------------------------
    # スコア
    # ------------------------------------------

    score = 50

    up_reasons = []

    down_reasons = []

    # 価格 vs MA20
    if ma20 is not None:

        if price > ma20:

            score += 15

            up_reasons.append(
                "現在価格がMA20を上回っています。"
            )

        else:

            score -= 15

            down_reasons.append(
                "現在価格がMA20を下回っています。"
            )

    # MA20 vs MA50
    if (
        ma20 is not None
        and ma50 is not None
    ):

        if ma20 > ma50:

            score += 15

            up_reasons.append(
                "MA20がMA50を上回っています。"
            )

        else:

            score -= 15

            down_reasons.append(
                "MA20がMA50を下回っています。"
            )

    # 価格 vs MA200
    if ma200 is not None:

        if price > ma200:

            score += 10

            up_reasons.append(
                "現在価格がMA200を上回っています。"
            )

        else:

            score -= 10

            down_reasons.append(
                "現在価格がMA200を下回っています。"
            )

    # RSI
    if rsi is not None:

        if 50 <= rsi < 70:

            score += 10

            up_reasons.append(
                f"RSIは{rsi:.1f}で比較的強い範囲です。"
            )

        elif rsi >= 70:

            score += 2

            up_reasons.append(
                f"RSIは{rsi:.1f}で高い水準です。過熱には注意が必要です。"
            )

        elif 30 < rsi < 50:

            score -= 5

            down_reasons.append(
                f"RSIは{rsi:.1f}でやや弱い範囲です。"
            )

        else:

            score -= 2

            down_reasons.append(
                f"RSIは{rsi:.1f}で低い水準です。"
            )

    # MACD
    if (
        macd is not None
        and signal is not None
    ):

        if macd > signal:

            score += 10

            up_reasons.append(
                "MACDがシグナルを上回っています。"
            )

        else:

            score -= 10

            down_reasons.append(
                "MACDがシグナルを下回っています。"
            )

    score = max(
        0,
        min(100, score)
    )

    if score >= 60:

        direction = "上がる予想"
        direction_class = "up"

    elif score <= 40:

        direction = "下がる予想"
        direction_class = "down"

    else:

        direction = "どちらとも言えない"
        direction_class = "neutral"

    return {
        "price": price,
        "score": score,
        "direction": direction,
        "direction_class": direction_class,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi,
        "macd": macd,
        "signal": signal,
        "support": support,
        "resistance": resistance,
        "up_reasons": up_reasons,
        "down_reasons": down_reasons
    }


# ==================================================
# チャート用データ作成
# ==================================================

def make_chart_data(data):

    data = data.copy()

    close = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data["MA20"] = close.rolling(
        20,
        min_periods=20
    ).mean()

    data["MA50"] = close.rolling(
        50,
        min_periods=50
    ).mean()

    data["MA200"] = close.rolling(
        200,
        min_periods=200
    ).mean()

    data["RSI"] = calculate_rsi(
        close
    )

    (
        data["MACD"],
        data["SIGNAL"],
        data["HISTOGRAM"]
    ) = calculate_macd(
        close
    )

    rows = []

    for index, row in data.iterrows():

        rows.append({

            "date":
                index.strftime(
                    "%Y-%m-%d"
                ),

            "open":
                safe_float(
                    row["Open"]
                ),

            "high":
                safe_float(
                    row["High"]
                ),

            "low":
                safe_float(
                    row["Low"]
                ),

            "close":
                safe_float(
                    row["Close"]
                ),

            "volume":
                safe_float(
                    row["Volume"]
                ),

            "ma20":
                safe_float(
                    row["MA20"]
                ),

            "ma50":
                safe_float(
                    row["MA50"]
                ),

            "ma200":
                safe_float(
                    row["MA200"]
                ),

            "rsi":
                safe_float(
                    row["RSI"]
                ),

            "macd":
                safe_float(
                    row["MACD"]
                ),

            "signal":
                safe_float(
                    row["SIGNAL"]
                ),

            "histogram":
                safe_float(
                    row["HISTOGRAM"]
                )
        })

    return rows


# ==================================================
# ページ
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==================================================
# 検索API
# ==================================================

@app.get("/api/search")
def api_search():

    keyword = request.args.get(
        "q",
        ""
    ).strip()

    return jsonify(
        search_stocks(keyword)
    )


# ==================================================
# 分析API
# ==================================================

@app.get("/api/analyze")
def api_analyze():

    symbol = request.args.get(
        "symbol",
        ""
    ).strip()

    period = request.args.get(
        "period",
        "1y"
    ).strip()

    allowed_periods = {
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "5y"
    }

    if period not in allowed_periods:

        period = "1y"

    if not symbol:

        return jsonify({
            "error":
                "銘柄コードがありません。"
        }), 400

    try:

        data = get_stock_data(
            symbol,
            period
        )

        if data is None:

            return jsonify({
                "error":
                    "株価データを取得できませんでした。"
            }), 404

        analysis = analyze_direction(
            data
        )

        chart = make_chart_data(
            data
        )

        return jsonify({

            "symbol":
                symbol,

            "analysis":
                analysis,

            "chart":
                chart

        })

    except Exception as exc:

        print(
            "========== API ERROR =========="
        )

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print(
            "================================"
        )

        return jsonify({

            "error":
                f"{type(exc).__name__}: {exc}"

        }), 500


# ==================================================
# 起動
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )