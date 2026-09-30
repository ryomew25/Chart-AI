from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time
import traceback
from yfinance import EquityQuery

app = Flask(__name__)

# Small in-process cache to reduce repeated Yahoo requests during local use.
CACHE = {}
CACHE_TTL = 20


def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def normalize_ticker(value):
    value = (value or "").strip().upper()
    # JPX 4-digit codes can be entered without .T
    if value.isdigit() and len(value) == 4:
        return value + ".T"
    return value


def get_history(ticker, period):
    ticker = normalize_ticker(ticker)
    stock = yf.Ticker(ticker)
    if period == "1d":
        return stock.history(
            period="1d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
        )
    return stock.history(
        period=period,
        interval="1d",
        auto_adjust=False,
    )


def calculate_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    middle = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Middle"] = middle
    df["BB_Upper"] = middle + 2 * std20
    df["BB_Lower"] = middle - 2 * std20

    previous_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    df["Stochastic"] = (
        (close - lowest14) / (highest14 - lowest14).replace(0, np.nan) * 100
    )
    df["ROC"] = close.pct_change(12) * 100
    df["STD"] = close.rolling(20).std()
    df["VolumeRatio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    return df


def technical_analysis(df):
    latest = df.iloc[-1]
    score = 50
    up_reasons = []
    down_reasons = []
    close = safe_float(latest["Close"])

    for key, points, label in [
        ("MA20", 7, "20日移動平均線"),
        ("MA50", 7, "50日移動平均線"),
    ]:
        ma = safe_float(latest[key])
        if close is not None and ma is not None:
            if close > ma:
                score += points
                up_reasons.append(f"株価が{label}を上回っています")
            else:
                score -= points
                down_reasons.append(f"株価が{label}を下回っています")

    rsi = safe_float(latest["RSI"])
    if rsi is not None:
        if rsi < 30:
            score += 6
            up_reasons.append("RSIが30未満です")
        elif rsi > 70:
            score -= 6
            down_reasons.append("RSIが70超です")
        elif rsi >= 50:
            score += 3
            up_reasons.append("RSIが50以上です")
        else:
            score -= 3
            down_reasons.append("RSIが50未満です")

    macd = safe_float(latest["MACD"])
    signal = safe_float(latest["Signal"])
    if macd is not None and signal is not None:
        if macd > signal:
            score += 8
            up_reasons.append("MACDがシグナルを上回っています")
        else:
            score -= 8
            down_reasons.append("MACDがシグナルを下回っています")

    roc = safe_float(latest["ROC"])
    if roc is not None:
        if roc > 0:
            score += 5
            up_reasons.append("ROCがプラスです")
        else:
            score -= 5
            down_reasons.append("ROCがマイナスです")

    volume_ratio = safe_float(latest["VolumeRatio"])
    if volume_ratio is not None and volume_ratio > 1.5 and len(df) > 1:
        previous = safe_float(df["Close"].iloc[-2])
        if close is not None and previous is not None:
            if close > previous:
                score += 5
                up_reasons.append("出来高増加を伴う上昇です")
            else:
                score -= 5
                down_reasons.append("出来高増加を伴う下落です")

    score = max(5, min(95, score))
    direction = "上昇寄り" if score >= 65 else "下降寄り" if score <= 35 else "中立"
    return {
        "direction": direction,
        "up_probability": round(score, 1),
        "down_probability": round(100 - score, 1),
        "score": round(score, 1),
        "up_reasons": up_reasons[:6],
        "down_reasons": down_reasons[:6],
        "method": "ルールベースのテクニカルシグナル。将来の確率を保証する予測ではありません。",
    }


def fundamental_analysis(stock):
    result = {
        "available": False,
        "company": {},
        "valuation": {},
        "profitability": {},
        "growth": {},
        "financial_health": {},
        "shareholder_return": {},
        "scores": {},
        "overall_score": None,
        "good": [],
        "warning": [],
        "bad": [],
    }
    try:
        info = stock.info
    except Exception:
        return result

    def pct(key):
        value = info.get(key)
        return safe_float(value * 100) if value is not None else None

    result["company"] = {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
    }
    result["valuation"] = {
        "market_cap": safe_float(info.get("marketCap")),
        "enterprise_value": safe_float(info.get("enterpriseValue")),
        "per": safe_float(info.get("trailingPE")),
        "forward_per": safe_float(info.get("forwardPE")),
        "pbr": safe_float(info.get("priceToBook")),
        "psr": safe_float(info.get("priceToSalesTrailing12Months")),
        "peg": safe_float(info.get("pegRatio")),
    }
    result["profitability"] = {
        "eps": safe_float(info.get("trailingEps")),
        "forward_eps": safe_float(info.get("forwardEps")),
        "roe": pct("returnOnEquity"),
        "roa": pct("returnOnAssets"),
        "profit_margin": pct("profitMargins"),
        "operating_margin": pct("operatingMargins"),
    }
    result["growth"] = {
        "revenue_growth": pct("revenueGrowth"),
        "earnings_growth": pct("earningsGrowth"),
        "earnings_quarterly_growth": pct("earningsQuarterlyGrowth"),
    }
    result["financial_health"] = {
        "total_cash": safe_float(info.get("totalCash")),
        "total_debt": safe_float(info.get("totalDebt")),
        "debt_to_equity": safe_float(info.get("debtToEquity")),
        "current_ratio": safe_float(info.get("currentRatio")),
        "quick_ratio": safe_float(info.get("quickRatio")),
    }
    result["shareholder_return"] = {
        "dividend_yield": pct("dividendYield"),
        "dividend_rate": safe_float(info.get("dividendRate")),
        "payout_ratio": pct("payoutRatio"),
    }

    g = 50
    for value in [result["growth"]["revenue_growth"], result["growth"]["earnings_growth"]]:
        if value is not None:
            g += 20 if value >= 15 else 10 if value >= 5 else -15 if value < 0 else 0
    g = max(0, min(100, g))

    p = 50
    for value in [result["profitability"]["roe"], result["profitability"]["roa"], result["profitability"]["operating_margin"]]:
        if value is not None:
            p += 15 if value >= 15 else 7 if value >= 5 else -15 if value < 0 else 0
    p = max(0, min(100, p))

    h = 50
    de = result["financial_health"]["debt_to_equity"]
    cr = result["financial_health"]["current_ratio"]
    if de is not None:
        h += 25 if de <= 30 else 10 if de <= 80 else -25 if de >= 200 else -10 if de >= 120 else 0
    if cr is not None:
        h += 20 if cr >= 2 else 8 if cr >= 1 else -15
    h = max(0, min(100, h))

    v = 50
    per = result["valuation"]["per"]
    pbr = result["valuation"]["pbr"]
    if per is not None:
        v += 20 if 0 < per <= 15 else 5 if per <= 25 else -20 if per > 50 else 0
    if pbr is not None:
        v += 20 if 0 < pbr <= 1 else 8 if pbr <= 2 else -20 if pbr > 5 else 0
    v = max(0, min(100, v))

    r = 50
    dy = result["shareholder_return"]["dividend_yield"]
    payout = result["shareholder_return"]["payout_ratio"]
    if dy is not None:
        r += 25 if dy >= 4 else 12 if dy >= 2 else 5 if dy > 0 else 0
    if payout is not None:
        r += 20 if 20 <= payout <= 60 else -20 if payout > 100 else 0
    r = max(0, min(100, r))

    overall = round(g * 0.25 + p * 0.25 + h * 0.25 + v * 0.15 + r * 0.10, 1)
    result["scores"] = {
        "growth": g,
        "profitability": p,
        "financial_health": h,
        "valuation": v,
        "shareholder_return": r,
    }
    result["overall_score"] = overall
    result["available"] = True

    if g >= 70:
        result["good"].append("成長性が比較的良好です")
    elif g <= 35:
        result["bad"].append("成長性に注意が必要です")
    if p >= 70:
        result["good"].append("収益性が比較的良好です")
    elif p <= 35:
        result["bad"].append("収益性に弱さがあります")
    if h >= 70:
        result["good"].append("財務健全性が比較的良好です")
    elif h <= 35:
        result["bad"].append("財務面に注意が必要です")
    if per is not None and per >= 40:
        result["warning"].append("PERが高い水準です")
    if pbr is not None and pbr >= 5:
        result["warning"].append("PBRが高い水準です")
    if result["profitability"]["roe"] is not None and result["profitability"]["roe"] < 0:
        result["bad"].append("ROEがマイナスです")
    return result


def chart_rows(df):
    rows = []
    for index, row in df.iterrows():
        rows.append({
            "date": index.strftime("%Y-%m-%d %H:%M"),
            "open": safe_float(row["Open"]),
            "high": safe_float(row["High"]),
            "low": safe_float(row["Low"]),
            "close": safe_float(row["Close"]),
            "volume": safe_float(row["Volume"]),
            "ma20": safe_float(row["MA20"]),
            "ma50": safe_float(row["MA50"]),
            "ma200": safe_float(row["MA200"]),
            "bb_upper": safe_float(row["BB_Upper"]),
            "bb_middle": safe_float(row["BB_Middle"]),
            "bb_lower": safe_float(row["BB_Lower"]),
            "rsi": safe_float(row["RSI"]),
            "macd": safe_float(row["MACD"]),
            "signal": safe_float(row["Signal"]),
            "macd_hist": safe_float(row["MACD_Hist"]),
            "stochastic": safe_float(row["Stochastic"]),
            "atr": safe_float(row["ATR"]),
            "roc": safe_float(row["ROC"]),
            "std": safe_float(row["STD"]),
            "volume_ratio": safe_float(row["VolumeRatio"]),
        })
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    try:
        result = yf.Search(
            query,
            max_results=15,
            news_count=0,
            enable_fuzzy_query=True,
        )
        rows = []
        for item in result.quotes:
            symbol = item.get("symbol")
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "name": item.get("longname") or item.get("shortname") or symbol,
                "exchange": item.get("exchange") or "",
                "type": item.get("quoteType") or "",
            })
        if query.isdigit() and len(query) == 4:
            jpx = query + ".T"
            if not any(row["symbol"] == jpx for row in rows):
                rows.insert(0, {
                    "symbol": jpx,
                    "name": f"{query}（JPX候補）",
                    "exchange": "JPX",
                    "type": "EQUITY",
                })
        return jsonify(rows)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze")
def analyze():
    ticker = normalize_ticker(request.args.get("ticker", ""))
    period = request.args.get("period", "6mo")
    allowed = {"1d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
    if period not in allowed:
        period = "6mo"
    if not ticker:
        return jsonify({"error": "銘柄コードを入力してください"}), 400

    try:
        df = get_history(ticker, period)
        if df is None or df.empty:
            return jsonify({"error": f"{ticker} の株価データを取得できませんでした"}), 404
        df = calculate_indicators(df)
        stock = yf.Ticker(ticker)
        technical = technical_analysis(df)
        fundamental = fundamental_analysis(stock)
        latest = df.iloc[-1]
        price = safe_float(latest["Close"])
        previous = safe_float(df["Close"].iloc[-2]) if len(df) > 1 else None
        change = safe_float(price - previous) if price is not None and previous is not None else None
        change_percent = safe_float((price / previous - 1) * 100) if price is not None and previous not in (None, 0) else None
        return jsonify({
            "ticker": ticker,
            "price": price,
            "change": change,
            "change_percent": change_percent,
            "volume": safe_float(latest["Volume"]),
            "rsi": safe_float(latest["RSI"]),
            "macd": safe_float(latest["MACD"]),
            "signal": safe_float(latest["Signal"]),
            "atr": safe_float(latest["ATR"]),
            "stochastic": safe_float(latest["Stochastic"]),
            "roc": safe_float(latest["ROC"]),
            "std": safe_float(latest["STD"]),
            "volume_ratio": safe_float(latest["VolumeRatio"]),
            "technical": technical,
            "fundamental": fundamental,
            "company": fundamental["company"],
            "chart": chart_rows(df),
            "meta": {"period": period, "intraday": period == "1d"},
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "分析中にエラーが発生しました。銘柄コード・期間・ネット接続を確認してください。"}), 500


@app.route("/api/screener")
def screener():
    region = request.args.get("region", "jp").lower()
    mode = request.args.get("mode", "marketcap")
    try:
        page = max(0, int(request.args.get("page", 0)))
        size = min(250, max(25, int(request.args.get("size", 100))))
    except ValueError:
        page, size = 0, 100
    offset = page * size

    # Each filter is optional and independent.  When a value is supplied,
    # Yahoo's screener applies it before pagination, so page 2 is still page 2
    # of the filtered universe rather than page 2 of the unfiltered market.
    filter_specs = {
        "per": ("PER", "peratio.lasttwelvemonths"),
        "pbr": ("PBR", "pricebookratio.quarterly"),
        "peg": ("PEG", "pegratio_5y"),
        "roe": ("ROE", "returnonequity.lasttwelvemonths"),
        "roa": ("ROA", "returnonassets.lasttwelvemonths"),
        "ebitda_margin": ("EBITDA margin", "ebitdamargin.lasttwelvemonths"),
        "revenue_growth": ("売上成長率", "totalrevenues1yrgrowth.lasttwelvemonths"),
        "eps_growth": ("EPS成長率", "epsgrowth.lasttwelvemonths"),
        "dividend_yield": ("配当利回り", "dividendyield"),
        "debt_equity": ("D/E", "totaldebtequity.lasttwelvemonths"),
        "current_ratio": ("流動比率", "currentratio.lasttwelvemonths"),
        "market_cap": ("時価総額", "intradaymarketcap"),
        "price_change": ("騰落率", "percentchange"),
        "volume": ("出来高", "dayvolume"),
    }

    def query_number(key, bound):
        raw = request.args.get(f"{key}_{bound}")
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    try:
        region_code = "jp" if region == "jp" else "us"
        conditions = [EquityQuery("eq", ["region", region_code])]
        applied_filters = []

        for key, (label, field) in filter_specs.items():
            lo = query_number(key, "min")
            hi = query_number(key, "max")
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            if lo is not None and hi is not None:
                conditions.append(EquityQuery("btwn", [field, lo, hi]))
            elif lo is not None:
                conditions.append(EquityQuery("gte", [field, lo]))
            else:
                conditions.append(EquityQuery("lte", [field, hi]))
            applied_filters.append({"key": key, "label": label, "min": lo, "max": hi})

        # EquityQuery("and", ...) requires at least two operands.
        # With no user filters, the region condition is already a complete query.
        query = conditions[0] if len(conditions) == 1 else EquityQuery("and", conditions)
        sort_field = {
            "marketcap": "intradaymarketcap",
            "change": "percentchange",
            "volume": "dayvolume",
            "pe": "peratio.lasttwelvemonths",
        }.get(mode, "intradaymarketcap")
        response = yf.screen(
            query,
            offset=offset,
            size=size,
            sortField=sort_field,
            sortAsc=False,
        )
        quotes = response.get("quotes", []) if isinstance(response, dict) else []
        rows = []
        for item in quotes:
            rows.append({
                "symbol": item.get("symbol"),
                "name": item.get("longName") or item.get("shortName") or item.get("displayName") or item.get("symbol"),
                "price": safe_float(item.get("regularMarketPrice") or item.get("intradayprice") or item.get("regularMarketPreviousClose")),
                "change": safe_float(item.get("regularMarketChangePercent") or item.get("percentchange")),
                "market_cap": safe_float(item.get("marketCap") or item.get("intradaymarketcap")),
                "volume": safe_float(item.get("regularMarketVolume") or item.get("dayvolume")),
                "pe": safe_float(item.get("trailingPE") or item.get("peratio.lasttwelvemonths")),
                "pbr": safe_float(item.get("priceToBook") or item.get("pricebookratio.quarterly")),
                "peg": safe_float(item.get("pegRatio") or item.get("pegratio_5y")),
                "roe": safe_float(item.get("returnOnEquity") or item.get("returnonequity.lasttwelvemonths")),
                "roa": safe_float(item.get("returnOnAssets") or item.get("returnonassets.lasttwelvemonths")),
                "ebitda_margin": safe_float(item.get("ebitdaMargins") or item.get("ebitdamargin.lasttwelvemonths")),
                "revenue_growth": safe_float(item.get("revenueGrowth") or item.get("totalrevenues1yrgrowth.lasttwelvemonths")),
                "eps_growth": safe_float(item.get("earningsGrowth") or item.get("epsgrowth.lasttwelvemonths")),
                "dividend_yield": safe_float(item.get("dividendYield") or item.get("dividendyield")),
                "debt_equity": safe_float(item.get("debtToEquity") or item.get("totaldebtequity.lasttwelvemonths")),
                "current_ratio": safe_float(item.get("currentRatio") or item.get("currentratio.lasttwelvemonths")),
                "exchange": item.get("exchange") or "",
                "sector": item.get("sector") or "",
            })
        return jsonify({
            "region": region_code,
            "mode": mode,
            "page": page,
            "size": size,
            "results": rows,
            "filters": applied_filters,
            "filter_count": len(applied_filters),
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"スクリーナーを取得できませんでした: {exc}"}), 500


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "サーバー内部でエラーが発生しました"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
