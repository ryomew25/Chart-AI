from flask import Flask, render_template, request, jsonify

import sys
print("PYTHON:", sys.executable)

import sklearn
print("SKLEARN:", sklearn.__version__)

print("XGBOOST: OK")

import yfinance as yf
import pandas as pd
import numpy as np
import pywt

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


# ==========================================================
# 基本設定
# ==========================================================

PERIODS = {
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y"
}


# ==========================================================
# 基本データ処理
# ==========================================================

def normalize_data(data):

    if data is None:
        return None

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in data.columns
        ]

    return data


def safe_float(value):

    try:

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


def get_stock_data(symbol, period="1y"):

    if period not in PERIODS:
        period = "1y"

    data = yf.download(
        symbol,
        period=PERIODS[period],
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    data = normalize_data(data)

    needed = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in needed:

        if column not in data.columns:
            return None

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
    ).copy()

    return data


# ==========================================================
# 1. Multi-Timeframe Trend Confirmation
# ==========================================================

def calculate_mtf_trend(symbol):

    results = {}

    settings = {
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y"
    }

    bullish = 0
    bearish = 0

    for label, period in settings.items():

        data = get_stock_data(
            symbol,
            period
        )

        if data is None or len(data) < 30:
            continue

        close = data["Close"].copy()

        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]

        if (
            pd.notna(ma20)
            and pd.notna(ma50)
        ):

            if price > ma20 and ma20 > ma50:

                trend = "上昇"
                bullish += 1

            elif price < ma20 and ma20 < ma50:

                trend = "下降"
                bearish += 1

            else:

                trend = "中立"

        else:

            trend = "データ不足"

        results[label] = trend

    if bullish >= 3:

        overall = "複数時間軸で上昇"

    elif bearish >= 3:

        overall = "複数時間軸で下降"

    else:

        overall = "時間軸で方向が分かれる"

    return {
        "periods": results,
        "overall": overall
    }


# ==========================================================
# 2. Kalman Filter
# ==========================================================

def kalman_filter(
    values,
    process_variance=1e-5,
    measurement_variance=1e-2
):

    values = np.asarray(
        values,
        dtype=float
    ).copy()

    if len(values) == 0:
        return values

    estimate = values[0]

    error_covariance = 1.0

    result = []

    for measurement in values:

        prediction_covariance = (
            error_covariance
            + process_variance
        )

        kalman_gain = (
            prediction_covariance
            /
            (
                prediction_covariance
                + measurement_variance
            )
        )

        estimate = (
            estimate
            +
            kalman_gain
            *
            (
                measurement
                - estimate
            )
        )

        error_covariance = (
            (1 - kalman_gain)
            *
            prediction_covariance
        )

        result.append(
            estimate
        )

    return np.array(
        result,
        dtype=float
    )


# ==========================================================
# 3. Wavelet Transform
# ==========================================================

def wavelet_smooth(values):

    values = np.asarray(
        values,
        dtype=float
    ).copy()

    if len(values) < 16:
        return values.copy()

    wavelet = "db4"

    max_level = pywt.dwt_max_level(
        len(values),
        pywt.Wavelet(wavelet).dec_len
    )

    level = min(
        3,
        max_level
    )

    if level < 1:
        return values.copy()

    coefficients = pywt.wavedec(
        values,
        wavelet,
        mode="symmetric",
        level=level
    )

    coefficients[-1] = np.zeros_like(
        coefficients[-1]
    )

    smoothed = pywt.waverec(
        coefficients,
        wavelet,
        mode="symmetric"
    )

    return np.asarray(
        smoothed[:len(values)],
        dtype=float
    )


# ==========================================================
# 4. Volume Profile + VWAP
# ==========================================================

def volume_profile(data, bins=20):

    typical_price = (
        data["High"]
        + data["Low"]
        + data["Close"]
    ) / 3

    volume = data["Volume"]

    cumulative_volume = volume.cumsum()

    vwap = (
        (
            typical_price
            * volume
        ).cumsum()
        /
        cumulative_volume.replace(
            0,
            np.nan
        )
    )

    price = typical_price.to_numpy(
        dtype=float,
        copy=True
    )

    volume_values = volume.to_numpy(
        dtype=float,
        copy=True
    )

    if len(price) == 0:

        return {
            "vwap": None,
            "point_of_control": None
        }

    min_price = np.nanmin(price)
    max_price = np.nanmax(price)

    if min_price == max_price:

        poc = min_price

    else:

        hist, edges = np.histogram(
            price,
            bins=bins,
            weights=volume_values
        )

        index = int(
            np.argmax(hist)
        )

        poc = (
            edges[index]
            +
            edges[index + 1]
        ) / 2

    return {
        "vwap": safe_float(
            vwap.iloc[-1]
        ),
        "point_of_control": safe_float(
            poc
        )
    }


# ==========================================================
# 5. Markov Regime Detector
# ==========================================================

def markov_regime(data):

    close = data["Close"].copy()

    returns = (
        close
        .pct_change()
        .dropna()
    )

    if len(returns) < 30:

        return {
            "regime": "データ不足",
            "volatility": None
        }

    recent_return = (
        returns.tail(20).mean()
    )

    volatility = (
        returns.tail(20).std()
        * np.sqrt(252)
    )

    if recent_return > 0.002:

        regime = "上昇レジーム"

    elif recent_return < -0.002:

        regime = "下降レジーム"

    elif volatility > 0.35:

        regime = "高ボラティリティ"

    else:

        regime = "レンジ"

    return {
        "regime": regime,
        "volatility": safe_float(
            volatility
        )
    }


# ==========================================================
# 6. HMM代替
#
# hmmlearn / GaussianHMMを一切使用しない。
#
# リターンとボラティリティを使って、
# 過去データを3つの状態に分類する簡易レジームモデル。
# ==========================================================

def hmm_regime(data):

    close = data["Close"].copy()

    returns = (
        close
        .pct_change()
        .dropna()
    )

    if len(returns) < 80:

        return {
            "state": "データ不足",
            "state_number": None,
            "states": []
        }

    rolling_volatility = (
        returns
        .rolling(10)
        .std()
        .fillna(
            returns.std()
        )
    )

    feature_returns = returns.to_numpy(
        dtype=float,
        copy=True
    )

    feature_volatility = rolling_volatility.to_numpy(
        dtype=float,
        copy=True
    )

    # ------------------------------------------
    # 3状態の基準を統計的に作る
    # ------------------------------------------

    return_low = np.percentile(
        feature_returns,
        33
    )

    return_high = np.percentile(
        feature_returns,
        67
    )

    vol_low = np.percentile(
        feature_volatility,
        33
    )

    vol_high = np.percentile(
        feature_volatility,
        67
    )

    states = []

    for ret, vol in zip(
        feature_returns,
        feature_volatility
    ):

        if ret > return_high:

            state = 2

        elif ret < return_low:

            state = 0

        else:

            state = 1

        # 強い変動時はボラティリティも考慮
        if vol > vol_high:

            if ret > 0:
                state = 2

            elif ret < 0:
                state = 0

        states.append(state)

    states = np.asarray(
        states,
        dtype=int
    )

    current_state = int(
        states[-1]
    )

    if current_state == 2:

        label = "強気状態"

    elif current_state == 0:

        label = "弱気状態"

    else:

        label = "中間状態"

    return {
        "state": label,
        "state_number": current_state,
        "states": states[-30:].tolist(),
        "method": "統計的3状態レジームモデル"
    }


# ==========================================================
# 7. Dynamic Time Warping
# ==========================================================

def dtw_distance(a, b):

    a = np.asarray(
        a,
        dtype=float
    ).copy()

    b = np.asarray(
        b,
        dtype=float
    ).copy()

    n = len(a)
    m = len(b)

    matrix = np.full(
        (n + 1, m + 1),
        np.inf,
        dtype=float
    )

    matrix[0, 0] = 0

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            cost = abs(
                a[i - 1]
                -
                b[j - 1]
            )

            matrix[i, j] = (
                cost
                +
                min(
                    matrix[i - 1, j],
                    matrix[i, j - 1],
                    matrix[i - 1, j - 1]
                )
            )

    return float(
        matrix[n, m]
    )


def find_similar_pattern(data):

    close = data["Close"].dropna().copy()

    window = 30

    if len(close) < 100:

        return {
            "distance": None,
            "future_direction": "データ不足"
        }

    current = close.iloc[
        -window:
    ].to_numpy(
        dtype=float,
        copy=True
    )

    current = (
        current
        /
        current[0]
        - 1
    )

    best_distance = np.inf

    best_direction = "不明"

    for end in range(
        window,
        len(close) - 10
    ):

        pattern = close.iloc[
            end - window:end
        ].to_numpy(
            dtype=float,
            copy=True
        )

        pattern = (
            pattern
            /
            pattern[0]
            - 1
        )

        distance = dtw_distance(
            current,
            pattern
        )

        if distance < best_distance:

            best_distance = distance

            future = (
                close.iloc[
                    end + 10
                ]
                /
                close.iloc[end]
                - 1
            )

            if future > 0.02:

                best_direction = "過去類似例は上昇"

            elif future < -0.02:

                best_direction = "過去類似例は下降"

            else:

                best_direction = "過去類似例は横ばい"

    return {
        "distance": safe_float(
            best_distance
        ),
        "future_direction": best_direction
    }


# ==========================================================
# 機械学習用特徴量
# ==========================================================

def create_features(data):

    df = data.copy()

    close = df["Close"].copy()

    df["return_1"] = close.pct_change()

    df["return_5"] = (
        close.pct_change(5)
    )

    df["return_20"] = (
        close.pct_change(20)
    )

    df["ma20_ratio"] = (
        close
        /
        close.rolling(20).mean()
        - 1
    )

    df["ma50_ratio"] = (
        close
        /
        close.rolling(50).mean()
        - 1
    )

    df["volatility"] = (
        df["return_1"]
        .rolling(20)
        .std()
    )

    df["volume_ratio"] = (
        df["Volume"]
        /
        df["Volume"]
        .rolling(20)
        .mean()
    )

    return df


# ==========================================================
# 8. XGBoost
# ==========================================================

def xgboost_prediction(data):

    df = create_features(data)

    df["target"] = (
        df["Close"].shift(-1)
        >
        df["Close"]
    ).astype(int)

    features = [
        "return_1",
        "return_5",
        "return_20",
        "ma20_ratio",
        "ma50_ratio",
        "volatility",
        "volume_ratio"
    ]

    train = df[
        features + ["target"]
    ].dropna()

    if len(train) < 120:

        return {
            "prediction": "データ不足",
            "score": None,
            "probability": None,
            "test_accuracy": None,
            "features": features
        }

    split = int(
        len(train) * 0.8
    )

    train_part = train.iloc[:split]
    test_part = train.iloc[split:]

    X_train = train_part[
        features
    ].copy()

    y_train = train_part[
        "target"
    ].copy()

    X_test = test_part[
        features
    ].copy()

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    latest = df[
        features
    ].dropna().iloc[-1:].copy()

    probability = float(
        model.predict_proba(
            latest
        )[0][1]
    )

    if probability >= 0.55:

        prediction = "上昇シグナル"

    elif probability <= 0.45:

        prediction = "下降シグナル"

    else:

        prediction = "中立"

    accuracy = None

    if len(X_test) > 0:

        predictions = model.predict(
            X_test
        )

        accuracy = float(
            (
                predictions
                ==
                test_part["target"].to_numpy()
            ).mean()
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "test_accuracy": accuracy,
        "features": features
    }


# ==========================================================
# 9. XGBoost Feature Importance
# ==========================================================

def explain_xgboost(data):

    df = create_features(data)

    features = [
        "return_1",
        "return_5",
        "return_20",
        "ma20_ratio",
        "ma50_ratio",
        "volatility",
        "volume_ratio"
    ]

    df["target"] = (
        df["Close"].shift(-1)
        >
        df["Close"]
    ).astype(int)

    train = df[
        features + ["target"]
    ].dropna()

    if len(train) < 120:
        return []

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        train[features].copy(),
        train["target"].copy()
    )

    importance = (
        model.feature_importances_
    )

    result = []

    for feature, value in zip(
        features,
        importance
    ):

        result.append({
            "feature": feature,
            "importance": safe_float(
                value
            )
        })

    result.sort(
        key=lambda x:
            x["importance"],
        reverse=True
    )

    return result


# ==========================================================
# 10. Walk-Forward Validation
# ==========================================================

def walk_forward(data):

    df = create_features(data)

    features = [
        "return_1",
        "return_5",
        "return_20",
        "ma20_ratio",
        "ma50_ratio",
        "volatility",
        "volume_ratio"
    ]

    df["target"] = (
        df["Close"].shift(-1)
        >
        df["Close"]
    ).astype(int)

    df = df.dropna().copy()

    if len(df) < 180:

        return {
            "accuracy": None,
            "samples": 0
        }

    results = []

    initial_train = 120
    step = 20

    for end in range(
        initial_train,
        len(df),
        step
    ):

        train = df.iloc[:end].copy()

        test = df.iloc[
            end:min(
                end + step,
                len(df)
            )
        ].copy()

        if len(test) == 0:
            break

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )

        model.fit(
            train[features].copy(),
            train["target"].copy()
        )

        predictions = model.predict(
            test[features].copy()
        )

        accuracy = (
            predictions
            ==
            test["target"].to_numpy()
        ).mean()

        results.append(
            float(accuracy)
        )

    if not results:

        return {
            "accuracy": None,
            "samples": 0
        }

    return {
        "accuracy":
            float(
                np.mean(results)
            ),

        "samples":
            len(results)
    }


# ==========================================================
# 総合分析
# ==========================================================

# ==========================================================
# 11. 総合方向判定・説明生成
# ==========================================================

def generate_overall_analysis(analysis):

    scores = []

    reasons = []

    warnings = []

    # ------------------------------------------------------
    # Multi-Timeframe
    # ------------------------------------------------------

    mtf = analysis.get("mtf", {})
    mtf_overall = mtf.get("overall", "")

    if "上昇" in mtf_overall:

        scores.append(2)

        reasons.append(
            "複数の時間軸で上昇傾向が確認されています。"
        )

    elif "下降" in mtf_overall:

        scores.append(-2)

        reasons.append(
            "複数の時間軸で下降傾向が確認されています。"
        )

    else:

        scores.append(0)

        reasons.append(
            "時間軸によって方向感が分かれています。"
        )

    # ------------------------------------------------------
    # Markov
    # ------------------------------------------------------

    markov = analysis.get("markov", {})
    regime = markov.get("regime", "")

    if "上昇" in regime:

        scores.append(2)

        reasons.append(
            "現在の市場レジームは上昇傾向です。"
        )

    elif "下降" in regime:

        scores.append(-2)

        reasons.append(
            "現在の市場レジームは下降傾向です。"
        )

    elif "高ボラティリティ" in regime:

        scores.append(0)

        warnings.append(
            "現在はボラティリティが高く、価格変動が大きくなる可能性があります。"
        )

    else:

        scores.append(0)

    # ------------------------------------------------------
    # DTW
    # ------------------------------------------------------

    dtw = analysis.get("dtw", {})
    dtw_direction = dtw.get("future_direction", "")

    if "上昇" in dtw_direction:

        scores.append(1)

        reasons.append(
            "過去の類似した価格パターンでは、その後に上昇した例が確認されています。"
        )

    elif "下降" in dtw_direction:

        scores.append(-1)

        reasons.append(
            "過去の類似した価格パターンでは、その後に下降した例が確認されています。"
        )

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    xgb = analysis.get("xgboost", {})
    prediction = xgb.get("prediction", "")

    probability = xgb.get("probability")

    if "上昇" in prediction:

        scores.append(2)

        reasons.append(
            "XGBoostモデルは上昇シグナルを示しています。"
        )

    elif "下降" in prediction:

        scores.append(-2)

        reasons.append(
            "XGBoostモデルは下降シグナルを示しています。"
        )

    else:

        scores.append(0)

    # ------------------------------------------------------
    # Volatility
    # ------------------------------------------------------

    volatility = markov.get("volatility")

    if volatility is not None:

        if volatility >= 0.35:

            warnings.append(
                "過去の値動きから見て変動幅が大きく、方向が外れるリスクにも注意が必要です。"
            )

        elif volatility <= 0.15:

            reasons.append(
                "直近のボラティリティは比較的低い状態です。"
            )

    # ------------------------------------------------------
    # 総合スコア
    # ------------------------------------------------------

    total = sum(scores)

    if total >= 4:

        direction = "上昇する可能性が高め"

        strength = "強い上昇傾向"

    elif total >= 2:

        direction = "上昇する可能性がやや高い"

        strength = "やや上昇寄り"

    elif total <= -4:

        direction = "下降する可能性が高め"

        strength = "強い下降傾向"

    elif total <= -2:

        direction = "下降する可能性がやや高い"

        strength = "やや下降寄り"

    else:

        direction = "上昇・下降の方向感は弱い"

        strength = "中立"

    # ------------------------------------------------------
    # 確率表示
    # ------------------------------------------------------

    probability_text = None

    if probability is not None:

        probability_text = (
            f"XGBoostの上昇確率は "
            f"{probability * 100:.1f}% です。"
        )

    # ------------------------------------------------------
    # 説明
    # ------------------------------------------------------

    explanation = " ".join(
        reasons[:4]
    )

    if not explanation:

        explanation = (
            "複数の分析結果から明確な方向性を判断できませんでした。"
        )

    # ------------------------------------------------------
    # 補足
    # ------------------------------------------------------

    supplement_parts = []

    if warnings:

        supplement_parts.extend(
            warnings[:2]
        )

    if probability_text:

        supplement_parts.append(
            probability_text
        )

    supplement_parts.append(
        "この判定は過去の市場データを複数の分析手法で評価したものであり、将来の価格を保証するものではありません。"
    )

    return {

        "direction":
            direction,

        "strength":
            strength,

        "score":
            total,

        "explanation":
            explanation,

        "supplement":
            " ".join(
                supplement_parts
            )

    }

def complete_analysis(
    symbol,
    data
):

    mtf = calculate_mtf_trend(
        symbol
    )

    kalman = kalman_filter(
        data["Close"].to_numpy(
            dtype=float,
            copy=True
        )
    )

    wavelet = wavelet_smooth(
        data["Close"].to_numpy(
            dtype=float,
            copy=True
        )
    )

    volume = volume_profile(
        data
    )

    markov = markov_regime(
        data
    )

    hmm = hmm_regime(
        data
    )

    dtw = find_similar_pattern(
        data
    )

    xgb = xgboost_prediction(
        data
    )

    importance = explain_xgboost(
        data
    )

    walk_forward_result = walk_forward(
        data
    )

    result = {

        "mtf":
            mtf,

        "kalman_latest":
            safe_float(
                kalman[-1]
            ),

        "wavelet_latest":
            safe_float(
                wavelet[-1]
            ),

        "volume":
            volume,

        "markov":
            markov,

        "hmm":
            hmm,

        "dtw":
            dtw,

        "xgboost":
            xgb,

        "feature_importance":
            importance,

        "walk_forward":
            walk_forward_result

    }

    result["overall"] = generate_overall_analysis(
        result
    )

    return result

    return {

        "mtf": mtf,

        "kalman_latest":
            safe_float(
                kalman[-1]
            ),

        "wavelet_latest":
            safe_float(
                wavelet[-1]
            ),

        "volume": volume,

        "markov": markov,

        "hmm": hmm,

        "dtw": dtw,

        "xgboost": xgb,

        "feature_importance":
            importance,

        "walk_forward":
            walk_forward_result
    }


# ==========================================================
# チャートデータ
# ==========================================================

def make_chart_data(data):

    close = data["Close"].to_numpy(
        dtype=float,
        copy=True
    )

    kalman = kalman_filter(
        close
    )

    wavelet = wavelet_smooth(
        close
    )

    rows = []

    for i, (index, row) in enumerate(
        data.iterrows()
    ):

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

            "kalman":
                safe_float(
                    kalman[i]
                ),

            "wavelet":
                safe_float(
                    wavelet[i]
                )
        })

    return rows


# ==========================================================
# API
# ==========================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


@app.get("/api/search")
def search_api():

    keyword = request.args.get(
        "q",
        ""
    ).strip()

    if not keyword:
        return jsonify([])

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

                "symbol":
                    item.get(
                        "symbol",
                        ""
                    ),

                "name":
                    (
                        item.get(
                            "longname"
                        )
                        or
                        item.get(
                            "shortname"
                        )
                        or
                        item.get(
                            "symbol"
                        )
                    ),

                "exchange":
                    item.get(
                        "exchange",
                        ""
                    )
            })

        return jsonify(
            stocks
        )

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


@app.get("/api/analyze")
def analyze_api():

    symbol = request.args.get(
        "symbol",
        ""
    ).strip()

    period = request.args.get(
        "period",
        "1y"
    )

    if not symbol:

        return jsonify({
            "error":
                "銘柄が指定されていません。"
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

        analysis = complete_analysis(
            symbol,
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
            "========== ANALYSIS ERROR =========="
        )

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print(
            "===================================="
        )

        return jsonify({
            "error":
                f"{type(exc).__name__}: {exc}"
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True
    )