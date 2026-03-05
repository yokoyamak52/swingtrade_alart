#!/usr/bin/env python3
"""
generate_data.py
yfinance で株価を取得し、テクニカル指標を計算して docs/data.json に出力する。
talib 不使用。pandas / numpy のみ。
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from watchlist import WATCHLIST

# ──────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────
JST = timezone(timedelta(hours=9))
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "data.json")


# ──────────────────────────────────────────────
# テクニカル指標計算
# ──────────────────────────────────────────────
def calc_sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder法 RSI（ewm alpha=1/period, adjust=False）"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bb(series: pd.Series, n: int = 20, k: float = 2.0):
    """ボリンジャーバンド (ddof=0)"""
    mid = series.rolling(n).mean()
    std = series.rolling(n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ──────────────────────────────────────────────
# シグナルスコアリング
# ──────────────────────────────────────────────
def score_ma(row, prev):
    """移動平均シグナル"""
    ma5, ma25, ma75 = row.get("ma5"), row.get("ma25"), row.get("ma75")
    close = row["close"]

    if any(v is None or np.isnan(v) for v in [ma5, ma25, ma75]):
        return {"score": 0, "label": "データ不足", "icon": "—", "type": "neutral"}

    # MA の傾き判定（前日比）
    pma5 = prev.get("ma5") if prev else None
    pma25 = prev.get("ma25") if prev else None
    pma75 = prev.get("ma75") if prev else None

    ma5_up = pma5 is not None and not np.isnan(pma5) and ma5 > pma5
    ma25_up = pma25 is not None and not np.isnan(pma25) and ma25 > pma25
    ma75_up = pma75 is not None and not np.isnan(pma75) and ma75 > pma75

    # パーフェクトオーダー: MA5>MA25>MA75 かつ全上向き
    if ma5 > ma25 > ma75 and ma5_up and ma25_up and ma75_up:
        return {"score": 2, "label": "パーフェクトオーダー", "icon": "📈", "type": "buy"}

    # ゴールデンクロス当日
    if prev:
        pma5_v = prev.get("ma5")
        pma25_v = prev.get("ma25")
        if (pma5_v is not None and pma25_v is not None
                and not np.isnan(pma5_v) and not np.isnan(pma25_v)):
            if ma5 > ma25 and pma5_v <= pma25_v:
                return {"score": 2, "label": "GC発生", "icon": "✨", "type": "buy"}
            if ma5 < ma25 and pma5_v >= pma25_v:
                return {"score": -2, "label": "DC発生", "icon": "💀", "type": "sell"}

    # 上昇継続
    if ma5 > ma25 and close > ma25:
        return {"score": 1, "label": "MA上昇中", "icon": "↗", "type": "buy"}

    # MA25割れ
    if close < ma25:
        return {"score": -1, "label": "MA25割れ", "icon": "↘", "type": "sell"}

    return {"score": 0, "label": "中立", "icon": "→", "type": "neutral"}


def score_rsi(rsi_val):
    if rsi_val is None or np.isnan(rsi_val):
        return {"score": 0, "label": "RSI —", "icon": "〜", "type": "neutral"}
    rv = round(rsi_val, 1)
    if 40 <= rsi_val <= 60:
        return {"score": 2, "label": f"RSI {rv} 押し目", "icon": "🎯", "type": "buy"}
    if 60 < rsi_val < 70:
        return {"score": 1, "label": f"RSI {rv}", "icon": "💪", "type": "buy"}
    if 70 <= rsi_val < 80:
        return {"score": -1, "label": f"RSI {rv} 過熱", "icon": "🔥", "type": "warn"}
    if rsi_val >= 80:
        return {"score": -2, "label": f"RSI {rv} 警戒", "icon": "🚨", "type": "sell"}
    if rsi_val < 30:
        return {"score": -2, "label": f"RSI {rv}", "icon": "📉", "type": "sell"}
    return {"score": 0, "label": f"RSI {rv}", "icon": "〜", "type": "neutral"}


def score_macd(row, prev):
    macd_val = row.get("macd")
    sig_val = row.get("macd_signal")
    if macd_val is None or sig_val is None or np.isnan(macd_val) or np.isnan(sig_val):
        return {"score": 0, "label": "データ不足", "icon": "—", "type": "neutral"}

    if prev:
        pm = prev.get("macd")
        ps = prev.get("macd_signal")
        if pm is not None and ps is not None and not np.isnan(pm) and not np.isnan(ps):
            if macd_val > sig_val and pm <= ps:
                return {"score": 2, "label": "GCクロス", "icon": "⚡", "type": "buy"}
            if macd_val < sig_val and pm >= ps:
                return {"score": -2, "label": "DCクロス", "icon": "⚠️", "type": "sell"}

    if macd_val > sig_val:
        return {"score": 1, "label": "MACD強気", "icon": "🔺", "type": "buy"}
    return {"score": -1, "label": "MACD弱気", "icon": "▽", "type": "sell"}


def score_bb(row):
    bb_u = row.get("bb_upper")
    bb_l = row.get("bb_lower")
    close = row["close"]
    if bb_u is None or bb_l is None or np.isnan(bb_u) or np.isnan(bb_l):
        return {"score": 0, "label": "データ不足", "icon": "—", "type": "neutral"}
    band_range = bb_u - bb_l
    if band_range <= 0:
        return {"score": 0, "label": "バンド収縮", "icon": "⊙", "type": "neutral"}
    bb_pos = (close - bb_l) / band_range
    bb_pos_r = round(bb_pos, 2)
    if bb_pos > 0.85:
        return {"score": 2, "label": "バンドウォーク", "icon": "🚀", "type": "buy"}
    if bb_pos > 0.60:
        return {"score": 1, "label": "上半分推移", "icon": "📊", "type": "buy"}
    if bb_pos < 0.15:
        return {"score": -2, "label": "-2σ割れ", "icon": "💥", "type": "sell"}
    if bb_pos < 0.40:
        return {"score": -1, "label": "下半分", "icon": "📉", "type": "sell"}
    return {"score": 0, "label": "バンド中央", "icon": "⊙", "type": "neutral"}


def score_volume(row):
    vr = row.get("vol_ratio")
    if vr is None or np.isnan(vr):
        return {"score": 0, "label": "—倍", "icon": "📊", "type": "neutral"}
    is_up = row["close"] > row["open"]
    vr_r = round(vr, 1)
    if vr >= 1.5 and is_up:
        return {"score": 2, "label": f"出来高{vr_r}倍↑", "icon": "🔊", "type": "buy"}
    if vr >= 1.2 and is_up:
        return {"score": 1, "label": f"{vr_r}倍", "icon": "📢", "type": "buy"}
    if vr >= 1.5 and not is_up:
        return {"score": -2, "label": f"出来高{vr_r}倍↓", "icon": "🔕", "type": "sell"}
    if vr < 0.7 and is_up:
        return {"score": -1, "label": "出来高細い", "icon": "📉", "type": "warn"}
    return {"score": 0, "label": f"{vr_r}倍", "icon": "📊", "type": "neutral"}


def calc_overall(total):
    if total >= 5:
        return {"label": "強い買い", "color": "#00e896", "icon": "🟢🟢"}
    if total >= 3:
        return {"label": "買い検討", "color": "#00c97a", "icon": "🟢"}
    if total >= 1:
        return {"label": "やや強気", "color": "#44aadd", "icon": "🔵"}
    if total >= -1:
        return {"label": "中立・様子見", "color": "#8899aa", "icon": "⚪"}
    if total >= -3:
        return {"label": "やや弱気", "color": "#ff9944", "icon": "🟡"}
    if total >= -5:
        return {"label": "売り検討", "color": "#ff5533", "icon": "🔴"}
    return {"label": "強い売り", "color": "#ff2244", "icon": "🔴🔴"}


def check_entry_conditions(row, prev):
    """エントリー4条件チェック"""
    ma25 = row.get("ma25")
    ma5 = row.get("ma5")
    close = row["close"]
    open_price = row["open"]
    rsi_val = row.get("rsi")
    vr = row.get("vol_ratio")

    cond1 = bool(ma25 and ma5 and not np.isnan(ma25) and not np.isnan(ma5)
                 and close > ma25 and ma5 > ma25)
    cond2 = bool(rsi_val is not None and not np.isnan(rsi_val)
                 and 40 <= rsi_val <= 60)
    cond3 = bool(close > open_price
                 and ma5 is not None and not np.isnan(ma5)
                 and close > ma5)
    cond4 = bool(vr is not None and not np.isnan(vr) and vr >= 1.0)

    return [
        {"label": "MA上向き＋株価がMA上", "ok": cond1},
        {"label": "日足押し目 RSI 40-60圏", "ok": cond2},
        {"label": "直近陽線 or 5MA反発確認", "ok": cond3},
        {"label": "出来高20日平均以上", "ok": cond4},
    ]


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
def process_stock(item: dict) -> dict:
    code = item["code"]
    name = item["name"]
    print(f"  処理中: {name} ({code})")

    try:
        ticker = yf.Ticker(code)
        df = ticker.history(period="6mo")

        if df.empty or len(df) < 30:
            print(f"    ⚠ データ不足: {len(df)}行")
            return {
                "code": code,
                "name": name,
                "sector": item["sector"],
                "theme": item["theme"],
                "risk": item["risk"],
                "error": True,
                "error_msg": "データ不足",
            }

        # カラム名を小文字に統一
        df.columns = [c.lower() for c in df.columns]

        # 指標計算
        df["ma5"] = calc_sma(df["close"], 5)
        df["ma25"] = calc_sma(df["close"], 25)
        df["ma75"] = calc_sma(df["close"], 75)

        bb_mid, bb_upper, bb_lower = calc_bb(df["close"], 20, 2.0)
        df["bb_mid"] = bb_mid
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower

        macd_line, signal_line, histogram = calc_macd(df["close"], 12, 26, 9)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram

        df["rsi"] = calc_rsi_wilder(df["close"], 14)

        vol_ma20 = calc_sma(df["volume"], 20)
        df["vol_ratio"] = df["volume"] / vol_ma20

        # 最新行・前日行を dict 化
        last = df.iloc[-1].to_dict()
        prev = df.iloc[-2].to_dict() if len(df) >= 2 else None

        # NaN を None に変換するヘルパー
        def clean(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        last_clean = {k: clean(v) for k, v in last.items()}
        prev_clean = {k: clean(v) for k, v in prev.items()} if prev else None

        # シグナルスコアリング
        sig_ma = score_ma(last_clean, prev_clean)
        sig_rsi = score_rsi(last_clean.get("rsi"))
        sig_macd = score_macd(last_clean, prev_clean)
        sig_bb = score_bb(last_clean)
        sig_vol = score_volume(last_clean)

        signals = {
            "ma": sig_ma,
            "rsi": sig_rsi,
            "macd": sig_macd,
            "bb": sig_bb,
            "vol": sig_vol,
        }

        total_score = sum(s["score"] for s in signals.values())
        overall = calc_overall(total_score)
        checks = check_entry_conditions(last_clean, prev_clean)

        # 前日比
        prev_close = prev_clean["close"] if prev_clean and prev_clean.get("close") else None
        change = None
        change_pct = None
        if prev_close and prev_close > 0:
            change = round(last_clean["close"] - prev_close, 1)
            change_pct = round((change / prev_close) * 100, 2)

        # 日付
        last_date = df.index[-1]
        date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

        return {
            "code": code,
            "name": name,
            "sector": item["sector"],
            "theme": item["theme"],
            "risk": item["risk"],
            "error": False,
            "date": date_str,
            "close": round(last_clean["close"], 1),
            "open": round(last_clean["open"], 1),
            "change": change,
            "change_pct": change_pct,
            "ma5": round(last_clean["ma5"], 1) if last_clean.get("ma5") is not None else None,
            "ma25": round(last_clean["ma25"], 1) if last_clean.get("ma25") is not None else None,
            "ma75": round(last_clean["ma75"], 1) if last_clean.get("ma75") is not None else None,
            "rsi": round(last_clean["rsi"], 1) if last_clean.get("rsi") is not None else None,
            "macd": round(last_clean["macd"], 1) if last_clean.get("macd") is not None else None,
            "macd_signal": round(last_clean["macd_signal"], 1) if last_clean.get("macd_signal") is not None else None,
            "bb_upper": round(last_clean["bb_upper"], 1) if last_clean.get("bb_upper") is not None else None,
            "bb_lower": round(last_clean["bb_lower"], 1) if last_clean.get("bb_lower") is not None else None,
            "vol_ratio": round(last_clean["vol_ratio"], 2) if last_clean.get("vol_ratio") is not None else None,
            "signals": signals,
            "total_score": total_score,
            "overall": overall,
            "checks": checks,
        }

    except Exception as e:
        print(f"    ❌ エラー: {e}")
        return {
            "code": code,
            "name": name,
            "sector": item["sector"],
            "theme": item["theme"],
            "risk": item["risk"],
            "error": True,
            "error_msg": str(e),
        }


def main():
    print("=" * 50)
    print("SWING-AI データ生成")
    print("=" * 50)

    now = datetime.now(JST)
    print(f"実行時刻: {now.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"銘柄数: {len(WATCHLIST)}")
    print()

    results = []
    errors = 0
    for item in WATCHLIST:
        result = process_stock(item)
        results.append(result)
        if result.get("error"):
            errors += 1

    # スコア降順ソート（エラー銘柄は末尾）
    results.sort(key=lambda x: (x.get("error", False), -(x.get("total_score", -99))))

    output = {
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at_iso": now.isoformat(),
        "total_stocks": len(results),
        "errors": errors,
        "stocks": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print(f"✅ 完了: {OUTPUT_PATH}")
    print(f"   正常: {len(results) - errors} / エラー: {errors}")

    # エラーがあっても 0 で終了（GitHub Actions を止めないため）
    # 全銘柄エラーの場合のみ異常終了
    if errors == len(results):
        print("❌ 全銘柄でエラーが発生しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
