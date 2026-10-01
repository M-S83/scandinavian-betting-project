"""
Chapter 3 - Can a simple rating model beat the closing line?

Walk-forward test: for every match, predict using only earlier matches,
then compare to the bookmaker's closing odds with the margin removed.

Run:
    python Src/scandi_backtest.py                 # uses whatever data it finds
    python Src/scandi_backtest.py --min-edge 0.04

Data it looks for (both optional, it uses what exists):
    Data/Raw/footballdata/DNK.csv, NOR.csv, SWE.csv   (football-data.co.uk, /new/ files)
    Data/Processed/Denmark2025.csv, Norway2025.csv, Sweden2025.csv  (FootyStats, already in repo)

The football-data files have many seasons and Pinnacle closing odds, so they
are the ones that can actually answer the question. The FootyStats files are
one season only, which is a smoke test, not a result.
"""
import argparse
import os
from math import factorial

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD_DIR = os.path.join(ROOT, "Data", "Raw", "footballdata")
FS_DIR = os.path.join(ROOT, "Data", "Processed")

# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------

FD_LEAGUES = {"DNK": "Denmark", "NOR": "Norway", "SWE": "Sweden"}


def _first_present(df, names):
    for n in names:
        if n in df.columns and df[n].notna().sum() > 0:
            return n
    return None


def load_footballdata():
    frames = []
    for code, league in FD_LEAGUES.items():
        path = os.path.join(FD_DIR, f"{code}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        need = ["Date", "Home", "Away", "HG", "AG"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(f"{path}: missing {missing}. Columns found: {list(d.columns)}")
        # closing odds: prefer Pinnacle closing, then Pinnacle, then market average
        sets = {
            "pinnacle_close": ("PSCH", "PSCD", "PSCA"),
            "pinnacle": ("PH", "PD", "PA"),
            "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
            "avg": ("AvgH", "AvgD", "AvgA"),
        }
        chosen = None
        for name, cols in sets.items():
            if all(c in d.columns for c in cols):
                chosen = (name, cols)
                break
        if chosen is None:
            raise ValueError(f"{path}: no usable 1X2 odds columns. Columns found: {list(d.columns)}")
        src, (h, dr, a) = chosen
        out = pd.DataFrame({
            "league": league,
            "date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce"),
            "home": d["Home"],
            "away": d["Away"],
            "hg": pd.to_numeric(d["HG"], errors="coerce"),
            "ag": pd.to_numeric(d["AG"], errors="coerce"),
            "odds_h": pd.to_numeric(d[h], errors="coerce"),
            "odds_d": pd.to_numeric(d[dr], errors="coerce"),
            "odds_a": pd.to_numeric(d[a], errors="coerce"),
            "odds_source": f"footballdata:{src}",
        })
        # best available price, for the "can you actually get this" check
        mh = _first_present(d, ["MaxCH", "MaxH"])
        md = _first_present(d, ["MaxCD", "MaxD"])
        ma = _first_present(d, ["MaxCA", "MaxA"])
        if mh and md and ma:
            out["max_h"] = pd.to_numeric(d[mh], errors="coerce")
            out["max_d"] = pd.to_numeric(d[md], errors="coerce")
            out["max_a"] = pd.to_numeric(d[ma], errors="coerce")
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_footystats():
    frames = []
    for league in FD_LEAGUES.values():
        path = os.path.join(FS_DIR, f"{league}2025.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d["status"] == "complete"]
        frames.append(pd.DataFrame({
            "league": league,
            "date": pd.to_datetime(d["timestamp"], unit="s"),
            "home": d["home_team_name"],
            "away": d["away_team_name"],
            "hg": d["home_team_goal_count"],
            "ag": d["away_team_goal_count"],
            "odds_h": d["odds_ft_home_team_win"],
            "odds_d": d["odds_ft_draw"],
            "odds_a": d["odds_ft_away_team_win"],
            "odds_source": "footystats:unknown_bookmaker",
        }))
    return pd.concat(frames, ignore_index=True) if frames else None


def load_all():
    df = load_footballdata()
    if df is None:
        print("No football-data files found, falling back to FootyStats (one season, smoke test only).")
        df = load_footystats()
    if df is None:
        raise SystemExit("No data found. See the docstring at the top of this file.")
    df = df.dropna(subset=["date", "hg", "ag", "odds_h", "odds_d", "odds_a"])
    df = df[(df[["odds_h", "odds_d", "odds_a"]] > 1).all(axis=1)]
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Odds -> fair probabilities
# ----------------------------------------------------------------------------

def fair_probs(oh, od, oa):
    """Proportional margin removal. Simple and standard, slightly
    overstates favourites' fair price compared with the Shin method."""
    raw = np.column_stack([1 / oh, 1 / od, 1 / oa])
    return raw / raw.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# 3. The model: team attack and defence ratings, updated after every match
# ----------------------------------------------------------------------------

def _poisson_1x2(lh, la, max_goals=10):
    g = np.arange(max_goals + 1)
    fact = np.array([factorial(i) for i in g], dtype=float)
    ph = np.exp(-lh) * lh ** g / fact
    pa = np.exp(-la) * la ** g / fact
    grid = np.outer(ph, pa)
    home = np.tril(grid, -1).sum()
    draw = np.trace(grid)
    away = np.triu(grid, 1).sum()
    s = home + draw + away
    return np.array([home, draw, away]) / s


class RatingModel:
    """
    log(expected home goals) = base_home + att[home] + dfn[away]
    log(expected away goals) = base_away + att[away] + dfn[home]
    After each match, nudge the four ratings involved toward what happened.
    Between seasons, pull all ratings part of the way back to zero
    (teams change). Ratings are kept per league because the scale of goals
    differs.
    """

    def __init__(self, lr=0.04, season_shrink=0.35):
        self.lr = lr
        self.season_shrink = season_shrink
        self.att = {}
        self.dfn = {}
        self.goals_h = {}
        self.goals_a = {}
        self.n = {}

    def _base(self, league):
        n = self.n.get(league, 0)
        if n < 30:  # not enough data yet, use typical values
            return np.log(1.5), np.log(1.15)
        return np.log(self.goals_h[league] / n), np.log(self.goals_a[league] / n)

    def predict(self, league, home, away):
        bh, ba = self._base(league)
        lh = np.exp(bh + self.att.get((league, home), 0) + self.dfn.get((league, away), 0))
        la = np.exp(ba + self.att.get((league, away), 0) + self.dfn.get((league, home), 0))
        return _poisson_1x2(lh, la), lh, la

    def update(self, league, home, away, hg, ag):
        _, lh, la = self.predict(league, home, away)
        eh, ea = hg - lh, ag - la  # gradient of the Poisson log-likelihood
        kh, ka = (league, home), (league, away)
        self.att[kh] = self.att.get(kh, 0) + self.lr * eh
        self.dfn[ka] = self.dfn.get(ka, 0) + self.lr * eh
        self.att[ka] = self.att.get(ka, 0) + self.lr * ea
        self.dfn[kh] = self.dfn.get(kh, 0) + self.lr * ea
        self.goals_h[league] = self.goals_h.get(league, 0) + hg
        self.goals_a[league] = self.goals_a.get(league, 0) + ag
        self.n[league] = self.n.get(league, 0) + 1

    def shrink_all(self):
        for k in self.att:
            self.att[k] *= 1 - self.season_shrink
        for k in self.dfn:
            self.dfn[k] *= 1 - self.season_shrink


def walk_forward(df, lr=0.04, season_shrink=0.35):
    """Predict every match using only matches played before it."""
    model = RatingModel(lr, season_shrink)
    # new season = a gap of more than 60 days within a league
    last_date = {}
    preds = np.zeros((len(df), 3))
    for i, r in enumerate(df.itertuples(index=False)):
        prev = last_date.get(r.league)
        if prev is not None and (r.date - prev).days > 60:
            model.shrink_all()
        last_date[r.league] = r.date
        preds[i], _, _ = model.predict(r.league, r.home, r.away)
        model.update(r.league, r.home, r.away, r.hg, r.ag)
    return preds


# ----------------------------------------------------------------------------
# 4. Scoring and betting test
# ----------------------------------------------------------------------------

def outcome_matrix(df):
    y = np.zeros((len(df), 3))
    y[df.hg.values > df.ag.values, 0] = 1
    y[df.hg.values == df.ag.values, 1] = 1
    y[df.hg.values < df.ag.values, 2] = 1
    return y


def log_loss(p, y):
    return -np.mean(np.log(np.clip((p * y).sum(axis=1), 1e-9, 1)))


def rps(p, y):
    cp, cy = np.cumsum(p, axis=1)[:, :2], np.cumsum(y, axis=1)[:, :2]
    return np.mean(((cp - cy) ** 2).sum(axis=1) / 2)


def bootstrap_roi(profits, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    if len(profits) == 0:
        return (np.nan, np.nan)
    means = [rng.choice(profits, len(profits)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def run_bets(df, model_p, fair_p, y, min_edge, price_cols=("odds_h", "odds_d", "odds_a"), label=""):
    odds = df[list(price_cols)].values
    edge = model_p * odds - 1
    pick = edge.argmax(axis=1)
    best_edge = edge[np.arange(len(df)), pick]
    bet = best_edge > min_edge
    if bet.sum() == 0:
        print(f"  {label}: no bets at edge > {min_edge:.0%}")
        return
    won = y[np.arange(len(df)), pick] == 1
    profit = np.where(won, odds[np.arange(len(df)), pick] - 1, -1.0)[bet]
    lo, hi = bootstrap_roi(profit)
    print(f"  {label}: {bet.sum()} bets, ROI {profit.mean():+.1%}  (95% range {lo:+.1%} to {hi:+.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.04)
    ap.add_argument("--burn-in-frac", type=float, default=0.25,
                    help="share of the earliest matches used only for learning ratings, not scored")
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--shrink", type=float, default=0.35)
    args = ap.parse_args()

    df = load_all()
    print(f"Matches: {len(df)} | {df.date.min().date()} to {df.date.max().date()}")
    print(df.groupby("league").size().to_string())
    print("Odds source:", ", ".join(sorted(df.odds_source.unique())))
    if df.date.dt.year.nunique() < 4:
        print("\nWARNING: fewer than 4 calendar years of data. Treat everything below as a pipeline check, "
              "not evidence of an edge.\n")

    model_p = walk_forward(df, args.lr, args.shrink)
    fair_p = fair_probs(df.odds_h.values, df.odds_d.values, df.odds_a.values)
    y = outcome_matrix(df)

    cut = int(len(df) * args.burn_in_frac)
    sl = slice(cut, None)
    d, m, f, yy = df.iloc[cut:], model_p[sl], fair_p[sl], y[sl]
    print(f"\nScored matches (after burn-in): {len(d)}")

    print("\nHow good are the probabilities? (lower is better)")
    print(f"  Model     log loss {log_loss(m, yy):.4f}   RPS {rps(m, yy):.4f}")
    print(f"  Bookmaker log loss {log_loss(f, yy):.4f}   RPS {rps(f, yy):.4f}   (margin removed)")

    # blend: shows whether the model adds anything on top of the market
    for w in (0.1, 0.25, 0.5):
        b = (1 - w) * f + w * m
        print(f"  Market + {w:.0%} model   log loss {log_loss(b, yy):.4f}")

    print(f"\nBetting test: flat 1 unit, bet only if model edge > {args.min_edge:.0%}")
    run_bets(d, m, f, yy, args.min_edge, label="at the quoted odds")
    if {"max_h", "max_d", "max_a"} <= set(d.columns):
        run_bets(d, m, f, yy, args.min_edge, ("max_h", "max_d", "max_a"), label="at best market odds")
    print("\nReminder: the quoted odds are CLOSING odds. In real life you bet earlier at different prices "
          "and the book may limit you. A model that cannot beat these odds will not beat real ones.")


if __name__ == "__main__":
    main()
