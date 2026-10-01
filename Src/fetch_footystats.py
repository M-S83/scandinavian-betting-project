"""
Download every available season of Danish Superliga, Norwegian Eliteserien
and Swedish Allsvenskan from the FootyStats API and save one CSV per season
with the same columns as Data/Processed/Denmark2025.csv.

Run:
    export FOOTYSTATS_KEY="..."              # never commit or print this
    python Src/fetch_footystats.py --dry-run # 1 call: show plan limit, seasons, calls needed
    python Src/fetch_footystats.py           # fetch, save CSVs, print shots/xG coverage
    python Src/fetch_footystats.py --refresh # re-download seasons already cached

Keeping the call count low:
    - 1 call for the league list (it also reports the plan's request limit)
    - 1 call per season (more only if a season has more than 1000 matches)
    - Raw JSON is cached in Data/Raw/footystats/_cache/ (git-ignored). Finished
      seasons are never fetched twice; only each league's latest season is
      refreshed on a re-run, because it may still be in progress.
    - The script stops before fetching if the plan doesn't have enough calls left.

Output:
    Data/Raw/footystats/<Country>_<season>.csv   e.g. Denmark_2024-2025.csv, Sweden_2023.csv
    Data/Raw/footystats/api_fields.txt           every field name the API returned,
                                                 for checking the column mapping
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Data", "Raw", "footystats")
CACHE_DIR = os.path.join(OUT_DIR, "_cache")
TEMPLATE = os.path.join(ROOT, "Data", "Processed", "Denmark2025.csv")
BASE_URL = "https://api.football-data-api.com"

# country -> word that must appear in the league name
LEAGUES = {"Denmark": "Superliga", "Norway": "Eliteserien", "Sweden": "Allsvenskan"}


# ----------------------------------------------------------------------------
# 1. API access. The key only ever lives in memory: it is never printed,
#    logged, put in an error message or written to a cache file.
# ----------------------------------------------------------------------------

class ApiError(Exception):
    pass


def _key():
    key = os.environ.get("FOOTYSTATS_KEY", "").strip()
    if not key:
        raise SystemExit("Set the FOOTYSTATS_KEY environment variable first.")
    return key


def _scrub(text, key):
    return text.replace(key, "***") if key else text


class Api:
    def __init__(self, key):
        self.key = key
        self.calls = 0
        self.metadata = {}

    def get(self, endpoint, **params):
        query = urllib.parse.urlencode({"key": self.key, **params})
        url = f"{BASE_URL}/{endpoint}?{query}"
        shown = f"/{endpoint} {params}"  # what we print instead of the URL
        for attempt in range(3):
            try:
                self.calls += 1
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = r.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = _scrub(e.read().decode("utf-8", "replace")[:300], self.key)
                if e.code == 429 or e.code >= 500:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                raise ApiError(f"{shown}: HTTP {e.code} {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                reason = _scrub(str(getattr(e, "reason", e)), self.key)
                raise ApiError(f"{shown}: network error {reason}") from None
        body = _scrub(body, self.key)
        data = json.loads(body)
        if isinstance(data, dict):
            if isinstance(data.get("metadata"), dict):
                self.metadata = data["metadata"]
            if data.get("success") is False:
                raise ApiError(f"{shown}: {data.get('message') or data.get('error') or body[:300]}")
        return data

    def remaining(self):
        try:
            return int(self.metadata.get("request_remaining"))
        except (TypeError, ValueError):
            return None


# ----------------------------------------------------------------------------
# 2. Leagues and seasons
# ----------------------------------------------------------------------------

def season_label(year):
    """20242025 -> '2024-2025', 2025 -> '2025'."""
    s = str(year)
    return f"{s[:4]}-{s[4:]}" if len(s) == 8 else s


def find_seasons(league_list):
    """Return [(country, league_name, season_id, label)] for the three leagues."""
    rows = league_list.get("data", []) if isinstance(league_list, dict) else league_list
    found = []
    for country, word in LEAGUES.items():
        hits = [lg for lg in rows
                if str(lg.get("country", "")).lower() == country.lower()
                and word.lower() in str(lg.get("name", "") + " " + str(lg.get("league_name", ""))).lower()]
        if not hits:
            print(f"  {country}: no league containing '{word}' in the league list")
            continue
        for lg in hits:
            for s in lg.get("season", []) or []:
                found.append((country, lg.get("name"), int(s["id"]), season_label(s.get("year"))))
    return sorted(found, key=lambda t: (t[0], t[3]))


def cache_path(season_id, page):
    return os.path.join(CACHE_DIR, f"season_{season_id}_p{page}.json")


def fetch_season(api, season_id, use_cache):
    """All matches for one season, following the pager. Uses the cache if allowed."""
    matches, page, max_page = [], 1, 1
    while page <= max_page:
        path = cache_path(season_id, page)
        if use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = api.get("league-matches", season_id=season_id, max_per_page=1000, page=page)
            data.pop("metadata", None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        matches.extend(data.get("data") or [])
        pager = data.get("pager") or {}
        try:
            max_page = int(pager.get("max_page", 1))
        except (TypeError, ValueError):
            max_page = 1
        page += 1
    return matches


# ----------------------------------------------------------------------------
# 3. API match JSON -> the FootyStats CSV export columns
# ----------------------------------------------------------------------------

def _num(v):
    """Numbers as float; FootyStats uses -1 (and sometimes '') for 'no data'."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if x == -1 else x


def _first(m, keys, conv=_num):
    for k in keys:
        if k in m and m[k] is not None and m[k] != "":
            return conv(m[k])
    return np.nan


def _timings(v):
    if isinstance(v, list):
        return ",".join(str(x).replace("+", "'") for x in v)
    return "" if v is None else str(v)


def _date_gmt(ts):
    if pd.isna(ts):
        return ""
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    hour = dt.hour % 12 or 12
    return f"{dt:%b} {dt.day} {dt.year} - {hour}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"


def _sum(a, b):
    return a + b if not (pd.isna(a) or pd.isna(b)) else np.nan


# CSV column -> API field names to try, in order
NUMERIC = {
    "Game Week": ["game_week"],
    "Pre-Match PPG (Home)": ["pre_match_home_ppg", "pre_match_teamA_overall_ppg"],
    "Pre-Match PPG (Away)": ["pre_match_away_ppg", "pre_match_teamB_overall_ppg"],
    "home_ppg": ["home_ppg"],
    "away_ppg": ["away_ppg"],
    "home_team_goal_count": ["homeGoalCount"],
    "away_team_goal_count": ["awayGoalCount"],
    "total_goal_count": ["totalGoalCount"],
    "home_team_goal_count_half_time": ["ht_goals_team_a"],
    "away_team_goal_count_half_time": ["ht_goals_team_b"],
    "total_goals_at_half_time": ["HTGoalCount"],
    "home_team_corner_count": ["team_a_corners"],
    "away_team_corner_count": ["team_b_corners"],
    "home_team_yellow_cards": ["team_a_yellow_cards"],
    "home_team_red_cards": ["team_a_red_cards"],
    "away_team_yellow_cards": ["team_b_yellow_cards"],
    "away_team_red_cards": ["team_b_red_cards"],
    "home_team_first_half_cards": ["team_a_fh_cards"],
    "home_team_second_half_cards": ["team_a_2h_cards"],
    "away_team_first_half_cards": ["team_b_fh_cards"],
    "away_team_second_half_cards": ["team_b_2h_cards"],
    "home_team_shots": ["team_a_shots"],
    "away_team_shots": ["team_b_shots"],
    "home_team_shots_on_target": ["team_a_shotsOnTarget"],
    "away_team_shots_on_target": ["team_b_shotsOnTarget"],
    "home_team_shots_off_target": ["team_a_shotsOffTarget"],
    "away_team_shots_off_target": ["team_b_shotsOffTarget"],
    "home_team_fouls": ["team_a_fouls"],
    "away_team_fouls": ["team_b_fouls"],
    "home_team_possession": ["team_a_possession"],
    "away_team_possession": ["team_b_possession"],
    "Home Team Pre-Match xG": ["team_a_xg_prematch"],
    "Away Team Pre-Match xG": ["team_b_xg_prematch"],
    "team_a_xg": ["team_a_xg"],
    "team_b_xg": ["team_b_xg"],
    "average_goals_per_match_pre_match": ["avg_potential"],
    "btts_percentage_pre_match": ["btts_potential"],
    "over_15_percentage_pre_match": ["o15_potential"],
    "over_25_percentage_pre_match": ["o25_potential"],
    "over_35_percentage_pre_match": ["o35_potential"],
    "over_45_percentage_pre_match": ["o45_potential"],
    "over_15_HT_FHG_percentage_pre_match": ["o15HT_potential"],
    "over_05_HT_FHG_percentage_pre_match": ["o05HT_potential"],
    "over_15_2HG_percentage_pre_match": ["o15_2H_potential"],
    "over_05_2HG_percentage_pre_match": ["o05_2H_potential"],
    "average_corners_per_match_pre_match": ["corners_potential"],
    "average_cards_per_match_pre_match": ["cards_potential"],
    "odds_ft_home_team_win": ["odds_ft_1"],
    "odds_ft_draw": ["odds_ft_x"],
    "odds_ft_away_team_win": ["odds_ft_2"],
    "odds_ft_over15": ["odds_ft_over15"],
    "odds_ft_over25": ["odds_ft_over25"],
    "odds_ft_over35": ["odds_ft_over35"],
    "odds_ft_over45": ["odds_ft_over45"],
    "odds_btts_yes": ["odds_btts_yes"],
    "odds_btts_no": ["odds_btts_no"],
}


def to_row(m):
    ts = _first(m, ["date_unix"])
    row = {
        "timestamp": int(ts) if not pd.isna(ts) else np.nan,
        "date_GMT": _date_gmt(ts),
        "status": m.get("status", ""),
        "attendance": _first(m, ["attendance"], str) if _num(m.get("attendance")) > 0 else "N/A",
        "home_team_name": _first(m, ["home_name", "homeName"], str),
        "away_team_name": _first(m, ["away_name", "awayName"], str),
        "referee": _first(m, ["referee_name", "referee"], str) if m.get("referee_name") or m.get("referee") else "N/A",
        "home_team_goal_timings": _timings(m.get("homeGoals")),
        "away_team_goal_timings": _timings(m.get("awayGoals")),
        "stadium_name": _first(m, ["stadium_name"], str),
    }
    for col, keys in NUMERIC.items():
        row[col] = _first(m, keys)
    if pd.isna(row["total_goals_at_half_time"]):
        row["total_goals_at_half_time"] = _sum(row["home_team_goal_count_half_time"],
                                               row["away_team_goal_count_half_time"])
    if pd.isna(row["total_goal_count"]):
        row["total_goal_count"] = _sum(row["home_team_goal_count"], row["away_team_goal_count"])
    return row


def to_frame(matches, columns):
    df = pd.DataFrame([to_row(m) for m in matches])
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns].sort_values("timestamp", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Coverage report
# ----------------------------------------------------------------------------

def _filled(df, a, b):
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    return (x > 0) & (y > 0)


def coverage(df):
    done = df[df["status"] == "complete"]
    shots = _filled(done, "home_team_shots", "away_team_shots")
    xg = _filled(done, "team_a_xg", "team_b_xg")
    return {"matches": len(df), "complete": len(done), "shots": int(shots.sum()),
            "xg": int(xg.sum()), "both": int((shots & xg).sum())}


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="only fetch the league list (1 call), show seasons and calls needed")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download every season, ignoring the cache")
    args = ap.parse_args()

    columns = list(pd.read_csv(TEMPLATE, nrows=0).columns)
    os.makedirs(CACHE_DIR, exist_ok=True)
    api = Api(_key())

    try:
        league_list = api.get("league-list")
    except ApiError as e:
        raise SystemExit(f"League list failed: {e}")
    meta = api.metadata
    print(f"Plan: request_limit={meta.get('request_limit', '?')}, "
          f"request_remaining={meta.get('request_remaining', '?')}"
          + (f" ({meta['request_reset_message']})" if meta.get("request_reset_message") else ""))

    seasons = find_seasons(league_list)
    if not seasons:
        raise SystemExit("None of the three leagues were found in the league list.")
    latest = {}
    for country, _, sid, label in seasons:
        latest[country] = max(latest.get(country, (label, sid)), (label, sid))
    latest_ids = {sid for _, sid in latest.values()}

    def cached(sid):
        return not args.refresh and sid not in latest_ids and os.path.exists(cache_path(sid, 1))

    to_fetch = [s for s in seasons if not cached(s[2])]
    print(f"Seasons found: {len(seasons)}. To download: {len(to_fetch)} "
          f"(about {len(to_fetch)} calls). Cached: {len(seasons) - len(to_fetch)}.")
    for country, name, sid, label in seasons:
        print(f"  {country:8} {name}  {label:10} season_id={sid}  {'cached' if cached(sid) else 'fetch'}")

    remaining = api.remaining()
    if remaining is not None and remaining < len(to_fetch):
        raise SystemExit(f"Only {remaining} calls left on the plan, need about {len(to_fetch)}. "
                         "Wait for the limit to reset and re-run (cached seasons are kept).")
    if args.dry_run:
        return

    fields, report = set(), []
    for country, name, sid, label in seasons:
        try:
            matches = fetch_season(api, sid, use_cache=cached(sid))
        except ApiError as e:
            print(f"  {country} {label}: FAILED ({e})")
            report.append((country, label, None))
            continue
        for m in matches:
            fields.update(m.keys())
        if not matches:
            print(f"  {country} {label}: no matches returned")
            report.append((country, label, None))
            continue
        df = to_frame(matches, columns)
        out = os.path.join(OUT_DIR, f"{country}_{label}.csv")
        df.to_csv(out, index=False)
        report.append((country, label, coverage(df)))

    if fields:
        with open(os.path.join(OUT_DIR, "api_fields.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(fields)) + "\n")
        missing = sorted(c for c, keys in NUMERIC.items() if not any(k in fields for k in keys))
        if missing:
            print("\nCSV columns left empty because the API had none of the expected fields:")
            print("  " + ", ".join(missing))
            print("  (all field names the API returned are in Data/Raw/footystats/api_fields.txt)")

    print(f"\nAPI calls made this run: {api.calls}. Remaining on plan: {api.metadata.get('request_remaining', '?')}")
    print("\nShots and xG coverage (completed matches where both teams' value is > 0)")
    print(f"{'League':8} {'Season':10} {'Matches':>7} {'Complete':>8} {'Shots':>6} {'xG':>6} {'Both':>6}")
    for country, label, c in report:
        if c is None:
            print(f"{country:8} {label:10} {'-':>7}")
            continue
        print(f"{country:8} {label:10} {c['matches']:>7} {c['complete']:>8} "
              f"{c['shots']:>6} {c['xg']:>6} {c['both']:>6}")


if __name__ == "__main__":
    main()
