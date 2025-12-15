# Scandinavian Betting Project – Pricing Efficiency in Football Markets

This repo is a portfolio project exploring how efficiently bookmakers price core football betting markets in **Scandinavian leagues**.

It’s structured as a series of chapters, each designed to look like a realistic **sports betting analyst workflow**:
- pull data
- clean and structure it
- build pricing and accuracy metrics
- turn that into trading-style insight

So far the project includes:

- **Chapter 1 – Single-League Pricing & Calibration**  
- **Chapter 2 – Cross-League & Multi-Market Efficiency**

---

## 🎯 Project Goal

To demonstrate how I’d approach a **Data / Trading Analyst role in sports betting**, using:

- Realistic league and odds data  
- A reproducible **Python + SQLite** pipeline  
- Proper **scoring rules** (calibration curves, MACE, Brier score)  
- Clear **visual storytelling** suitable for trading & risk stakeholders  

The focus is on football (soccer) pre-match markets, starting with **1X2** and **BTTS**.

---

## 🧱 Data & Scope

- **Leagues**
  - Norway – *Eliteserien*
  - Sweden – *Allsvenskan*
  - Denmark – *Superliga*

- **Season**
  - 2025 (current or most recent available season at time of export)

- **Markets analysed so far**
  - **1X2** (match result)
  - **BTTS** (Both Teams to Score – Yes/No)

- **Data source**
  - CSV exports from **FootyStats** (match results + pre-match odds)

- **Odds**
  - Decimal, **pre-match closing prices** from static CSV snapshots  
  - No line-movement or in-play prices yet – all analysis is based on the final quoted pre-match odds.

> 💡 Note: The 2025 Danish Superliga season was still in progress at the time of export. The analysis includes all completed matches and pre-match odds available at that date.

---

## 📘 Chapter 1 – Single-League Pricing & Calibration (Eliteserien 2025)

**Core question**

> *How efficiently do bookmakers price 1X2 home wins in one league?*

### Workflow

- Load **Norway Eliteserien 2025** match + odds data into **SQLite** (`data/processed/scandi.db`).
- Build an analysis-ready table with:
  - Match result (home / draw / away)
  - Binary home-win outcome
  - Implied probabilities from 1X2 odds
  - **Overround** (bookmaker margin) on the 1X2 book

- Visuals (in `notebooks/scandi_chapter1.ipynb`):
  - Distribution of 1X2 overround across matches
  - Average 1X2 margin for the league
  - Basic **calibration curve** for home-win probabilities vs actual home-win rates

### What Chapter 1 shows

- Understanding of:
  - Odds → implied probabilities  
  - Overround and margin construction  
  - Calibration as “do these prices behave like probabilities?”

- Ability to:
  - Clean and structure raw odds data  
  - Produce **league-level pricing diagnostics** for a single market  
  - Present findings in a deck for non-technical stakeholders  
    - `presentations/Scandinavian Betting Portfolio Project - Chapter 1.pptx`

---

## 📗 Chapter 2 – Cross-League & Multi-Market Efficiency  
*(Norway, Sweden, Denmark – 1X2 & BTTS)*

**Core question**

> *How do pricing, margins and forecast accuracy vary across leagues and between 1X2 and BTTS?*

Chapter 2 scales the framework from Chapter 1:

- From **one league → three leagues**
- From **one market → two markets (1X2 & BTTS)**

Code + analysis are in:

- `notebooks/Scandi_Chapter_2.ipynb`
- `src/scandi_chapter2.py`

### 1. League & Market Coverage

- Validate match counts and odds coverage for:
  - Allsvenskan (Sweden) – 239 matches – 1X2, BTTS  
  - Eliteserien (Norway) – 239 matches – 1X2, BTTS  
  - Superliga (Denmark) – 108 matches – 1X2, BTTS  

- Construct a unified dataframe with:

  ```text
  league | season | home_goals | away_goals
         | odds_home | odds_draw | odds_away
         | odds_btts_yes | odds_btts_no
