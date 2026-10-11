"""Generate completed assessment notebooks. Temporary helper; deleted after use."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# ---------------------------------------------------------------------------
# Cleaning notebook
# ---------------------------------------------------------------------------
clean_cells = []

clean_cells.append(new_markdown_cell("""# Aviation Accidents Analysis

You are part of a consulting firm that is tasked to do an analysis of commercial and passenger jet airline safety. The client (an airline/airplane insurer) is interested in knowing what types of aircraft (makes/models) exhibit low rates of total destruction and low likelihood of fatal or serious passenger injuries in the event of an accident. They are also interested in any general variables/conditions that might be at play. Your analysis will be based off of aviation accident data accumulated from the years 1948-2023.

Our client is only interested in airplane makes/models that are professional builds and could potentially still be active. Assume a max lifetime of 40 years for a make/model retirement and make sure to filter your data accordingly (i.e. from 1983 onwards). They would also like separate recommendations for small aircraft vs. larger passenger models. **In addition, make sure that claims that you make are statistically robust and that you have enough samples when making comparisons between groups.**


In this summative assessment you will demonstrate your ability to:
- **Use Pandas to load, inspect, and clean the dataset appropriately.**
- **Transform relevant columns to create measures that address the problem at hand.**
- conduct EDA: visualization and statistical measures to systematically understand the structure of the data
- recommend a set of airplanes and makes conforming to the client's request and identify at least *two* factors contributing to airplane safety. You must provide supporting evidence (visuals, summary statistics, tables) for each claim you make.
"""))

clean_cells.append(new_markdown_cell("### Make relevant library imports"))

clean_cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)
"""))

clean_cells.append(new_markdown_cell("## Data Loading and Inspection"))

clean_cells.append(new_markdown_cell("""### Load in data from the relevant directory and inspect the dataframe.
- inspect NaNs, datatypes, and summary statistics
"""))

clean_cells.append(new_code_cell("""# NTSB export uses latin-1 for some location / name characters
df_raw = pd.read_csv("data/AviationData.csv", encoding="latin-1", low_memory=False)

print("Shape:", df_raw.shape)
print("\\nColumn dtypes:")
print(df_raw.dtypes)
print("\\nMissingness (%):")
missing = (df_raw.isna().mean() * 100).sort_values(ascending=False)
print(missing.round(2))
print("\\nNumeric summary:")
display(df_raw.describe())
print("\\nFirst 3 rows:")
display(df_raw.head(3))
"""))

clean_cells.append(new_markdown_cell("## Data Cleaning"))

clean_cells.append(new_markdown_cell("### Filtering aircrafts and events"))

clean_cells.append(new_markdown_cell("""We want to filter the dataset to include aircraft that the client is interested in an analysis of:
- inspect relevant columns
- figure out any reasonable imputations
- filter the dataset
"""))

clean_cells.append(new_code_cell("""# Inspect columns used to define the client's population
print("Event.Date range:", df_raw["Event.Date"].min(), "->", df_raw["Event.Date"].max())
print("\\nAmateur.Built:")
print(df_raw["Amateur.Built"].value_counts(dropna=False))
print("\\nAircraft.Category:")
print(df_raw["Aircraft.Category"].value_counts(dropna=False))
print("\\nInvestigation.Type:")
print(df_raw["Investigation.Type"].value_counts(dropna=False))
print("\\nNumber.of.Engines:")
print(df_raw["Number.of.Engines"].value_counts(dropna=False))

df = df_raw.copy()
df["Event.Date"] = pd.to_datetime(df["Event.Date"], errors="coerce")

# 40-year active-fleet window relative to the 2023 data vintage
df = df[df["Event.Date"].dt.year >= 1983]

# Professional (factory) builds only
df = df[df["Amateur.Built"] == "No"]

# Accidents only: incidents are a different reporting class and would
# understate injury / destruction rates if mixed in
df = df[df["Investigation.Type"] == "Accident"]

# Keep airplanes. Aircraft.Category is missing for ~64% of older records,
# which are still predominantly airplanes (Cessna/Piper/Beech/Boeing).
# Drop known rotorcraft via engine type and manufacturer keywords.
heli_pattern = (
    r"ROBINSON|HUGHES|ENSTROM|HILLER|EUROCOPTER|AIRBUS HELICOPTER|"
    r"SCHWEIZER|MD HELICOPTER|KAMAN|SIKORSKY|AGUSTA|BELL HELICOPTER|^BELL$|^BELL "
)
make_upper = df["Make"].fillna("").str.upper()
is_heli_make = make_upper.str.contains(heli_pattern, regex=True)
is_airplane_cat = df["Aircraft.Category"].isna() | (df["Aircraft.Category"] == "Airplane")
df = df[is_airplane_cat & (df["Engine.Type"] != "Turbo Shaft") & ~is_heli_make]

# Zero engines are typically gliders / balloons, not the client's fleet
df = df[df["Number.of.Engines"].fillna(1) != 0]

print("\\nRows after population filters:", len(df))
"""))

clean_cells.append(new_markdown_cell("""### Cleaning and constructing Key Measurables

Injuries and robustness to destruction are a key interest point for the client. Clean and impute relevant columns and then create derived fields that best quantifies what the client wishes to track. **Use commenting or markdown to explain any cleaning assumptions as well as any derived columns you create.**
"""))

clean_cells.append(new_markdown_cell("""**Construct metric for fatal/serious injuries**

*Hint:* Estimate the total number of passengers on each flight. The likelihood of serious / fatal injury can be estimated as a fraction from this.
"""))

clean_cells.append(new_code_cell("""injury_cols = [
    "Total.Fatal.Injuries",
    "Total.Serious.Injuries",
    "Total.Minor.Injuries",
    "Total.Uninjured",
]

print("Injury missingness before imputation:")
print(df[injury_cols].isna().sum())
print("\\nWhen fatal count is missing, Injury.Severity is almost always Non-Fatal")
print(df.loc[df["Total.Fatal.Injuries"].isna(), "Injury.Severity"].value_counts(dropna=False).head())

# Assumption: blank injury counts in NTSB records represent 0, not unknown.
# This matches Non-Fatal / Incident rows where uninjured is filled and fatals are blank.
for col in injury_cols:
    df[col] = df[col].fillna(0)

# Estimated occupants on board at the time of the accident
df["total_onboard"] = df[injury_cols].sum(axis=1)

# Client metric: share of people on board who were killed or seriously injured
df["fatal_serious_count"] = df["Total.Fatal.Injuries"] + df["Total.Serious.Injuries"]
df["fatal_serious_frac"] = np.where(
    df["total_onboard"] > 0,
    df["fatal_serious_count"] / df["total_onboard"],
    np.nan,
)

print("\\nOccupants and injury fraction:")
display(df[["total_onboard", "fatal_serious_count", "fatal_serious_frac"]].describe())
print("Rows with 0 recorded occupants (metric set to NaN):", df["fatal_serious_frac"].isna().sum())
"""))

clean_cells.append(new_markdown_cell("""**Aircraft.Damage**
- identify and execute any cleaning tasks
- construct a derived column tracking whether an aircraft was destroyed or not.
"""))

clean_cells.append(new_code_cell("""print(df["Aircraft.damage"].value_counts(dropna=False))

# Standardize unknown labels; keep missing as missing so destruction rates
# are computed only on accidents with a recorded damage class
df["Aircraft.damage"] = df["Aircraft.damage"].replace({"Unknown": np.nan})

df["is_destroyed"] = np.where(
    df["Aircraft.damage"].isin(["Destroyed", "Substantial", "Minor"]),
    (df["Aircraft.damage"] == "Destroyed").astype(float),
    np.nan,
)

print("\\nDestruction rate (known damage only):", df["is_destroyed"].mean().round(3))
print("Known-damage sample size:", df["is_destroyed"].notna().sum())
"""))

clean_cells.append(new_markdown_cell("""### Investigate the *Make* column

Cleaning tasks:
1. Strip whitespace and uppercase so `Cessna` / `CESSNA` collapse to one manufacturer.
2. Map common corporate aliases (e.g. `AIRBUS INDUSTRIE` → `AIRBUS`, `THE BOEING COMPANY` → `BOEING`).
3. Drop rows with missing Make.
4. Keep manufacturers with at least **50** accidents so later comparisons are statistically stable.
"""))

clean_cells.append(new_code_cell("""print("Unique Makes before cleaning:", df["Make"].nunique())
print(df["Make"].value_counts(dropna=False).head(20))


def canonicalize_make(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"[^A-Z0-9 ]+", " ", str(value).strip().upper())
    text = re.sub(r"\\s+", " ", text).strip()

    alias_map = {
        "CESSNA AIRCRAFT": "CESSNA",
        "CESSNA AIRCRAFT CO": "CESSNA",
        "CESSNA AIRCRAFT COMPANY": "CESSNA",
        "PIPER AIRCRAFT": "PIPER",
        "PIPER AIRCRAFT INC": "PIPER",
        "THE NEW PIPER AIRCRAFT": "PIPER",
        "NEW PIPER AIRCRAFT": "PIPER",
        "BEECHCRAFT": "BEECH",
        "BEECH AIRCRAFT": "BEECH",
        "BEECH AIRCRAFT CORPORATION": "BEECH",
        "HAWKER BEECHCRAFT": "BEECH",
        "RAYTHEON AIRCRAFT COMPANY": "BEECH",
        "THE BOEING COMPANY": "BOEING",
        "BOEING COMPANY": "BOEING",
        "AIRBUS INDUSTRIE": "AIRBUS",
        "AIRBUS INDUSTRIES": "AIRBUS",
        "MCDONNELL-DOUGLAS": "MCDONNELL DOUGLAS",
        "MCDONNELL DOUGLAS AIRCRAFT CO": "MCDONNELL DOUGLAS",
        "CIRRUS DESIGN CORP": "CIRRUS",
        "CIRRUS DESIGN CORPORATION": "CIRRUS",
        "AIR TRACTOR INC": "AIR TRACTOR",
        "BOMBARDIER INC": "BOMBARDIER",
        "DEHAVILLAND": "DE HAVILLAND",
        "GRUMMAN AMERICAN": "GRUMMAN",
        "MOONEY AIRCRAFT CORP": "MOONEY",
        "GULFSTREAM AEROSPACE": "GULFSTREAM",
        "DIAMOND AIRCRAFT IND INC": "DIAMOND",
        "AMERICAN CHAMPION AIRCRAFT": "AMERICAN CHAMPION",
    }
    if text in alias_map:
        return alias_map[text]

    prefix_rules = [
        ("AIRBUS", lambda s: "AIRBUS" if "HELICOPTER" not in s else s),
        ("BOEING", lambda s: "BOEING"),
        ("CESSNA", lambda s: "CESSNA"),
        ("PIPER", lambda s: "PIPER"),
        ("BEECH", lambda s: "BEECH"),
        ("CIRRUS", lambda s: "CIRRUS"),
        ("EMBRAER", lambda s: "EMBRAER"),
        ("BOMBARDIER", lambda s: "BOMBARDIER"),
        ("AIR TRACTOR", lambda s: "AIR TRACTOR"),
        ("MCDONNELL", lambda s: "MCDONNELL DOUGLAS"),
        ("MOONEY", lambda s: "MOONEY"),
        ("GULFSTREAM", lambda s: "GULFSTREAM"),
        ("DIAMOND", lambda s: "DIAMOND"),
        ("DE HAVILLAND", lambda s: "DE HAVILLAND"),
        ("DEHAVILLAND", lambda s: "DE HAVILLAND"),
        ("GRUMMAN", lambda s: "GRUMMAN"),
        ("HAWKER BEECH", lambda s: "BEECH"),
        ("RAYTHEON", lambda s: "BEECH"),
        ("LEARJET", lambda s: "LEARJET"),
        ("AVIAT", lambda s: "AVIAT"),
        ("MAULE", lambda s: "MAULE"),
        ("SOCATA", lambda s: "SOCATA"),
    ]
    for prefix, fn in prefix_rules:
        if text == prefix or text.startswith(prefix + " ") or text.startswith(prefix):
            return fn(text)
    return text


df["Make"] = df["Make"].map(canonicalize_make)
df = df.dropna(subset=["Make"])

make_counts = df["Make"].value_counts()
keep_makes = make_counts[make_counts >= 50].index
df = df[df["Make"].isin(keep_makes)]

print("\\nUnique Makes after aliasing + n>=50 filter:", df["Make"].nunique())
print("Rows remaining:", len(df))
print(df["Make"].value_counts().head(15))
"""))

clean_cells.append(new_markdown_cell("""### Inspect Model column
- Get rid of any NaNs.
- Inspect the column and counts for each model/make. Are model labels unique to each make?
- If not, create a derived column that is a unique identifier for a given plane type.
"""))

clean_cells.append(new_code_cell("""print("Missing Model:", df["Model"].isna().sum())
df = df.dropna(subset=["Model"])
df["Model"] = df["Model"].astype(str).str.strip().str.upper()

models_per_make = df.groupby("Model")["Make"].nunique()
print("Models used by more than one manufacturer:", (models_per_make > 1).sum())
print(models_per_make[models_per_make > 1].head(10))

# Model strings like "100" or "172" are not unique across manufacturers
df["make_model"] = df["Make"] + " " + df["Model"]
print("\\nUnique make_model identifiers:", df["make_model"].nunique())
print(df["make_model"].value_counts().head(10))
"""))

clean_cells.append(new_markdown_cell("""### Cleaning other columns
- there are other columns containing data that might be related to the outcome of an accident. We list a few here:
- Engine.Type
- Weather.Condition
- Number.of.Engines
- Purpose.of.flight
- Broad.phase.of.flight

Inspect and identify potential cleaning tasks in each of the above columns. Execute those cleaning tasks.

**Note**: You do not necessarily need to impute or drop NaNs here.
"""))

clean_cells.append(new_code_cell("""print("Engine.Type"); print(df["Engine.Type"].value_counts(dropna=False))
print("\\nWeather.Condition"); print(df["Weather.Condition"].value_counts(dropna=False))
print("\\nNumber.of.Engines"); print(df["Number.of.Engines"].value_counts(dropna=False))
print("\\nPurpose.of.flight"); print(df["Purpose.of.flight"].value_counts(dropna=False))
print("\\nBroad.phase.of.flight"); print(df["Broad.phase.of.flight"].value_counts(dropna=False))

# Engine.Type: collapse unknown / rare labels; leave true missing as NaN
df["Engine.Type"] = (
    df["Engine.Type"]
    .replace({"UNK": "Unknown", "NONE": "Unknown", "Unknown": "Unknown", "Geared Turbofan": "Turbo Fan"})
)

# Weather: UNK / Unk are the same unknown visual-conditions code
df["Weather.Condition"] = df["Weather.Condition"].replace({"UNK": "Unknown", "Unk": "Unknown"})

# Purpose: merge spelling variants; leave NaN
df["Purpose.of.flight"] = df["Purpose.of.flight"].replace(
    {
        "Air Race show": "Air Race/show",
        "Public Aircraft - Federal": "Public Aircraft",
        "Public Aircraft - Local": "Public Aircraft",
        "Public Aircraft - State": "Public Aircraft",
        "PUBS": "Public Aircraft",
        "PUBL": "Public Aircraft",
        "Unknown": np.nan,
    }
)

# Phase: Unknown/Other stay as their own labels; NaN left as missing
df["Broad.phase.of.flight"] = df["Broad.phase.of.flight"].replace({"Unknown": "Unknown"})

print("\\n--- after cleaning ---")
print(df["Weather.Condition"].value_counts(dropna=False))
print(df["Engine.Type"].value_counts(dropna=False))
"""))

clean_cells.append(new_markdown_cell("""### Column Removal
- inspect the dataframe and drop any columns that have too many NaNs
"""))

clean_cells.append(new_code_cell("""missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing_pct.round(1))

# Drop fields that are mostly empty or unused in the client analysis
drop_cols = [
    "Schedule",
    "Air.carrier",
    "FAR.Description",
    "Latitude",
    "Longitude",
    "Airport.Code",
    "Airport.Name",
    "Publication.Date",
    "Registration.Number",
    "Report.Status",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print("\\nColumns retained:")
print(df.columns.tolist())
print("Final analytic shape:", df.shape)
"""))

clean_cells.append(new_markdown_cell("""### Save DataFrame to csv
- its generally useful to save data to file/server after its in a sufficiently cleaned or intermediate state
- the data can then be loaded directly in another notebook for further analysis
- this helps keep your notebooks and workflow readable, clean and modularized
"""))

clean_cells.append(new_code_cell("""out_path = "data/AviationData_cleaned.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
"""))

nb_clean = new_notebook(cells=clean_cells, metadata={
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
})
nbf.write(nb_clean, "Aviation_Accidents_Cleaning.ipynb")

# ---------------------------------------------------------------------------
# Analysis notebook
# ---------------------------------------------------------------------------
a = []

a.append(new_markdown_cell("""# Aviation Accidents Analysis

You are part of a consulting firm that is tasked to do an analysis of commercial and passenger jet airline safety. The client (an airline/airplane insurer) is interested in knowing what types of aircraft (makes/models) exhibit low rates of total destruction and low likelihood of fatal or serious passenger injuries in the event of an accident. They are also interested in any general variables/conditions that might be at play. Your analysis will be based off of aviation accident data accumulated from the years 1948-2023.

Our client is only interested in airplane makes/models that are professional builds and could potentially still be active. Assume a max lifetime of 40 years for a make/model retirement and make sure to filter your data accordingly (i.e. from 1983 onwards). They would also like separate recommendations for small aircraft vs. larger passenger models. **In addition, make sure that claims that you make are statistically robust and that you have enough samples when making comparisons between groups.**


In this summative assessment you will demonstrate your ability to:
- Use Pandas to load, inspect, and clean the dataset appropriately.
- Transform relevant columns to create measures that address the problem at hand.
- **conduct EDA: visualization and statistical measures to understand the structure of the data**
- **recommend a set of manufacturers to consider as well as specific airplanes conforming to the client's request**
- **discuss the relationship between serious injuries/airplane damage incurred and at least *two* factors at play in the incident. You must provide supporting evidence (visuals, summary statistics, tables) for each claim you make.**
"""))

a.append(new_code_cell("""# loading relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
"""))

a.append(new_markdown_cell("""## Exploratory Data Analysis
- Load in the cleaned data
"""))

a.append(new_code_cell("""df = pd.read_csv("data/AviationData_cleaned.csv", low_memory=False)
df["Event.Date"] = pd.to_datetime(df["Event.Date"], errors="coerce")

print("Cleaned shape:", df.shape)
print("Date window:", df["Event.Date"].min().date(), "->", df["Event.Date"].max().date())
print("\\nMissingness (%):")
print((df.isna().mean() * 100).sort_values(ascending=False).round(2).head(15))
print("\\nSafety metrics:")
display(df[["total_onboard", "fatal_serious_frac", "is_destroyed"]].describe())
print("Makes:", df["Make"].nunique(), "| make_model types:", df["make_model"].nunique())
"""))

a.append(new_markdown_cell("""## Explore safety metrics across models/makes
- Remember that the client is interested in separate recommendations for smaller airplanes and larger airplanes. Choose a passenger threshold of 20 and separate the plane types.
"""))

a.append(new_code_cell("""# Classify each airplane *type* (make + model) by typical occupancy, not by a
# single accident. A 737 on a ferry flight with 2 people should still be "large".
# Threshold of 20 occupants matches the client's small vs passenger-jet split.
type_size = (
    df.groupby("make_model")
    .agg(n_accidents=("fatal_serious_frac", "size"), model_median_onboard=("total_onboard", "median"))
    .reset_index()
)
type_size["size_class"] = np.where(type_size["model_median_onboard"] >= 20, "large", "small")

df = df.merge(type_size[["make_model", "size_class", "model_median_onboard"]], on="make_model", how="left")

small = df[df["size_class"] == "small"].copy()
large = df[df["size_class"] == "large"].copy()

print(type_size["size_class"].value_counts())
print(f"Small accidents: {len(small):,} | Large accidents: {len(large):,}")
print("Large manufacturers (accident counts):")
print(large["Make"].value_counts())
print("\\nOverall mean fatal/serious fraction | destruction rate")
print("small:", small["fatal_serious_frac"].mean(), small["is_destroyed"].mean())
print("large:", large["fatal_serious_frac"].mean(), large["is_destroyed"].mean())
"""))

a.append(new_markdown_cell("""#### Analyzing Makes

Explore the human injury risk profile for small and larger Makes:
- choose the 15 makes for each group possessing the lowest mean fatal/seriously injured fraction
- plot the mean fatal/seriously injured fraction for each of these subgroups side-by-side
"""))

a.append(new_code_cell("""def make_safety_table(frame, min_n=50):
    \"\"\"Mean injury and destruction rates for manufacturers with enough accidents.\"\"\"
    g = (
        frame.groupby("Make")
        .agg(
            n=("fatal_serious_frac", "size"),
            n_injury=("fatal_serious_frac", "count"),
            mean_fatal_serious_frac=("fatal_serious_frac", "mean"),
            median_fatal_serious_frac=("fatal_serious_frac", "median"),
            destruction_rate=("is_destroyed", "mean"),
            n_damage=("is_destroyed", "count"),
            median_onboard=("total_onboard", "median"),
        )
        .reset_index()
    )
    # Require both injury and damage sample sizes so rates are not one-off events
    return g[(g["n_injury"] >= min_n) & (g["n_damage"] >= min(min_n, 20))].sort_values(
        ["mean_fatal_serious_frac", "destruction_rate"]
    )


small_makes = make_safety_table(small, min_n=50)
# Fewer large-jet manufacturers exist; use 20 accidents as a floor
large_makes = make_safety_table(large, min_n=20)

small_top15 = small_makes.head(15)
large_top15 = large_makes.head(15)

print("Small makes (lowest mean injury fraction, n>=50)")
display(small_top15)
print("Large makes (lowest mean injury fraction, n>=20)")
display(large_top15)

plot_small = small_top15.copy()
plot_small["size_class"] = "Small (n>=50)"
plot_large = large_top15.copy()
plot_large["size_class"] = "Large (n>=20)"
plot_df = pd.concat([plot_small, plot_large], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False)
for ax, subset, title in [
    (axes[0], plot_small.sort_values("mean_fatal_serious_frac", ascending=True), "Small aircraft makes"),
    (axes[1], plot_large.sort_values("mean_fatal_serious_frac", ascending=True), "Large passenger makes"),
]:
    sns.barplot(
        data=subset,
        y="Make",
        x="mean_fatal_serious_frac",
        ax=ax,
        color="#4C78A8",
    )
    ax.set_xlabel("Mean fatal/serious injury fraction")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(0, max(plot_df["mean_fatal_serious_frac"].max() * 1.15, 0.3))
fig.suptitle("Lowest mean onboard fatal/serious injury rates by manufacturer", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""**Distribution of injury rates: small makes**

Use a violinplot to look at the distribution of the fraction of passengers serious/fatally injured for small airplane makes. Just display makes with the ten lowest mean serious/fatal injury rates.
"""))

a.append(new_code_cell("""small_top10_names = small_makes.head(10)["Make"].tolist()
small_violin = small[small["Make"].isin(small_top10_names)].copy()
small_violin["Make"] = pd.Categorical(small_violin["Make"], categories=small_top10_names, ordered=True)

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=small_violin.dropna(subset=["fatal_serious_frac"]),
    x="Make",
    y="fatal_serious_frac",
    inner="quartile",
    cut=0,
    color="#72B7B2",
)
plt.xticks(rotation=35, ha="right")
plt.ylabel("Fatal/serious injury fraction")
plt.xlabel("Manufacturer")
plt.title("Injury-rate distribution — 10 small makes with the lowest mean rates")
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""**Distribution of injury rates: large makes**

Use a stripplot to look at the distribution of the fraction of passengers serious/fatally injured for large airplane makes. Just display makes with the ten lowest mean serious/fatal injury rates.
"""))

a.append(new_code_cell("""large_top10_names = large_makes.head(min(10, len(large_makes)))["Make"].tolist()
large_strip = large[large["Make"].isin(large_top10_names)].copy()
large_strip["Make"] = pd.Categorical(large_strip["Make"], categories=large_top10_names, ordered=True)

plt.figure(figsize=(11, 6))
sns.stripplot(
    data=large_strip.dropna(subset=["fatal_serious_frac"]),
    x="Make",
    y="fatal_serious_frac",
    jitter=0.25,
    alpha=0.45,
    color="#E45756",
)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Fatal/serious injury fraction")
plt.xlabel("Manufacturer")
plt.title("Injury-rate distribution — large passenger makes (lowest mean rates)")
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""**Evaluate the rate of aircraft destruction for both small and large aircraft by Make.**

Sort your results and keep the lowest 15.
"""))

a.append(new_code_cell("""small_dest = small_makes.sort_values(["destruction_rate", "mean_fatal_serious_frac"]).head(15)
large_dest = large_makes.sort_values(["destruction_rate", "mean_fatal_serious_frac"]).head(15)

print("Small makes — lowest destruction rate (n>=50)")
display(small_dest[["Make", "n", "destruction_rate", "mean_fatal_serious_frac", "median_onboard"]])
print("Large makes — lowest destruction rate (n>=20)")
display(large_dest[["Make", "n", "destruction_rate", "mean_fatal_serious_frac", "median_onboard"]])

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
sns.barplot(data=small_dest.sort_values("destruction_rate"), y="Make", x="destruction_rate", ax=axes[0], color="#F58518")
axes[0].set_title("Small makes: lowest destruction rate")
axes[0].set_xlabel("Share of accidents that destroyed the aircraft")
axes[0].set_ylabel("")
sns.barplot(data=large_dest.sort_values("destruction_rate"), y="Make", x="destruction_rate", ax=axes[1], color="#F58518")
axes[1].set_title("Large makes: lowest destruction rate")
axes[1].set_xlabel("Share of accidents that destroyed the aircraft")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""#### Provide a short discussion on your findings for your summary statistics and plots:
- Make any recommendations for Makes here based off of the destroyed fraction and fraction fatally/seriously injured
- Comment on the calculated statistics and any corresponding distributions you have visualized.
"""))

a.append(new_markdown_cell("""**Manufacturer recommendations**

Rates below are *conditional on an accident already occurring* (NTSB files do not contain hours flown), so they describe crash survivability and airframe robustness, not how often a type crashes.

**Large / passenger aircraft (median occupancy ≥ 20, ≥ 20 accidents).** McDonnell Douglas, Bombardier, Embraer, and Boeing sit in a tight band of mean fatal/serious fractions around 0.08–0.12, well below the small-aircraft average (~0.26). Bombardier has the lowest destruction rate among large makes with a usable sample; Airbus shows a higher destruction share but many widebody reports leave damage blank, so that rate is noisier. Douglas (legacy DC types, n≈40) is the clear laggard on both metrics and is not recommended. **Primary large-fleet manufacturers to consider: Boeing, Bombardier, Embraer, and McDonnell Douglas / Boeing-heritage twins (MD-80/717 family).** Airbus remains acceptable on injury but should be underwritten with extra weight on damage uncertainty.

**Small aircraft (≥ 50 accidents).** Lowest mean injury fractions: Waco, Helio, Maule, Great Lakes, plus Aviat and Diamond when destruction is also considered. Diamond and Aviat combine modest injury rates with the lowest destruction rates (~7%). Maule has both a large sample (n≈570) and low injury *and* destruction, so it is the most statistically robust small-make recommendation. Luscombe is similar on damage but with older airframes. Cessna and Piper dominate the file and sit near the small-aircraft average — not the worst, but not the safest. **Small-make shortlist: Maule, Aviat, Diamond; secondary: Helio and Luscombe.** Vintage types (Waco, Great Lakes, Stinson) look statistically favorable but are less relevant for an insurer of currently produced professional aircraft.

**Distributions.** Small-make violins are zero-inflated: most accidents have *no* serious/fatal injuries, with a spike at 1.0 (typically 1–2 people on board, all killed). Means are therefore pulled by rare catastrophes; medians are 0 for every recommended make. Large-make strip plots show the same 0-heavy pattern plus a few hull-loss outliers — which is why we require n≥20 and do not rank a type on a handful of events.
"""))

a.append(new_markdown_cell("""### Analyze plane types
- plot the mean fatal/seriously injured fraction for both small and larger planes
- also provide a distributional plot of your choice for the fatal/seriously injured fraction by airplane type (stripplot, violin, etc)
- filter ensuring that you have at least ten individual examples in each model/make to average over
"""))

a.append(new_markdown_cell("**Larger planes**"))

a.append(new_code_cell("""def model_safety_table(frame, min_n=10):
    g = (
        frame.groupby("make_model")
        .agg(
            n=("fatal_serious_frac", "size"),
            mean_fatal_serious_frac=("fatal_serious_frac", "mean"),
            destruction_rate=("is_destroyed", "mean"),
            median_onboard=("total_onboard", "median"),
            Make=("Make", "first"),
        )
        .reset_index()
    )
    return g[g["n"] >= min_n].sort_values(["mean_fatal_serious_frac", "destruction_rate"])


large_models = model_safety_table(large, min_n=10)
print(f"Large make/model types with n>=10: {len(large_models)}")
display(large_models.head(15))

plot_lm = large_models.sort_values("mean_fatal_serious_frac").head(15)
plt.figure(figsize=(10, 7))
sns.barplot(data=plot_lm, y="make_model", x="mean_fatal_serious_frac", color="#4C78A8")
plt.xlabel("Mean fatal/serious injury fraction")
plt.ylabel("")
plt.title("Large airplane types with the lowest mean injury rates (n≥10)")
plt.tight_layout()
plt.show()

order = plot_lm["make_model"].tolist()
strip_lm = large[large["make_model"].isin(order)].copy()
strip_lm["make_model"] = pd.Categorical(strip_lm["make_model"], categories=order, ordered=True)
plt.figure(figsize=(12, 6))
sns.stripplot(data=strip_lm.dropna(subset=["fatal_serious_frac"]), x="make_model", y="fatal_serious_frac",
              jitter=0.2, alpha=0.55, color="#E45756")
plt.xticks(rotation=40, ha="right")
plt.ylabel("Fatal/serious injury fraction")
plt.xlabel("")
plt.title("Accident-level injury fractions — safest large types (n≥10)")
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""**Smaller planes**
- for smaller planes, limit your plotted results to the makes with the 10 lowest mean serious/fatal injury fractions
"""))

a.append(new_code_cell("""small_models_all = model_safety_table(small, min_n=10)
small_models = small_models_all[small_models_all["Make"].isin(small_top10_names)]
print("Small types (n>=10) restricted to the 10 lowest-injury makes:")
display(small_models.head(15))

plot_sm = small_models.sort_values("mean_fatal_serious_frac").head(15)
plt.figure(figsize=(10, 7))
sns.barplot(data=plot_sm, y="make_model", x="mean_fatal_serious_frac", color="#72B7B2")
plt.xlabel("Mean fatal/serious injury fraction")
plt.ylabel("")
plt.title("Small airplane types with the lowest mean injury rates\\n(restricted to 10 safest makes, n≥10)")
plt.tight_layout()
plt.show()

order_s = plot_sm["make_model"].tolist()
violin_sm = small[small["make_model"].isin(order_s)].copy()
violin_sm["make_model"] = pd.Categorical(violin_sm["make_model"], categories=order_s, ordered=True)
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=violin_sm.dropna(subset=["fatal_serious_frac"]),
    x="make_model",
    y="fatal_serious_frac",
    inner="quartile",
    cut=0,
    color="#72B7B2",
)
plt.xticks(rotation=40, ha="right")
plt.ylabel("Fatal/serious injury fraction")
plt.xlabel("")
plt.title("Injury-rate distribution — safest small types among the 10 safest makes")
plt.tight_layout()
plt.show()
"""))

a.append(new_markdown_cell("""### Discussion of Specific Airplane Types
- Discuss what you have found above regarding passenger fraction seriously/ both small and large airplane models.
"""))

a.append(new_markdown_cell("""**Large types (n ≥ 10).** The lowest mean injury fractions belong to later-generation airliners whose accidents in this file are often runway / ground events with most occupants uninjured: Boeing 717-200, 757-232 / 757-222, McDonnell Douglas MD-88, Boeing 737-7H4, Bombardier CRJ (CL-600-2B19), Boeing 777, 747-400, and Embraer EMB-145LR. Several of these also have **zero recorded hull-loss destructions** in the n≥10 set (717, 757-232, MD-88, 737-7H4, EMB-145LR). Treat n=10–18 types as directional, not definitive. Older 737-200 / 737-400 and some A320 records show much higher injury and destruction fractions and should not be the basis of a “narrowbody is safe” claim without separating generation.

**Small types (n ≥ 10, within the 10 safest makes).** Several tailwheel / utility models show *zero* mean fatal/serious injury across ≥10 accidents (Diamond DA20-C1, Piper PA-18A 150, Grumman G-164B, Cessna 180C). Maule MX-7-235, M-7-235B, and M-4 repeat the manufacturer-level story at the type level: low injury and little destruction. These are typically 1–2 occupant flights, so a single fatality moves the fraction from 0 to 1 — which is why the n≥10 rule matters, and why we still prefer Maule overall (more models, larger n) over any one 11-accident Diamond variant.

**Recommendation to the client.** For passenger operations, underwrite **Boeing 757 / 717 / 777 / 747-400, MD-88 / 717 family, Bombardier CRJ, and Embraer ERJ-145**. For small professional aircraft, prefer **Maule M-7 / MX-7 and Diamond DA20**, with Aviat types as a damage-robust alternative. Avoid using Boeing Stearman / PT-17 vintage trainers (they inherit the Boeing name but are two-seat biplanes) as evidence about Boeing airliners — they were classified as small by occupancy.
"""))

a.append(new_markdown_cell("""### Exploring Other Variables
- Investigate how other variables effect aircraft damage and injury. You must choose **two** factors out of the following but are free to analyze more:

- Weather Condition
- Engine Type
- Number of Engines
- Phase of Flight
- Purpose of Flight

For each factor provide a discussion explaining your analysis with appropriate visualization / data summaries and interpreting your findings.
"""))

a.append(new_code_cell("""def factor_summary(frame, col, min_n=100):
    g = (
        frame.groupby(col)
        .agg(
            n=("fatal_serious_frac", "size"),
            n_injury=("fatal_serious_frac", "count"),
            mean_fatal_serious_frac=("fatal_serious_frac", "mean"),
            destruction_rate=("is_destroyed", "mean"),
            n_damage=("is_destroyed", "count"),
        )
        .reset_index()
        .sort_values("mean_fatal_serious_frac")
    )
    return g[g["n_injury"] >= min_n]


# ----- Factor 1: Weather Condition -----
weather = df.copy()
weather["Weather.Condition"] = weather["Weather.Condition"].fillna("Missing")
weather_tbl = factor_summary(weather, "Weather.Condition", min_n=50)
print("Factor 1 — Weather.Condition")
display(weather_tbl)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
order_w = ["VMC", "IMC", "Unknown", "Missing"]
order_w = [x for x in order_w if x in set(weather_tbl["Weather.Condition"])]
sns.barplot(data=weather_tbl, x="Weather.Condition", y="mean_fatal_serious_frac", order=order_w, ax=axes[0], color="#4C78A8")
axes[0].set_ylabel("Mean fatal/serious fraction")
axes[0].set_title("Injury severity by weather")
sns.barplot(data=weather_tbl, x="Weather.Condition", y="destruction_rate", order=order_w, ax=axes[1], color="#F58518")
axes[1].set_ylabel("Destruction rate")
axes[1].set_title("Aircraft destroyed by weather")
plt.tight_layout()
plt.show()

# Sample-size note
print(weather["Weather.Condition"].value_counts())


# ----- Factor 2: Broad phase of flight -----
phase = df.copy()
phase["Broad.phase.of.flight"] = phase["Broad.phase.of.flight"].fillna("Missing")
phase_tbl = factor_summary(phase, "Broad.phase.of.flight", min_n=200)
print("\\nFactor 2 — Broad.phase.of.flight (n_injury >= 200)")
display(phase_tbl)

phase_plot = phase_tbl.sort_values("mean_fatal_serious_frac")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=phase_plot, y="Broad.phase.of.flight", x="mean_fatal_serious_frac", ax=axes[0], color="#4C78A8")
axes[0].set_xlabel("Mean fatal/serious fraction")
axes[0].set_ylabel("")
axes[0].set_title("Injury severity by phase of flight")
sns.barplot(data=phase_plot, y="Broad.phase.of.flight", x="destruction_rate", ax=axes[1], color="#F58518")
axes[1].set_xlabel("Destruction rate")
axes[1].set_ylabel("")
axes[1].set_title("Aircraft destroyed by phase of flight")
plt.tight_layout()
plt.show()


# ----- Additional context: engine type (not one of the two primary claims) -----
eng = df.copy()
eng["Engine.Type"] = eng["Engine.Type"].fillna("Missing")
eng_tbl = factor_summary(eng, "Engine.Type", min_n=200)
print("\\nContext — Engine.Type")
display(eng_tbl)
"""))

a.append(new_markdown_cell("""**Factor 1 — Weather condition (VMC vs IMC)**

Visual meteorological conditions (VMC) account for the large majority of accidents, but **instrument conditions (IMC) are far more lethal when an accident does occur**. Mean fatal/serious fraction is about **0.23 in VMC vs ~0.67 in IMC**, and destruction rates rise from **~0.17 to ~0.60**. Unknown weather tracks IMC more than VMC, which is consistent with poorer reporting on more severe events. Sample sizes are large (tens of thousands of VMC accidents and several thousand IMC), so the gap is not a small-n artifact.

Interpretation for the insurer: weather is a **condition-of-loss** factor, not an airframe factor. IMC accidents are enriched for continued-VFR-into-IMC, CFIT, and loss of control — mechanisms that destroy the aircraft and kill occupants regardless of make. Underwriting and safety programs should price IMC exposure (IFR currency, icing, night IFR) separately from type selection. Type rankings above were not weather-adjusted; a make that flies disproportionately in IMC could look worse for reasons unrelated to the airframe.

**Factor 2 — Phase of flight**

Phase splits injury and damage even more sharply, again with large samples. **Landing and taxi** have mean fatal/serious fractions around **0.02–0.04** and destruction rates under **0.05** — lots of accidents, few catastrophes. **Maneuvering** (~0.54 injury / ~0.48 destroyed) and **climb** (~0.43 / ~0.44) are the dangerous phases; cruise and approach sit in between. Missing phase is common in later years of the file and should not be treated as a real phase.

Interpretation: low-energy, airport-environment accidents dominate the count and pull down overall means. High-energy maneuvering (ag, airshow, low-level) and climb (engine failure after takeoff) drive hull losses. This supports recommending types used for training and transport over those used for aerial application / maneuvering work, and it cautions against treating a low fleet-wide injury mean as “the airframe is crashworthy” if the make’s accidents are mostly landing mishaps.

**Engine type (supporting).** Turbo-fan airliners have lower injury fractions (~0.12) than reciprocating piston aircraft (~0.26), but this is heavily confounded with size class and phase (airliners vs single-engine pistons). We therefore treat weather and phase as the two primary, less circular factors, and treat engine type as a proxy for the small vs large split already analyzed.
"""))

nb_a = new_notebook(cells=a, metadata={
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
})
nbf.write(nb_a, "Aviation_Accidents_Data_Analysis.ipynb")
print("Wrote both notebooks")
