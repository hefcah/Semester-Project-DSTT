"""Streamlit interface for electricity-bill prediction and waste checks."""

from __future__ import annotations

import calendar
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"
CITIES = ["Islamabad", "Karachi", "Lahore", "Multan", "Peshawar", "Skardu"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def bill(
    units: float,
    protected: bool,
    sanctioned_kw: float,
    tariff: Mapping[str, Any],
) -> tuple[float, float, float, str, str]:
    """Return energy charge, fixed charge, total, reached slab, and status."""
    units = max(0.0, float(units))
    sanctioned_kw = max(0.0, float(sanctioned_kw))
    protected_applies = bool(protected and units <= 200)

    if protected_applies:
        slabs = tariff["protected"]
        first, second = slabs[0], slabs[1]
        energy = min(units, float(first["up_to"])) * float(first["rate"])
        energy += max(units - float(first["up_to"]), 0.0) * float(second["rate"])
        reached = first if units <= float(first["up_to"]) else second
        status = "Protected"
    else:
        slabs = tariff["unprotected"]
        reached = next(
            (slab for slab in slabs if units <= float(slab["up_to"])),
            slabs[-1],
        )
        energy = units * float(reached["rate"])
        status = "Unprotected"

    fixed = float(reached["fixed_per_kw"]) * sanctioned_kw
    label = f"Up to {int(reached['up_to'])} kWh"
    return energy, fixed, energy + fixed, label, status


def units_before_next_slab(
    units: float,
    protected: bool,
    tariff: Mapping[str, Any],
) -> float:
    """Return remaining units to the next active slab boundary."""
    units = max(0.0, float(units))
    if protected and units <= 200:
        if units >= 200:
            return 0.0
        upper = next(
            (float(slab["up_to"]) for slab in tariff["protected"] if units < float(slab["up_to"])),
            200.0,
        )
    else:
        upper = next(
            (float(slab["up_to"]) for slab in tariff["unprotected"] if units < float(slab["up_to"])),
            float(tariff["unprotected"][-1]["up_to"]),
        )
    return max(0.0, upper - units)


def _feature_value(
    feature: str,
    values: Mapping[str, float],
    city: str,
) -> float:
    """Resolve a saved model feature or fail with its exact missing name."""
    if feature.startswith("city_"):
        feature_city = feature.removeprefix("city_")
        if feature_city not in CITIES:
            raise ValueError(f"Cannot build required feature '{feature}': unknown city dummy")
        return float(city == feature_city)
    if feature not in values:
        raise ValueError(f"Cannot build required feature '{feature}'")
    value = values[feature]
    if not np.isfinite(float(value)):
        raise ValueError(f"Cannot build required feature '{feature}': value is not finite")
    return float(value)


def build_monthly_feature_row(
    features: Sequence[str],
    city: str,
    month: int,
    weather: Mapping[str, float],
    lag1_kwh: float | None = None,
    temperature_override: float | None = None,
) -> pd.DataFrame:
    """Build the exact monthly feature columns listed by a saved model."""
    if city not in CITIES:
        raise ValueError(f"Cannot build city features for unknown city '{city}'")
    if not 1 <= int(month) <= 12:
        raise ValueError(f"Cannot build calendar features for month '{month}'")
    values = {key: float(value) for key, value in weather.items()}
    if temperature_override is not None:
        temperature = float(temperature_override)
        days = calendar.monthrange(2026, int(month))[1]
        values["temp_mean"] = temperature
        values["cdd_sum"] = days * max(temperature - 24.0, 0.0)
        values["hdd_sum"] = days * max(18.0 - temperature, 0.0)
    values["month_sin"] = math.sin(2 * math.pi * int(month) / 12)
    values["month_cos"] = math.cos(2 * math.pi * int(month) / 12)
    if lag1_kwh is not None:
        values["lag1_kwh"] = float(lag1_kwh)
    row = {
        feature: _feature_value(feature, values, city)
        for feature in features
    }
    return pd.DataFrame([row], columns=list(features), dtype=float)


def build_daily_expected_feature_row(
    features: Sequence[str],
    city: str,
    day: date,
    previous_kwh: Sequence[float],
    weather: Mapping[str, float],
    temperature_override: float | None = None,
) -> pd.DataFrame:
    """Build saved expected-use features from weather and preceding daily use."""
    valid_previous = [float(value) for value in previous_kwh if np.isfinite(float(value))]
    if len(valid_previous) < 4:
        raise ValueError("At least four valid previous daily kWh values are required")
    valid_previous = valid_previous[-7:]
    if any(value < 0 for value in valid_previous):
        raise ValueError("Previous daily kWh values must be non-negative")
    values = {key: float(value) for key, value in weather.items()}
    temperature = (
        float(temperature_override)
        if temperature_override is not None
        else values.get("temp_mean")
    )
    if temperature is None or not np.isfinite(temperature):
        raise ValueError("Cannot build required feature 'temp_mean'")
    values["temp_mean"] = temperature
    values["cdd"] = max(temperature - 24.0, 0.0)
    values["hdd"] = max(18.0 - temperature, 0.0)
    values["prev7_mean"] = float(np.mean(valid_previous))
    values["prev_day_kwh"] = valid_previous[-1]
    values["dow"] = float(day.weekday())
    values["month_sin"] = math.sin(2 * math.pi * day.month / 12)
    values["month_cos"] = math.cos(2 * math.pi * day.month / 12)
    row = {
        feature: _feature_value(feature, values, city)
        for feature in features
    }
    return pd.DataFrame([row], columns=list(features), dtype=float)


def predict_kwh_from_log_target(model: Any, row: pd.DataFrame) -> float:
    """Predict kWh from an estimator trained on log1p(kWh)."""
    fitted_regressor = getattr(model, "regressor_", model)
    predicted_log = float(np.asarray(fitted_regressor.predict(row)).reshape(-1)[0])
    return max(0.0, float(np.expm1(predicted_log)))


def load_model_bundles(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Load and validate all app model bundles."""
    import joblib

    filenames = {
        "coldstart": "best_coldstart.joblib",
        "knownhouse": "best_knownhouse.joblib",
        "waste": "waste_detector.joblib",
    }
    bundles = {
        name: joblib.load(models_dir / filename)
        for name, filename in filenames.items()
    }
    required = {
        "coldstart": {"model", "features", "name", "setting"},
        "knownhouse": {"model", "features", "name", "setting"},
        "waste": {
            "expected_model", "expected_features", "iforest", "if_features",
            "city_mad", "z_threshold", "contamination", "cities",
        },
    }
    for name, keys in required.items():
        missing = keys - set(bundles[name])
        if missing:
            raise ValueError(f"{filenames[name]} is missing bundle keys: {sorted(missing)}")
    return bundles


def _load_climatology(path: Path, city: str, month: int) -> dict[str, float]:
    table = pd.read_csv(path)
    matches = table.loc[
        table["city"].eq(city) & table["calendar_month"].eq(int(month))
    ]
    if matches.empty:
        raise ValueError(f"No climatology found for city '{city}', month {month}")
    return {
        key: float(value)
        for key, value in matches.iloc[0].items()
        if key not in {"city", "calendar_month"} and pd.notna(value)
    }


def _format_money(value: float) -> str:
    return f"Rs {value:,.2f}"


def main() -> None:
    """Render the Streamlit interface; model loading errors are shown verbatim."""
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Electricity Bill Predictor", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #FFFFFF; color: #0F172A; }
        .hero-banner {
            background: linear-gradient(115deg, #0F766E 0%, #2563EB 100%);
            color: #FFFFFF; padding: 1.6rem 2rem; border-radius: 18px;
            margin: 0.25rem 0 1.5rem 0; box-shadow: 0 8px 24px #0f172a18;
        }
        .hero-banner h1 { color: #FFFFFF; margin: 0; font-size: 2rem; }
        .hero-banner p { color: #E0F2FE; margin: 0.45rem 0 0 0; font-size: 1.05rem; }
        .metric-card {
            background: #FFFFFF; border: 1px solid #DDE7E5;
            border-left: 6px solid #0F766E; border-radius: 14px;
            padding: 1rem 1.1rem; min-height: 145px;
            box-shadow: 0 3px 12px #0f172a0b; margin-bottom: 0.6rem;
        }
        .metric-card.low { border-left-color: #16A34A; }
        .metric-card.high { border-left-color: #64748B; }
        .metric-card.green { border-left-color: #16A34A; }
        .metric-card.amber { border-left-color: #D97706; }
        .metric-card.orange { border-left-color: #EA580C; }
        .metric-card.red { border-left-color: #DC2626; }
        .metric-label { color: #475569; font-size: 0.9rem; font-weight: 650; }
        .metric-value { color: #0F172A; font-size: 1.55rem; font-weight: 750; margin: 0.25rem 0; }
        .metric-detail { color: #334155; font-size: 0.88rem; line-height: 1.5; }
        .status-banner {
            border-radius: 14px; padding: 1rem 1.25rem; margin: 0.8rem 0 1rem 0;
            font-size: 1.2rem; font-weight: 750;
        }
        .status-banner.normal { color: #166534; background: #DCFCE7; border: 1px solid #86EFAC; }
        .status-banner.waste { color: #991B1B; background: #FEE2E2; border: 1px solid #FCA5A5; }
        div[data-testid="stTabs"] button {
            border-radius: 10px 10px 0 0; font-weight: 650;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0F766E; border-bottom: 3px solid #0F766E;
        }
        </style>
        <div class="hero-banner">
          <h1>Electricity Bill Predictor &amp; Waste Check</h1>
          <p>Estimate next month's bill, see which slab you land in, and check whether today's use looks wasteful</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("How to use")
        st.markdown(
            "1. Choose your city and month, then add last month's units if you know them.\n"
            "2. Check the estimated units, bill range, and tariff slab.\n"
            "3. For a waste check, enter the previous days and today's use."
        )
        with st.expander("What do these words mean?"):
            st.markdown(
                "- **Sanctioned load:** the maximum connection load approved by your utility, shown on your bill.\n"
                "- **Protected consumer:** a household meeting the tariff's protected-use eligibility rules.\n"
                "- **Slab:** all units in the reached slab are billed at that slab's rate.\n"
                "- **kWh / unit:** one kilowatt-hour, the electricity unit printed on your bill."
            )

    @st.cache_resource(show_spinner="Loading saved models...")
    def cached_load_models(models_dir: str) -> dict[str, dict[str, Any]]:
        return load_model_bundles(Path(models_dir))

    try:
        models = cached_load_models(str(MODELS_DIR))
    except Exception as exc:
        st.error(f"Could not load model files: {exc}")
        st.stop()
        return

    with (ASSETS_DIR / "tariff.json").open(encoding="utf-8") as file:
        tariff = json.load(file)
    monthly_climatology_path = ASSETS_DIR / "climatology_monthly.csv"
    daily_climatology_path = ASSETS_DIR / "climatology_daily.csv"
    bill_tab, waste_tab, about_tab = st.tabs(
        ["Bill prediction", "Waste check", "About"]
    )

    def metric_card(
        title: str,
        value: str,
        detail: str,
        variant: str = "",
    ) -> None:
        st.markdown(
            f"""
            <div class="metric-card {variant}">
              <div class="metric-label">{title}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bill_tab:
        st.subheader("Estimate next month's bill")

        def set_bill_example(example: str) -> None:
            examples = {
                "small_flat": {
                    "bill_city": "Islamabad", "bill_month": "January",
                    "known_lag": True, "lag_units": 150.0,
                    "sanctioned_kw": 2.0, "protected": True,
                    "use_temp_override": False, "temperature": 18.0,
                },
                "family_ac": {
                    "bill_city": "Lahore", "bill_month": "July",
                    "known_lag": True, "lag_units": 450.0,
                    "sanctioned_kw": 3.0, "protected": False,
                    "use_temp_override": False, "temperature": 30.0,
                },
                "protected_limit": {
                    "bill_city": "Islamabad", "bill_month": "January",
                    "known_lag": True, "lag_units": 195.0,
                    "sanctioned_kw": 2.0, "protected": True,
                    "use_temp_override": False, "temperature": 18.0,
                },
            }
            st.session_state.update(examples[example])

        example_columns = st.columns(3)
        examples = (
            ("Example: small flat", "small_flat", "example_small_flat"),
            ("Example: family home with AC", "family_ac", "example_family_ac"),
            ("Example: protected household near the 200-unit limit",
             "protected_limit", "example_protected_limit"),
        )
        for column, (label, example, key) in zip(example_columns, examples):
            column.button(
                label,
                key=key,
                help="Fill the bill inputs with an example household.",
                on_click=set_bill_example,
                args=(example,),
                width="stretch",
            )

        left, right = st.columns(2)
        with left:
            bill_city = st.selectbox(
                "City", CITIES, key="bill_city",
                help="Choose the city matching the household's electricity meter.",
            )
            month_name = st.selectbox(
                "Billing month", MONTH_NAMES, key="bill_month",
                help="Choose the month you want to estimate.",
            )
            bill_month = MONTH_NAMES.index(month_name) + 1
            known_lag = st.checkbox(
                "I know last month's units", value=False, key="known_lag",
                help="Turn this on if last month's bill is available.",
            )
            lag_units = st.number_input(
                "Last month's units (kWh)", min_value=0.0, max_value=3000.0,
                value=0.0, step=1.0, key="lag_units", disabled=not known_lag,
                help="Last month's units: the units consumed line on last month's bill.",
            )
        with right:
            sanctioned_kw = st.number_input(
                "Sanctioned load (kW)", min_value=0.1, max_value=100.0,
                value=3.0, step=0.1, key="sanctioned_kw",
                help="Sanctioned load is printed on your electricity bill, usually 1 to 5 kW.",
            )
            protected = st.checkbox(
                "Protected consumer (200 units or less in each of the last 6 months)",
                value=False, key="protected",
                help="Use the protected option only if your household meets this tariff condition.",
            )
            bill_weather = _load_climatology(
                monthly_climatology_path, bill_city, bill_month
            )
            use_temp_override = st.checkbox(
                "Override climatology temperature", value=False,
                key="use_temp_override",
                help="Turn this on only if you want to replace the city's monthly weather average.",
            )
            temperature = st.number_input(
                f"Mean temperature (°C; climatology {bill_weather['temp_mean']:.1f})",
                value=float(bill_weather["temp_mean"]), step=0.5, key="temperature",
                disabled=not use_temp_override,
                help="Use a representative monthly temperature from a local weather forecast.",
            )

        model_key = "knownhouse" if known_lag else "coldstart"
        bundle = models[model_key]
        monthly_row = build_monthly_feature_row(
            bundle["features"], bill_city, bill_month, bill_weather,
            lag1_kwh=float(lag_units) if known_lag else None,
            temperature_override=float(temperature) if use_temp_override else None,
        )
        predicted_units = predict_kwh_from_log_target(bundle["model"], monthly_row)
        typical_error = 121.0 if known_lag else 171.0
        low_units = max(0.0, predicted_units - typical_error)
        high_units = predicted_units + typical_error
        energy, fixed, predicted_total, predicted_slab, predicted_status = bill(
            predicted_units, protected, sanctioned_kw, tariff
        )
        st.markdown(
            f"In **{month_name}** a household in **{bill_city}** will probably use "
            f"about **{predicted_units:,.0f} units**, about **{_format_money(predicted_total)}**."
        )
        st.caption(
            f"Model used: {bundle['name']} ({bundle['setting']}). Typical absolute error "
            f"on unseen houses, 6 holdout houses: ±{typical_error:.0f} kWh."
        )

        estimates = (
            ("Low estimate", low_units, "low"),
            ("Predicted", predicted_units, ""),
            ("High estimate", high_units, "high"),
        )
        cards = st.columns(3)
        predicted_band = (
            "green" if predicted_units <= 200 else
            "amber" if predicted_units <= 400 else
            "orange" if predicted_units <= 600 else "red"
        )
        for column, (label, units, variant) in zip(cards, estimates):
            energy_part, fixed_part, total, slab, status = bill(
                units, protected, sanctioned_kw, tariff
            )
            if label == "Predicted":
                variant = predicted_band
            with column:
                metric_card(
                    label, f"{units:,.0f} kWh",
                    f"Bill: {_format_money(total)}<br>"
                    f"Energy: {_format_money(energy_part)} · Fixed: {_format_money(fixed_part)}<br>"
                    f"{slab} · {status}",
                    variant,
                )

        active_slabs = tariff["protected"] if protected and predicted_units <= 200 else tariff["unprotected"]
        lower_bound = 0.0
        upper_bound = float(active_slabs[-1]["up_to"])
        for slab_row in active_slabs:
            upper_bound = float(slab_row["up_to"])
            if predicted_units <= upper_bound:
                break
            lower_bound = upper_bound
        progress = min(1.0, max(0.0, (predicted_units - lower_bound) / max(upper_bound - lower_bound, 1.0)))
        st.progress(progress, text=f"{progress:.0%} through the current slab ({lower_bound:,.0f}–{upper_bound:,.0f} kWh)")
        st.write(
            f"Units left before the next slab: "
            f"{units_before_next_slab(predicted_units, protected, tariff):,.1f} kWh."
        )

        if protected and (predicted_units > 200 or high_units > 200):
            st.warning(
                "The predicted use or the high end of its error band exceeds 200 kWh. "
                "The protected rate does not apply above 200 units."
            )
            with_protection = bill(predicted_units, True, sanctioned_kw, tariff)
            without_protection = bill(predicted_units, False, sanctioned_kw, tariff)
            st.write(
                f"At predicted use: with protected selection {_format_money(with_protection[2])} "
                f"({with_protection[4]}, {with_protection[3]}); without protection "
                f"{_format_money(without_protection[2])} "
                f"({without_protection[4]}, {without_protection[3]})."
            )

        chart_rows = []
        chart_units = np.arange(0.0, 801.0, 10.0)
        for series, chart_protected in (
            ("Unprotected", False),
            ("Protected household (only up to 200 units)", True),
        ):
            for units in chart_units:
                chart_rows.append({
                    "Units (kWh)": units,
                    "Bill (Rs)": bill(units, chart_protected, sanctioned_kw, tariff)[2],
                    "Series": series,
                })
        bill_chart_data = pd.DataFrame(chart_rows)
        prediction_point = pd.DataFrame([{
            "Units (kWh)": predicted_units,
            "Bill (Rs)": predicted_total,
            "Series": "Predicted use",
        }])
        limit_data = pd.DataFrame([{"Units (kWh)": 200.0, "label": "protected limit"}])
        lines = alt.Chart(bill_chart_data).mark_line().encode(
            x=alt.X("Units (kWh):Q", title="Units (kWh)", scale=alt.Scale(domain=[0, 800])),
            y=alt.Y("Bill (Rs):Q", title="Bill (Rs)", scale=alt.Scale(domainMin=0)),
            color=alt.Color("Series:N", title=None),
            tooltip=["Series:N", "Units (kWh):Q", alt.Tooltip("Bill (Rs):Q", format=",.2f")],
        )
        limit_rule = alt.Chart(limit_data).mark_rule(
            color="#475569", strokeDash=[7, 4], strokeWidth=2,
        ).encode(x="Units (kWh):Q")
        limit_label = alt.Chart(limit_data).mark_text(
            angle=270, align="left", dx=8, color="#475569",
        ).encode(
            x="Units (kWh):Q", y=alt.value(35), text="label:N",
        )
        point = alt.Chart(prediction_point).mark_point(
            filled=True, size=150, color="#7C3AED",
        ).encode(
            x="Units (kWh):Q", y="Bill (Rs):Q",
            tooltip=["Series:N", "Units (kWh):Q", alt.Tooltip("Bill (Rs):Q", format=",.2f")],
        )
        st.subheader("Bill by monthly usage")
        st.altair_chart(
            (lines + limit_rule + limit_label + point).properties(height=360),
            width="stretch",
        )

        savings_rows = []
        for cut in (0.05, 0.10, 0.20):
            new_units = predicted_units * (1.0 - cut)
            new_bill = bill(new_units, protected, sanctioned_kw, tariff)[2]
            savings_rows.append({
                "Usage reduction": f"{int(cut * 100)}%",
                "Units": new_units,
                "Bill": new_bill,
                "Saving": predicted_total - new_bill,
            })
        st.caption("If you cut usage")
        savings_table = pd.DataFrame(savings_rows)
        st.dataframe(
            savings_table.style.format({
                "Units": "{:,.1f}", "Bill": "Rs {:,.2f}", "Saving": "Rs {:,.2f}",
            }).map(lambda _: "color: #15803D; font-weight: 700", subset=["Saving"]),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Estimate excludes GST, fuel price adjustment and other surcharges; "
            "slabs are read from tariff.json (editable). The model was trained on "
            "53 REWDP houses from 6 cities and is an estimate, not an official bill."
        )

    with waste_tab:
        st.subheader("Check whether today's use is unusually high")
        st.info(
            "Enter the previous 7 days' units, then today's units; the app compares "
            "today with what is expected for the weather."
        )

        def set_waste_example(example: str) -> None:
            previous_days = "12.4, 10.8, 11.6, 13.1, 9.7, 12.0, 11.2"
            st.session_state.update({
                "waste_city": "Lahore",
                "waste_date": date.today(),
                "today_kwh": 11.8 if example == "normal" else 24.0,
                "previous_text": previous_days,
                "outage_minutes": 0,
                "use_waste_temp_override": False,
                "waste_temperature": 30.0,
            })

        normal_column, high_column = st.columns(2)
        normal_column.button(
            "Fill normal day example", key="example_normal_day",
            help="Fill seven recent daily readings and a typical current day.",
            on_click=set_waste_example, args=("normal",), width="stretch",
        )
        high_column.button(
            "Fill high day example", key="example_high_day",
            help="Fill the same recent readings and a much higher current day.",
            on_click=set_waste_example, args=("high",), width="stretch",
        )

        waste_left, waste_right = st.columns(2)
        with waste_left:
            waste_city = st.selectbox(
                "City", CITIES, key="waste_city",
                help="Choose the city where the meter is located.",
            )
            waste_date = st.date_input(
                "Date", value=date.today(), key="waste_date",
                help="Choose the date of today's meter reading.",
            )
            today_kwh = st.number_input(
                "Today's electricity use (kWh)", min_value=0.0, value=0.0,
                step=0.1, key="today_kwh",
                help="Today's use: today's meter reading minus yesterday's.",
            )
            previous_text = st.text_area(
                "Previous 7 days of daily kWh (comma separated)",
                placeholder="12.4, 10.8, 11.6, 13.1, 9.7, 12.0, 11.2",
                key="previous_text",
                help="Enter daily kWh from the previous week; at least four values are required.",
            )
        with waste_right:
            outage_minutes = st.number_input(
                "Outage minutes today", min_value=0, max_value=1440,
                value=0, step=15, key="outage_minutes",
                help="Use your outage record or estimate; outages of 240 minutes or more suppress alerts.",
            )
            daily_weather = _load_climatology(
                daily_climatology_path, waste_city, waste_date.month
            )
            use_waste_temp_override = st.checkbox(
                "Override today's climatology temperature", value=False,
                key="use_waste_temp_override",
                help="Enable only when you have a better temperature estimate for this date.",
            )
            waste_temperature = st.number_input(
                f"Temperature (°C; climatology {daily_weather['temp_mean']:.1f})",
                value=float(daily_weather["temp_mean"]), step=0.5,
                key="waste_temperature", disabled=not use_waste_temp_override,
                help="Use a local weather reading or forecast for the selected date.",
            )

        parsed_previous: list[float] = []
        invalid_previous = False
        for token in previous_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError:
                invalid_previous = True
                continue
            if not np.isfinite(value) or value < 0:
                invalid_previous = True
                continue
            parsed_previous.append(value)
        if invalid_previous:
            st.warning("Invalid previous-day entries were ignored; enter non-negative kWh values.")
        if len(parsed_previous) < 4:
            st.info("Enter at least 4 valid previous daily kWh values to run the waste check.")
        else:
            waste_bundle = models["waste"]
            expected_row = build_daily_expected_feature_row(
                waste_bundle["expected_features"], waste_city, waste_date,
                parsed_previous[-7:], daily_weather,
                temperature_override=(
                    float(waste_temperature) if use_waste_temp_override else None
                ),
            )
            expected_kwh = predict_kwh_from_log_target(
                waste_bundle["expected_model"], expected_row
            )
            residual = math.log1p(float(today_kwh)) - math.log1p(expected_kwh)
            if waste_city not in waste_bundle["city_mad"]:
                raise ValueError(f"No residual calibration found for city '{waste_city}'")
            median, mad = waste_bundle["city_mad"][waste_city]
            if float(mad) > 0:
                robust_z = (residual - float(median)) / float(mad)
            else:
                robust_z = math.inf if residual > float(median) else 0.0
            flagged = bool(
                outage_minutes < 240 and robust_z > float(waste_bundle["z_threshold"])
            )
            excess = float(today_kwh) - expected_kwh
            if flagged:
                st.markdown(
                    '<div class="status-banner waste">Unusually high - possible waste</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "- Check the AC set point and hours of use.\n"
                    "- Check whether a water heater/geyser or pump was left on.\n"
                    "- Check standby loads and appliances that may have been left running."
                )
            else:
                st.markdown(
                    '<div class="status-banner normal">Looks normal</div>',
                    unsafe_allow_html=True,
                )
                if outage_minutes >= 240:
                    st.caption("No alert is raised because an outage of at least 240 minutes suppresses flags.")

            waste_cards = st.columns(4)
            waste_values = (
                ("Expected", f"{expected_kwh:,.1f} kWh", "Weather-adjusted expected use", "green"),
                ("Actual", f"{today_kwh:,.1f} kWh", "Today's entered use", ""),
                ("Excess", f"{excess:,.1f} kWh", "Actual minus expected", "orange"),
                ("Robust z-score", f"{robust_z:.2f}" if np.isfinite(robust_z) else "∞",
                 f"Alert threshold: {waste_bundle['z_threshold']}", "red" if flagged else "green"),
            )
            for column, (title, value, detail, variant) in zip(waste_cards, waste_values):
                with column:
                    metric_card(title, value, detail, variant)

            waste_chart = pd.DataFrame([
                {"Measure": "Expected", "kWh": expected_kwh},
                {"Measure": "Actual", "kWh": float(today_kwh)},
            ])
            bar_chart = alt.Chart(waste_chart).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("Measure:N", title=None),
                y=alt.Y("kWh:Q", title="Energy (kWh)", scale=alt.Scale(domainMin=0)),
                color=alt.Color("Measure:N", scale=alt.Scale(
                    domain=["Expected", "Actual"], range=["#0F766E", "#F97316"]
                ), legend=None),
                tooltip=["Measure:N", alt.Tooltip("kWh:Q", format=",.1f")],
            ).properties(height=230)
            st.altair_chart(bar_chart, width="stretch")
        st.caption(
            "The alert is statistical (robust z-score of the difference from weather-adjusted "
            "expected use), validated only on synthetic injected waste (about 54% of injected "
            "days caught, about 29% of alerts correct), so treat it as a hint."
        )

    with about_tab:
        st.subheader("About this project")
        st.markdown(
            "- **Data:** REWDP electricity records from 59 houses across 6 Pakistani cities.\n"
            "- **Cold-start:** weather, season, and city features; holdout MAE about 171 kWh.\n"
            "- **Known-house:** adds the previous month's use; holdout MAE about 121 kWh.\n"
            "- **Limitations:** roughly one year of observations, limited coverage of high-consumption "
            "households, solar-house usage is gross rather than net consumption, and Skardu behaves "
            "differently from the other cities.\n"
            "- **Tariffs:** values are attributed to secondary websites and have not been checked "
            "against the official NEPRA document.\n"
            "- **Project:** Hafsa Shakeel; supervised by Dr. Mateen Yaqoob, FAST-NUCES Islamabad."
        )
        holdout_table = pd.DataFrame([
            {"Setting": "Cold-start Ridge", "Holdout MAE (kWh)": 171.20, "Holdout R2": 0.463},
            {"Setting": "Known-house Random Forest", "Holdout MAE (kWh)": 121.21, "Holdout R2": 0.355},
        ])
        st.dataframe(
            holdout_table.style.format({
                "Holdout MAE (kWh)": "{:.2f}", "Holdout R2": "{:.3f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption("These are estimates, not official electricity bills.")

    st.caption(
        "Hafsa Shakeel (26i-8017) | FAST-NUCES Islamabad | "
        "Supervisor: Dr. Mateen Yaqoob"
    )


if __name__ == "__main__":
    main()
