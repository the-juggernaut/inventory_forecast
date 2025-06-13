import numpy as np
import pandas as pd
import os
from pathlib import Path

class FeatureExtractor:
    """
    Feature engineering pipeline for time series forecasting tasks like:
    - Demand prediction
    - Stockout risk classification

    Features:
    - Lag, rolling, expanding stats
    - Price volatility
    - Calendar features (dow, month, season, quarter)
    - Event impact features
    - Optional: simulated stock level & stockout risk flags
    """

    EVENT_CATEGORIES = ["Sporting", "Cultural", "National", "Religious"]
    KEY_EVENTS = [
        "ValentinesDay",
        "Easter",
        "Christmas",
        "SuperBowl",
        "IndependenceDay",
        "Thanksgiving",
        "NewYear",
        "Mother's day",
    ]
    LAGS = [7, 14, 28]
    ROLLING_WINDOWS = [7, 14, 28, 56]

    def __init__(
        self,
        calendar_df,
        sell_prices_df,
        sales_df,
        task_type="demand",
        stock_threshold=10,
        initial_stock=1000,
    ):
        self.calendar = calendar_df.copy()
        self.prices = sell_prices_df.copy()
        self.sales = sales_df.copy()

        self.task_type = task_type  # "demand" or "stockout"
        self.stock_threshold = stock_threshold
        self.initial_stock = initial_stock

        # Preprocess calendar
        self.calendar["date"] = pd.to_datetime(self.calendar["date"])
        self.calendar.drop(
            columns=[col for col in self.calendar.columns if "snap" in col],
            errors="ignore",
            inplace=True,
        )

    @staticmethod
    def assign_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        if month in [9, 10, 11]:
            return "Fall"

    @staticmethod
    def reduce_mem_usage(df):
        for col in df.columns:
            col_type = df[col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.api.types.is_integer_dtype(col_type):
                    if (
                        c_min >= np.iinfo(np.int8).min
                        and c_max <= np.iinfo(np.int8).max
                    ):
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min >= np.iinfo(np.int16).min
                        and c_max <= np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min >= np.iinfo(np.int32).min
                        and c_max <= np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                else:
                    if (
                        c_min >= np.finfo(np.float16).min
                        and c_max <= np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min >= np.finfo(np.float32).min
                        and c_max <= np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
            elif pd.api.types.is_object_dtype(col_type):
                df[col] = df[col].astype("category")
        return df

    def prepare_features(self):
        id_vars = ["item_id", "dept_id", "cat_id"]
        value_vars = [col for col in self.sales.columns if col.startswith("d_")]

        # Reshape to long format
        sales_long = self.sales.melt(
            id_vars=id_vars, value_vars=value_vars, var_name="d", value_name="sales"
        )

        # Merge with calendar and prices
        df = sales_long.merge(self.calendar, on="d", how="left")
        df = df.merge(self.prices, on=["item_id", "wm_yr_wk"], how="left")
        df.sort_values(by=["item_id", "date"], inplace=True)

        # Fill forward price
        df["sell_price"] = df.groupby("item_id")["sell_price"].ffill()

        # Lag features (for demand and stockout)
        for lag in self.LAGS:
            df[f"lag_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

        # Rolling stats (for demand and stockout)
        for window in self.ROLLING_WINDOWS:
            df[f"rolling_mean_{window}"] = df.groupby("item_id")["sales"].transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
            df[f"rolling_std_{window}"] = df.groupby("item_id")["sales"].transform(
                lambda x: x.shift(1).rolling(window).std()
            )

        # Expanding mean (for trend)
        df["expanding_mean"] = df.groupby("item_id")["sales"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Price change features
        df["price_change_pct"] = df.groupby("item_id")["sell_price"].pct_change()
        df["rolling_price_std_4w"] = df.groupby("item_id")["sell_price"].transform(
            lambda x: x.rolling(28).std()
        )

        # Time-based features
        df["day_of_week"] = df["date"].dt.weekday
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year

        # Season
        df["season"] = df["month"].apply(self.assign_season)
        df = pd.get_dummies(df, columns=["season"], prefix="season")

        # Day of week
        df = pd.get_dummies(df, columns=["day_of_week"], prefix="dow")

        # Events
        df["event_type_1"] = df["event_type_1"].fillna("None")
        df["event_type_2"] = df["event_type_2"].fillna("None")
        df["event_name_1"] = df["event_name_1"].fillna("None")
        df["event_name_2"] = df["event_name_2"].fillna("None")

        for cat in self.EVENT_CATEGORIES:
            df[f"is_{cat.lower()}"] = (
                (df["event_type_1"] == cat) | (df["event_type_2"] == cat)
            ).astype("int")

        df["has_event"] = (
            (df["event_type_1"] != "None") | (df["event_type_2"] != "None")
        ).astype("int")

        for event in self.KEY_EVENTS:
            col = f"is_{event.lower().replace(' ', '_').replace('\'', '')}"
            df[col] = (
                (df["event_name_1"] == event) | (df["event_name_2"] == event)
            ).astype("int")

        # Task-specific processing
        if self.task_type == "stockout":
            # Simulate stock level and risk
            df["cumulative_sales"] = df.groupby("item_id")["sales"].cumsum()
            df["stock_level"] = self.initial_stock - df["cumulative_sales"]
            df["stockout_risk"] = (df["stock_level"] <= self.stock_threshold).astype(
                int
            )

        # Drop unneeded columns
        df.drop(
            columns=[
                "d",
                "wm_yr_wk",
                "event_name_1",
                "event_name_2",
                "weekday",
                "date",
            ],
            inplace=True,
            errors="ignore",
        )

        # Reduce memory usage
        df = self.reduce_mem_usage(df)

        return df
