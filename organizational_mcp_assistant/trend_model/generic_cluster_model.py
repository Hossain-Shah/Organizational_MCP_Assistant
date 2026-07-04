import pickle
import pandas as pd


class GenericClusterModel:
    """
    Generic prediction wrapper around your saved attendance trend model.
    This keeps the external MCP name generic instead of attendance-specific.
    """

    def __init__(self):
        self.student_models = {}
        self.trend_df = pd.DataFrame()
        self.target_col = "attendance_status"
        self.date_col = "attendance_date"
        self.student_col = "student_id"
        self.threshold = 0.01

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.student_models = data.get("student_models", {})
        self.trend_df = data.get("trend_df", pd.DataFrame())
        self.target_col = data.get("target_col", "attendance_status")
        self.date_col = data.get("date_col", "attendance_date")
        self.student_col = data.get("student_col", "student_id")
        self.threshold = data.get("threshold", 0.01)

    def predict(self, new_df: pd.DataFrame) -> pd.DataFrame:
        if new_df.empty:
            return pd.DataFrame()

        df = new_df.copy()

        df[self.student_col] = df[self.student_col].astype(str).str.strip()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        df = df.dropna(subset=[self.student_col, self.date_col, self.target_col])

        # Keep latest record per student per day
        df["_day"] = df[self.date_col].dt.date
        df = (
            df.sort_values(self.date_col)
            .groupby([self.student_col, "_day"])
            .tail(1)
            .drop(columns=["_day"])
            .reset_index(drop=True)
        )

        full_trend_rows = []

        if self.trend_df is not None and not self.trend_df.empty:
            full_trend_rows.extend(self.trend_df.to_dict("records"))

        for student_id, val_group in df.groupby(self.student_col):
            val_group = val_group.sort_values(self.date_col).reset_index(drop=True)

            hist = self.student_models.get(student_id, pd.DataFrame())

            if not hist.empty:
                hist = hist.copy()
                hist[self.date_col] = pd.to_datetime(hist[self.date_col], errors="coerce")
                hist = hist.sort_values(self.date_col)

                start_date = hist[self.date_col].min()
                hist["days_since_start"] = (hist[self.date_col] - start_date).dt.days

                x = hist["days_since_start"].values
                y = pd.to_numeric(hist[self.target_col], errors="coerce").values

                cum_n = len(hist)
                cum_x = x.sum()
                cum_y = y.sum()
                cum_x2 = (x ** 2).sum()
                cum_xy = (x * y).sum()
            else:
                start_date = val_group[self.date_col].min()
                hist = pd.DataFrame()

                cum_n = 0
                cum_x = 0
                cum_y = 0
                cum_x2 = 0
                cum_xy = 0

            for _, row in val_group.iterrows():
                x_new = (row[self.date_col] - start_date).days
                y_new = pd.to_numeric(row[self.target_col], errors="coerce")

                cum_n += 1
                cum_x += x_new
                cum_y += y_new
                cum_x2 += x_new ** 2
                cum_xy += x_new * y_new

                denominator = cum_n * cum_x2 - cum_x ** 2

                if denominator != 0:
                    slope = (cum_n * cum_xy - cum_x * cum_y) / denominator
                else:
                    slope = 0.0

                if cum_n < 3:
                    label = ""
                elif slope > self.threshold:
                    label = "improving"
                elif slope < -self.threshold:
                    label = "declining"
                else:
                    label = "stable"

                full_trend_rows.append({
                    self.student_col: student_id,
                    "date": row[self.date_col].date(),
                    "target": row[self.target_col],
                    "generic_cluster_score": slope,
                    "generic_cluster_label": label,
                    "absence_reason": row.get("absence_reason"),
                    "school_id": row.get("school_id"),
                    "year": row.get("year"),
                    "month": row.get("month"),
                    "country_name": row.get("country_name"),
                    "academic_year": row.get("academic_year"),
                })

            self.student_models[student_id] = pd.concat(
                [hist, val_group],
                ignore_index=True
            )

        result_df = pd.DataFrame(full_trend_rows)

        if result_df.empty:
            return result_df

        result_df["date"] = pd.to_datetime(result_df["date"]).dt.date

        result_df = (
            result_df.sort_values(["date", self.student_col])
            .groupby([self.student_col, "date"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        return result_df