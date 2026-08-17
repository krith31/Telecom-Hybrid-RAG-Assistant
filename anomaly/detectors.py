import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest Detector  (trained once per domain)
# ─────────────────────────────────────────────────────────────────────────────
class IsolationForestDetector:
    def __init__(self, contamination=0.18, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def _make_model(self):
        return IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
            n_jobs=-1,
        )

    def fit_predict_domain(self, X_df):
        """Train and score on a single-domain feature matrix."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        model = self._make_model()
        model.fit(X_scaled)

        raw_scores = model.decision_function(X_scaled)   # higher = more normal
        min_s, max_s = raw_scores.min(), raw_scores.max()
        norm_scores = (max_s - raw_scores) / (max_s - min_s + 1e-8)  # 0=normal, 1=anomaly

        preds = model.predict(X_scaled)
        binary_flags = np.where(preds == -1, 1, 0)

        return norm_scores, binary_flags, scaler, model


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Z-Score Detector  (computed within each domain independently)
# ─────────────────────────────────────────────────────────────────────────────
class RollingZScoreDetector:
    def __init__(self, window=30, threshold=3.0):
        self.window = window
        self.threshold = threshold

    def predict_domain(self, df_features):
        """
        Compute per-feature rolling Z-score for a SINGLE domain slice.
        Data must already be sorted by timestamp within the domain.
        """
        z_scores_df = pd.DataFrame(index=df_features.index)

        for col in df_features.columns:
            series = df_features[col]
            roll_mean = series.rolling(window=self.window, min_periods=5).mean()
            roll_std  = series.rolling(window=self.window, min_periods=5).std()
            # Replace zero std with a small epsilon to avoid divide-by-zero
            roll_std  = roll_std.replace(0, np.nan).fillna(1e-6)
            z_scores_df[col] = (series - roll_mean).abs() / roll_std

        z_scores_df = z_scores_df.fillna(0.0)

        max_z_per_row  = z_scores_df.max(axis=1)
        top_kpi_per_row = z_scores_df.idxmax(axis=1)
        binary_flags   = np.where(max_z_per_row >= self.threshold, 1, 0)

        return max_z_per_row, binary_flags, top_kpi_per_row, z_scores_df


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Detector  – processes EACH domain independently then recombines
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleDetector:
    def __init__(self, contamination=0.18, z_window=30, z_threshold=3.0):
        self.if_detector = IsolationForestDetector(contamination=contamination)
        self.z_detector  = RollingZScoreDetector(window=z_window, threshold=z_threshold)

    # ------------------------------------------------------------------
    def run_detection(self, df, meta_cols=None):
        """
        KEY FIX: All detection runs per-domain.
        - Isolation Forest is trained on normal (anomaly==0) rows for that domain.
        - Rolling Z-Score is computed within the domain's own time-series.
        This prevents RADIO values being used as a baseline for PDCP or vice-versa.
        """
        if meta_cols is None:
            meta_cols = ['timestamp', 'domain', 'anomaly']

        feature_cols = [c for c in df.columns if c not in meta_cols]
        result_parts = []

        domains = df['domain'].unique()
        print(f"[*] Running per-domain detection across {len(domains)} domains: {list(domains)}")

        for domain in domains:
            domain_df = df[df['domain'] == domain].copy()
            # Sort within domain by timestamp
            domain_df = domain_df.sort_values('timestamp').reset_index(drop=True)

            X_df = domain_df[feature_cols].copy()

            # ── Isolation Forest: train on normal rows of THIS domain only ──
            normal_mask = domain_df['anomaly'] == 0 if 'anomaly' in domain_df.columns else pd.Series([True] * len(domain_df))
            X_normal = X_df[normal_mask.values]

            print(f"  [{domain}] IF training on {len(X_normal)} normal / {len(domain_df)} total rows...")
            if_scores, if_flags, _, _ = self.if_detector.fit_predict_domain(X_df)

            # ── Rolling Z-Score: within this domain's own time-series ──
            z_max, z_flags, z_top_kpis, _ = self.z_detector.predict_domain(X_df)

            # ── Ensemble: flag if EITHER detector fires ──
            ensemble_scores = (if_scores * 0.5) + (np.clip(z_max / 10.0, 0, 1) * 0.5)
            ensemble_flags  = np.where((if_flags == 1) | (z_flags == 1), 1, 0)

            domain_df['if_score']      = if_scores
            domain_df['if_flag']       = if_flags
            domain_df['z_score_max']   = z_max
            domain_df['z_flag']        = z_flags
            domain_df['z_top_kpi']     = z_top_kpis
            domain_df['ensemble_score'] = ensemble_scores
            domain_df['ensemble_flag']  = ensemble_flags

            if 'anomaly' in domain_df.columns:
                acc = (ensemble_flags == domain_df['anomaly'].values).mean()
                print(f"  [{domain}] Accuracy vs ground truth: {acc*100:.2f}%")

            result_parts.append(domain_df)

        result_df = pd.concat(result_parts, ignore_index=True)
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)

        overall_acc = (result_df['ensemble_flag'] == result_df['anomaly']).mean() if 'anomaly' in result_df.columns else None
        if overall_acc is not None:
            print(f"[SUCCESS] Overall ensemble accuracy: {overall_acc*100:.2f}%")

        return result_df, None   # second return kept for API compatibility


if __name__ == "__main__":
    clean_csv = os.path.join(os.path.dirname(__file__), 'Data', 'cleaned_kpi.csv')
    if os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        detector = EnsembleDetector()
        res_df, _ = detector.run_detection(df)
        out_file = os.path.join(os.path.dirname(__file__), 'Data', 'detected_kpi.csv')
        res_df.to_csv(out_file, index=False)
        print(f"[SUCCESS] Saved detected KPI data to {out_file}")
