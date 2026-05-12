
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

st.title("Email Retention Dashboard")

# upload file
uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file is not None:

    # loading data
    df = pd.read_csv(uploaded_file, sep=';')

    # timestamps
    cols = ['send_ts', 'delivery_ts', 'read_ts', 'click_ts']

    for col in cols:
        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')

    # flags
    df['read'] = df['read_ts'].notna()
    df['clicked'] = df['click_ts'].notna()

    df['not_free_credits'] = df['not_free_credits'].fillna(0)

    df['paid'] = df['not_free_credits'] > 0

    # tabs
    tab1, tab2 = st.tabs(["Monitoring", "A/B Analysis"])

    # MONITORING


    with tab1:

        st.header("Monitoring")

        # metrics
        open_rate = df['read'].mean()
        click_rate = df['clicked'].mean()
        paid_rate = df['paid'].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric("Open Rate", f"{open_rate:.2%}")
        col2.metric("CTR", f"{click_rate:.2%}")
        col3.metric("Paid Rate", f"{paid_rate:.2%}")

        # trend
        df['date_parsed'] = df['send_ts'].dt.date

        trend = df.groupby('date_parsed')['clicked'].mean()

        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(trend.index, trend.values, marker='o')

        ax.set_title("CTR Trend")

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        # response analysis
        st.subheader("Response Type CTR")

        response_ctr = df.groupby('response')['clicked'].mean()

        st.bar_chart(response_ctr)

        # AI summary
        st.subheader("AI Summary")

        if click_rate > 0.04:
            st.success("Email channel shows strong engagement performance.")

        else:
            st.warning("CTR is relatively low and may indicate audience fatigue.")

    # A/B ANALYSIS

    with tab2:

        st.header("A/B Analysis")

        group = st.selectbox(
            "Select Test Group",
            ['group_1', 'group_2', 'group_3', 'group_4']
        )

        ab = df[df[group].notna()]

        metrics = ab.groupby(group)['clicked'].mean()

        st.subheader("CTR")

        st.bar_chart(metrics)

        # z-test
        test_group = ab[ab[group] == 'Test']

        control_group = ab[ab[group] == 'Control']

        # counts
        clicks_test = test_group['clicked'].sum()

        clicks_control = control_group['clicked'].sum()

        n_test = len(test_group)

        n_control = len(control_group)

        # ctr
        ctr_test = clicks_test / n_test

        ctr_control = clicks_control / n_control

        # pooled probability
        p_pool = (clicks_test + clicks_control) / (n_test + n_control)

        # standard error
        se = np.sqrt(
            p_pool * (1 - p_pool) * (1/n_test + 1/n_control)
        )

        # z-score
        z = (ctr_test - ctr_control) / se

        # p-value
        p_val = 2 * (1 - norm.cdf(abs(z)))

        st.write(f"P-value: {p_val:.4f}")

        # recommendation
        st.subheader("Recommendation")

        if p_val < 0.05 and ctr_test > ctr_control:

            st.success(
                "Test performs better and should be scaled."
            )

        else:

            st.warning(
                "No statistically significant improvement detected."
            )
