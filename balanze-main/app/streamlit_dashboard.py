yes"""
Interactive Streamlit dashboard for gait analysis visualization (Step 10)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
	page_title="Gait & Balance Analysis Dashboard",
	page_icon="🚶",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
<style>
.metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
.status-normal { color: green; font-weight: bold; }
.status-warning { color: orange; font-weight: bold; }
.status-danger { color: red; font-weight: bold; }
</style>
""",
	unsafe_allow_html=True,
)

st.sidebar.title("🚶 Gait Analysis System")
page = st.sidebar.radio(
	"Navigation",
	["Dashboard", "Upload Video", "Real-time Analysis", "Patient Records", "Reports"],
)

if page == "Dashboard":
	st.title("📊 Analysis Dashboard")
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Analyses", "1,234", "+12 today")
	with col2:
		st.metric("Avg Gait Quality", "0.87", "+5%")
	with col3:
		st.metric("High Risk Cases", "23", "↑ 3")
	with col4:
		st.metric("Referrals Sent", "18", "This week")

	st.divider()
	st.subheader("Recent Analyses")
	recent_data = pd.DataFrame(
		{
			"Patient ID": ["P001", "P002", "P003", "P004"],
			"Age": [65, 72, 58, 70],
			"Gait Type": ["Normal", "Parkinsonian", "Normal", "Cerebellar"],
			"Balance": ["Stable", "Impaired", "Stable", "Severely Impaired"],
			"Disorder Risk": ["Low", "High", "Low", "High"],
			"Timestamp": [
				datetime.now() - timedelta(hours=1),
				datetime.now() - timedelta(hours=3),
				datetime.now() - timedelta(hours=5),
				datetime.now() - timedelta(hours=8),
			],
		}
	)
	st.dataframe(recent_data, use_container_width=True)

	st.subheader("30-Day Trends")
	col1, col2 = st.columns(2)
	with col1:
		dates = pd.date_range("2024-01-01", periods=30, freq="D")
		data = pd.DataFrame(
			{
				"Date": dates,
				"Avg Gait Quality": np.random.uniform(0.75, 0.95, 30),
				"Avg Balance Score": np.random.uniform(0.60, 0.90, 30),
			}
		)
		fig = px.line(
			data,
			x="Date",
			y="Avg Gait Quality",
			title="Gait Quality Trend",
			markers=True,
			line_shape="spline",
		)
		fig.update_layout(hovermode="x unified")
		st.plotly_chart(fig, use_container_width=True)
	with col2:
		disorder_dist = pd.DataFrame(
			{
				"Disorder": ["Normal", "Parkinsons", "Cerebellar", "Stroke", "Neuropathy"],
				"Count": [45, 23, 18, 12, 8],
			}
		)
		fig = px.pie(
			disorder_dist,
			values="Count",
			names="Disorder",
			title="Disorder Distribution",
			hole=0.4,
		)
		st.plotly_chart(fig, use_container_width=True)

elif page == "Upload Video":
	st.title("📹 Upload Gait Video for Analysis")
	col1, col2 = st.columns([2, 1])
	with col1:
		uploaded_file = st.file_uploader("Choose video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
		if uploaded_file is not None:
			st.video(uploaded_file)
			with st.expander("Patient Information"):
				col1a, col2a = st.columns(2)
				with col1a:
					st.text_input("Patient ID", "P001")
					st.number_input("Age", 20, 100, 65)
				with col2a:
					st.selectbox("Gender", ["Male", "Female", "Other"])
					st.number_input("Weight (kg)", 40, 150, 75)
			if st.button("🔍 Analyze Video", use_container_width=True):
				st.success("✓ Analysis Complete! (demo)")
				col1b, col2b, col3b = st.columns(3)
				with col1b:
					st.metric("Gait Type", "Normal", f"{0.92:.1%}")
				with col2b:
					st.metric("Balance Status", "Stable", f"{0.88:.1%}")
				with col3b:
					st.metric("Disorder Risk", "None", f"{0.95:.1%}")

elif page == "Real-time Analysis":
	st.title("🎥 Real-time Analysis")
	st.info("Start webcam analysis to monitor gait in real-time (demo)")
	duration = st.slider("Analysis Duration (seconds)", 10, 60, 20)
	if st.button("▶️ Start Webcam Analysis", use_container_width=True):
		placeholder = st.empty()
		chart_placeholder = st.empty()
		progress_bar = st.progress(0)
		timestamps: list = []
		gait_scores: list = []
		balance_scores: list = []
		for i in range(duration):
			timestamps.append(datetime.now())
			gait_scores.append(float(np.clip(0.8 + np.random.normal(0, 0.05), 0, 1)))
			balance_scores.append(float(np.clip(0.75 + np.random.normal(0, 0.08), 0, 1)))
			with placeholder.container():
				colx, coly, colz = st.columns(3)
				with colx:
					st.metric("Current Gait Quality", f"{gait_scores[-1]:.2%}")
				with coly:
					st.metric("Balance Stability", f"{balance_scores[-1]:.2%}")
				with colz:
					st.metric("Frame Count", f"{i+1}/{duration}")
			if i % 5 == 0:
				df_live = pd.DataFrame(
					{"Time": timestamps, "Gait Quality": gait_scores, "Balance Stability": balance_scores}
				)
				fig = px.line(df_live, x="Time", y=["Gait Quality", "Balance Stability"], title="Real-time Metrics", markers=True)
				chart_placeholder.plotly_chart(fig, use_container_width=True)
			progress_bar.progress((i + 1) / max(duration, 1))
			time.sleep(1)
		st.success("✓ Real-time analysis complete!")

elif page == "Patient Records":
	st.title("👥 Patient Records")
	patients_df = pd.DataFrame(
		{
			"Patient ID": ["P001", "P002", "P003", "P004", "P005"],
			"Name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams", "Charlie Brown"],
			"Age": [65, 72, 58, 70, 62],
			"Last Analysis": [
				datetime.now() - timedelta(days=1),
				datetime.now() - timedelta(days=3),
				datetime.now() - timedelta(days=7),
				datetime.now() - timedelta(days=14),
				datetime.now() - timedelta(days=30),
			],
			"Gait Status": ["Normal", "Parkinsonian", "Normal", "Cerebellar", "Hemiplegic"],
			"Risk Level": ["Low", "High", "Low", "High", "Medium"],
		}
	)
	st.dataframe(patients_df, use_container_width=True, hide_index=True)

elif page == "Reports":
	st.title("📋 Reports & Exports")
	report_type = st.selectbox("Select report type", ["Patient Summary", "Clinical Summary", "Trend Analysis", "Batch Export"])
	if report_type == "Patient Summary":
		st.info("Generate patient report via backend or local utilities.")

if __name__ == "__main__":
	st.info("Gait & Balance Analysis Dashboard Ready!")


