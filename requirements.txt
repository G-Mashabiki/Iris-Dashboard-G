# app.py
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
page_title="Iris Interactive Dashboard",
layout="wide"
)
st.title("Interactive Iris Dashboard")
st.write("Explore the Iris dataset using interactive controls and Plotly visualizations.")
# Load data
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(
dict(enumerate(iris.target_names))
)
# Sidebar controls
st.sidebar.header("Filter Options")
species_selected = st.sidebar.multiselect(
"Select species:",
options=df["species"].unique(),
default=df["species"].unique()
)
x_axis = st.sidebar.selectbox(
"Select X-axis variable:",
options=iris.feature_names,
index=0
)
y_axis = st.sidebar.selectbox(
"Select Y-axis variable:",
options=iris.feature_names,
index=1
)
# Filter data
filtered_df = df[df["species"].isin(species_selected)]
# Key metrics
col1, col2, col3 = st.columns(3)
col1.metric("Observations", len(filtered_df))
col2.metric("Species Count", filtered_df["species"].nunique())
col3.metric("Avg Sepal Length", round(filtered_df["sepal length (cm)"].mean(), 2))
# Interactive scatter plot
fig = px.scatter(
filtered_df,
x=x_axis,
y=y_axis,
color="species",
title=f"{y_axis} vs {x_axis}",
hover_data=iris.feature_names
)
st.plotly_chart(fig, use_container_width=True)
# Distribution plot
st.subheader("Feature Distribution")
feature_selected = st.selectbox(
"Select feature:",
iris.feature_names
)
hist_fig = px.histogram(
filtered_df,
x=feature_selected,
color="species",
barmode="overlay",
opacity=0.7
)
st.plotly_chart(hist_fig, use_container_width=True)
# Data preview
with st.expander("View Data"):
 st.dataframe(filtered_df)

 #In the terminal, run
 #streamlit run dashboard.py
