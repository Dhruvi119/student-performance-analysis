import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("student_performance_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

df = load_data()
model = load_model()

st.title("🎓 Student Performance Analysis Dashboard")
st.write("Explore student data, academic performance, and predictions.")

page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Performance Analysis", "Student Data", "Result Prediction"]
)

if page == "Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Pass Students", int((df["Result"] == "Pass").sum()))
    col3.metric("Fail Students", int((df["Result"] == "Fail").sum()))

    st.subheader("Student Records")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Performance Analysis":
    st.header("Performance Analysis")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Result", ax=ax)
    ax.set_title("Pass vs Fail Students")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="Study_Hours",
        y="Percentage",
        hue="Result",
        ax=ax
    )
    ax.set_title("Study Hours vs Percentage")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Attendance", kde=True, ax=ax)
    ax.set_title("Attendance Distribution")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Student Data":
    result_filter = st.selectbox(
        "Filter Students",
        ["All", "Pass", "Fail"]
    )

    filtered = df if result_filter == "All" else df[df["Result"] == result_filter]
    st.dataframe(filtered, use_container_width=True)

elif page == "Result Prediction":
    st.header("Predict Student Result")
    st.warning(
        "Educational demonstration using synthetic data. "
        "Predictions are estimates, not guaranteed outcomes."
    )

    gender = st.selectbox("Gender", ["Female", "Male"])
    attendance = st.slider("Attendance (%)", 40, 100, 75)
    study_hours = st.slider("Study Hours per Day", 0.5, 8.0, 3.0, 0.5)

    if st.button("Predict Result"):
        student = pd.DataFrame([{
            "Gender": gender,
            "Attendance": attendance,
            "Study_Hours": study_hours
        }])

        prediction = model.predict(student)[0]
        st.subheader(f"Predicted Result: {prediction}")

        probabilities = model.predict_proba(student)[0]
        for label, probability in zip(model.classes_, probabilities):
            st.write(f"{label}: {probability:.1%}")
