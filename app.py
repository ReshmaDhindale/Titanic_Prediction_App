# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Predictor", layout="wide")

# ---- Sidebar Input Form ----
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", width=250)
st.sidebar.header("🧾 Passenger Details")

pclass = st.sidebar.radio("Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 29)
sibsp = st.sidebar.selectbox("Siblings / Spouses Aboard", list(range(0, 6)))
parch = st.sidebar.selectbox("Parents / Children Aboard", list(range(0, 6)))
fare = st.sidebar.slider("Fare Paid ($)", 0.0, 600.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode
sex_code = 0 if sex == "male" else 1
embarked_code = {"S": 0, "C": 1, "Q": 2}[embarked]

input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_code],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_code]
})

# ---- Page Tabs ----
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Explore Model", "📖 About"])

with tab1:
    st.title("🎯 Titanic Survival Prediction")
    st.markdown("Enter passenger details in the sidebar, then click below to predict.")

    # 👁️ Show input summary as a table
    st.markdown("### 👤 Passenger Input Summary")
    show_data = input_df.copy()
    show_data.columns = [
        "Passenger Class",
        "Sex (0 = Male, 1 = Female)",
        "Age",
        "Siblings/Spouses",
        "Parents/Children",
        "Fare Paid ($)",
        "Embarked (0=S, 1=C, 2=Q)"
    ]
    st.table(show_data.T)

    # 🚀 Predict button
    if st.button("🚀 Predict Now"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        # 🎯 Prediction result
        result_text = "Survived" if pred == 1 else "Did Not Survive"
        if pred == 1:
            st.success("🎉 The model predicts this passenger **would survive** ✅")
        else:
            st.error("💀 The model predicts this passenger **would not survive** ❌")

        # 📊 Prediction probability chart
        st.markdown("### 🔍 Prediction Probabilities")
        st.bar_chart(pd.DataFrame({
            "Survival Probability": [prob[0], prob[1]]
        }, index=["Did Not Survive", "Survived"]))

        # 🧠 Explanation table
        st.markdown("### 🧠 How Your Data Affects Survival")
        explanation_data = [
            ("Passenger Class", pclass, "✅ 1st class had better survival chances" if pclass == 1 else "❌ Lower class had worse outcomes"),
            ("Sex", sex, "✅ Females survived more" if sex == "female" else "❌ Males had lower survival"),
            ("Age", age, "✅ Young children were prioritized" if age < 10 else "⚠️ Older passengers had mixed survival"),
            ("Siblings/Spouses", sibsp, "✅ Traveling with 1 person was safer" if sibsp <= 1 else "❌ Larger groups faced risk"),
            ("Parents/Children", parch, "✅ Small families had higher survival" if parch <= 1 else "⚠️ Larger families varied"),
            ("Fare", fare, "✅ Higher fare = more likely 1st class" if fare > 50 else "❌ Low fare = lower survival"),
            ("Embarked", embarked, "✅ C and S had more survivors" if embarked in ["C", "S"] else "❌ Fewer survivors from Q"),
        ]
        expl_df = pd.DataFrame(explanation_data, columns=["Feature", "Your Input", "Effect on Survival"])
        st.dataframe(expl_df)

        # 📁 Download as CSV
        st.markdown("### 📁 Download Your Prediction")
        output_data = input_df.copy()
        output_data["Prediction"] = result_text
        csv = output_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="titanic_prediction.csv",
            mime="text/csv"
        )




with tab2:
    st.title("🔍 Behind the Scenes: How the Model Thinks")
    
    st.markdown("""
    Ever wondered how the model decides if someone survives the Titanic disaster?

    This chart shows which passenger details (called **features**) are most important to the model overall — based on training with real Titanic data.
    """)

    # Friendly feature names
    display_features = [
        "Passenger Class (1st, 2nd, 3rd)",
        "Sex (Male/Female)",
        "Age (in years)",
        "Siblings/Spouses Aboard",
        "Parents/Children Aboard",
        "Ticket Fare",
        "Embarked Port"
    ]

    importances = model.feature_importances_

    # Create a DataFrame for visualization
    feat_df = pd.DataFrame({
        "Feature": display_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="crest", ax=ax)

    # Add percentage labels
    for i, val in enumerate(feat_df["Importance"]):
        ax.text(val + 0.005, i, f"{val:.2%}", va='center', color='black', fontsize=9)

    ax.set_title("🧠 What the Model Pays Attention To")
    ax.set_xlabel("Importance (higher = more impact on survival)")
    ax.set_ylabel("")

    st.pyplot(fig)

    # Extra friendly explanation
    st.success("""
    ✅ **Quick Guide to Read the Chart**:

    - The **longer the bar**, the more the model relies on that feature when predicting.
    - For example, **Sex** usually has a strong impact — females had a higher chance of survival.
    - **Class and Fare** also matter — 1st class passengers and those who paid more had better chances.

    This view is for **overall model behavior**, not just your current input.
    """)


with tab3:
    st.title("About This App")
    st.markdown("""
    - 🚢 This app predicts Titanic passenger survival using a trained Random Forest classifier.
    - 📊 Built with Python and Streamlit.
    - 🎨 Designed for interactivity and clarity.
    - 💡 Enter data in the sidebar and view results instantly.
    """)
    

