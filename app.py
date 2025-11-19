import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Restaurant Recommendation System", layout="wide")

st.title("🍽️ Restaurant Recommendation System")
st.write("Upload your dataset → Train Model → Get Recommendations")

st.header("📤 Step 1: Upload Restaurant Dataset")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    st.write("Columns detected:", data.columns.tolist())

    possible_name_cols = ["name", "Name", "restaurant_name", "Restaurant Name", "rest_name"]
    possible_cuisine_cols = ["cuisines", "Cuisines", "Cuisine", "cuisine"]
    possible_location_cols = ["location", "Location", "place", "Place", "city"]
    possible_resttype_cols = ["rest_type", "Rest_Type", "restType", "type", "Type"]

    def detect_column(possible_cols):
        for col in possible_cols:
            if col in data.columns:
                return col
        return None

    name_col = detect_column(possible_name_cols)
    cuisine_col = detect_column(possible_cuisine_cols)
    location_col = detect_column(possible_location_cols)
    resttype_col = detect_column(possible_resttype_cols)

    if name_col is None:
        st.error("No valid restaurant name column found. Add a column named 'name'.")
    else:
        st.success(f"Restaurant name column detected: {name_col}")

    if cuisine_col is None: cuisine_col = name_col
    if location_col is None: location_col = name_col
    if resttype_col is None: resttype_col = name_col

    st.header("⚙️ Step 2: Train the Recommendation Model")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            data[cuisine_col] = data[cuisine_col].fillna("")
            data[location_col] = data[location_col].fillna("")
            data[resttype_col] = data[resttype_col].fillna("")

            data["combined_features"] = (
                data[cuisine_col].astype(str) + " " +
                data[location_col].astype(str) + " " +
                data[resttype_col].astype(str)
            )

            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(data["combined_features"])

            similarity = cosine_similarity(tfidf_matrix)

            os.makedirs("models", exist_ok=True)
            with open("models/similarity.pkl", "wb") as f:
                pickle.dump(similarity, f)

            st.success("Model training completed!")

    st.header("🔍 Step 3: Get Restaurant Recommendations")

    if name_col:
        restaurant_names = data[name_col].dropna().unique().tolist()
        user_choice = st.selectbox("Select a Restaurant:", restaurant_names)

        if st.button("Recommend Restaurants"):
            try:
                with open("models/similarity.pkl", "rb") as f:
                    similarity = pickle.load(f)

                index = data[data[name_col] == user_choice].index[0]
                distances = similarity[index]

                recommendations = sorted(
                    list(enumerate(distances)),
                    reverse=True,
                    key=lambda x: x[1]
                )[1:6]

                st.subheader("Recommended Restaurants:")
                for idx, score in recommendations:
                    st.write(f"- {data.iloc[idx][name_col]}")

            except FileNotFoundError:
                st.error("Train the model first.")
