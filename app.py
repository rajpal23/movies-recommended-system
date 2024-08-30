import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and models
new_movies_df = pd.read_csv(r'C:\Users\Dell\Desktop\My work\VS CODE PROGRAM\machine learning\project\movies recomended system\movies_df.csv')

# Check if 'tags' column exists
if 'tags' not in new_movies_df.columns:
    st.error("The 'tags' column is missing from the CSV file.")
    st.stop()

# Preprocess the 'tags' column (clean and prepare text data)
new_movies_df["tags"] = new_movies_df["tags"].fillna("").apply(lambda x: x.lower())

# Instantiate CountVectorizer and fit_transform on the tags column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_movies_df["tags"]).toarray()

# Compute the cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save CountVectorizer and similarity matrix to pickle files
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

# Load the saved CountVectorizer and similarity matrix
with open('count_vectorizer.pkl', 'rb') as f:
    loaded_cv = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    loaded_similarity = pickle.load(f)

def recommend(movie):
    movie = movie.strip().lower()
    matches = new_movies_df[new_movies_df["title"].str.strip().str.lower() == movie]
    if matches.empty:
        return ["Movie not found in the dataset."]
    
    movie_idx = matches.index[0]
    distances = loaded_similarity[movie_idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_movies_df.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Streamlit app layout
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
   
st.title("Movie Recommendation System")
st.markdown(
    """
    <style>
    .main {
        color: #333;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
   
st.sidebar.header("Movie Recommendation System")
movie_title = st.sidebar.text_input("Enter a movie title")

if st.sidebar.button("Recommend"):
    if movie_title:
        recommendations = recommend(movie_title)
        st.write(f"### Recommendations for '{movie_title}':")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("Please enter a movie title.")
