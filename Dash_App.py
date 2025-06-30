# Import necessary libraries
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input
import numpy as np # Needed for array operations
import tensorflow as tf # Needed to load the Keras model
from sklearn.preprocessing import LabelEncoder # Needed for encoders

# --- Load Recommender System Components (replace mock data) ---
# Assuming your Jupyter notebook code is adapted here or imported

# 1. Load Data and create refined_dataset
# You will need to re-run the initial data loading and preprocessing steps
# to get `refined_dataset`, `user_enc`, `item_enc`, `min_rating`, `max_rating`

# Example (you'd copy this from your Jupyter notebook and adapt paths)
overall_stats = pd.read_csv('ml-100k/u.info', header=None)
column_names1 = ['user id','movie id','rating','timestamp']
ratings_dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
movie_dataset = items_dataset[['movie id','movie title']]
merged_dataset = pd.merge(ratings_dataset, movie_dataset, how='inner', on='movie id')
refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})

user_enc = LabelEncoder()
refined_dataset['user'] = user_enc.fit_transform(refined_dataset['user id'].values)
n_users = refined_dataset['user'].nunique()

item_enc = LabelEncoder()
refined_dataset['movie'] = item_enc.fit_transform(refined_dataset['movie title'].values)
n_movies = refined_dataset['movie'].nunique()

refined_dataset['rating'] = refined_dataset['rating'].values.astype(np.float32)
min_rating = min(refined_dataset['rating'])
max_rating = max(refined_dataset['rating'])

# 2. Load the trained model
# Make sure 'movie_recommender_model.h5' is in the same directory as your Dash app,
# or provide the full path.
try:
    model = tf.keras.models.load_model('movie_recommender_model.h5', compile=False)
    # Recompile the model if needed, especially if you saved with compile=False
    # model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    # Note: If your app crashes here, check the path and if the model was saved correctly.
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'movie_recommender_model.h5' exists and was saved correctly.")
    # Exit or handle gracefully if model cannot be loaded
    model = None # Set to None to prevent further errors

# 3. Adapt the recommender_system function for Dash
# This function will be called by your Dash callback
def get_dnn_recommendations(user_id_from_input, num_recommendations=10):
    if model is None:
        return pd.DataFrame() # Return empty if model not loaded

    # Ensure user_id_from_input is an integer (Dash input might be string)
    user_id_from_input = int(user_id_from_input)

    try:
        encoded_user_id = user_enc.transform([user_id_from_input])
    except ValueError:
        # Handle cases where user_id might not exist in the training data
        print(f"User ID {user_id_from_input} not found in training data.")
        return pd.DataFrame() # Return empty DataFrame

    seen_movies_encoded = refined_dataset[refined_dataset['user id'] == user_id_from_input]['movie'].tolist()

    # Get all possible movie encoded IDs from 0 to n_movies-1
    all_movie_encoded_ids = list(range(n_movies))

    # Identify unseen movies
    unseen_movies_encoded = [mid for mid in all_movie_encoded_ids if mid not in seen_movies_encoded]

    if not unseen_movies_encoded:
        print(f"User {user_id_from_input} has seen all movies or no unseen movies available.")
        return pd.DataFrame() # No recommendations if all movies are seen

    # Prepare model input
    # Repeat the encoded user ID for each unseen movie
    user_input_array = np.asarray([encoded_user_id[0]] * len(unseen_movies_encoded))
    movie_input_array = np.asarray(unseen_movies_encoded)

    model_input = [user_input_array, movie_input_array]

    # Predict ratings
    predicted_probabilities = model.predict(model_input)

    # In your Softmax model, `predicted_probabilities` is (num_unseen_movies, 9)
    # where 9 is the number of rating classes.
    # To get a single "score" for sorting, we'll take the max probability for each movie.
    # This assumes that a higher max probability indicates a higher likelihood of a preferred rating.
    predicted_scores = np.max(predicted_probabilities, axis=1)

    # Sort movies by predicted score in descending order
    sorted_indices = np.argsort(predicted_scores)[::-1]

    # Get the top N recommended movie encoded IDs
    top_n_unseen_movies_encoded = np.array(unseen_movies_encoded)[sorted_indices[:num_recommendations]]

    # Inverse transform to get movie titles
    recommended_movie_titles = item_enc.inverse_transform(top_n_unseen_movies_encoded)

    # Create a DataFrame for the recommendations (optional, but good for display)
    recommendations_df = pd.DataFrame({
        'Title': recommended_movie_titles,
        'Predicted Score': predicted_scores[sorted_indices[:num_recommendations]] # Corresponding scores
    })

    # Optional: Add genre information for display (requires merging with items_dataset or movie_dataset)
    # To do this cleanly, you'd need the movie_dataset or similar `items_dataset` loaded globally too
    # For now, let's just use the titles.
    # If you want genre, you'd load items_dataset globally and merge:
    # movie_titles_genres = items_dataset[['movie title', 'Action', ..., 'Western']]
    # recommendations_df = recommendations_df.merge(movie_titles_genres, on='Title', how='left')

    return recommendations_df

# Initialize the Dash app
app = Dash(__name__)

# --- App Layout ---
app.layout = html.Div(className="container mx-auto p-6 bg-gray-100 min-h-screen font-sans", children=[
    html.Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
    html.Script(src="https://cdn.tailwindcss.com"),
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
        rel="stylesheet"
    ),
    html.Div(children=[
        html.Style("""
            body { font-family: 'Inter', sans-serif; }
            .container { max-width: 1200px; }
            .card {
                background-color: white;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .recommendation-item {
                display: flex;
                justify-content: space-between;
                padding: 0.75rem 0;
                border-bottom: 1px solid #e5e7eb;
                align-items: center; /* Vertically align items */
            }
            .recommendation-item:last-child {
                border-bottom: none;
            }
            .movie-title {
                font-weight: 500;
                color: #333;
                flex-grow: 1; /* Allows title to take up more space */
            }
            .movie-score {
                font-weight: 600;
                color: #2563eb; /* A nice blue */
                margin-left: 1rem; /* Space between title and score */
            }
        """)
    ]),

    html.H1(
        "Movie Recommendation Dashboard (Softmax DNN)",
        className="text-4xl font-bold text-center text-gray-800 mb-8 mt-4"
    ),

    # Input Card: User Selection (changed from movie selection)
    html.Div(className="card", children=[
        html.H2("Enter User ID", className="text-2xl font-semibold text-gray-700 mb-4"),
        dcc.Input(
            id='user-id-input',
            type='number', # User IDs are numbers
            placeholder="Enter user ID (e.g., 1, 943)",
            value=1, # Default user ID
            min=int(refined_dataset['user id'].min()), # Min user ID from data
            max=int(refined_dataset['user id'].max()), # Max user ID from data
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        ),
        html.P(id='seen-movies-output', className="text-sm text-gray-500 mt-2")
    ]),

    # Output Card: Recommendations
    html.Div(className="card", children=[
        html.H2("Recommended Movies", className="text-2xl font-semibold text-gray-700 mb-4"),
        html.Div(id='recommendations-output', className="text-lg text-gray-600")
    ]),

    # Visualization Card: Genre Distribution (unchanged for now, but uses global df)
    html.Div(className="card", children=[
        html.H2("Movie Genre Distribution", className="text-2xl font-semibold text-gray-700 mb-4"),
        dcc.Graph(id='genre-distribution-chart')
    ]),
])

# --- Callbacks ---

# Callback to update seen movies and recommendations
@callback(
    Output('recommendations-output', 'children'),
    Output('seen-movies-output', 'children'), # Also output seen movies
    Input('user-id-input', 'value')
)
def update_recommendations(selected_user_id):
    if selected_user_id is None:
        return html.P("Please enter a User ID to see recommendations."), ""

    # Get seen movies
    seen_movies_titles = list(refined_dataset[refined_dataset['user id'] == selected_user_id]['movie title'].unique())
    seen_movies_html = [html.Strong("Movies already seen by User "), html.Span(str(selected_user_id) + ":")]
    if seen_movies_titles:
        # Join the list of movies for display, limit to first 10 for brevity
        seen_movies_html.append(html.P(", ".join(seen_movies_titles[:10]) + ("..." if len(seen_movies_titles) > 10 else "")))
    else:
        seen_movies_html.append(html.P("No movies seen by this user in the dataset."))

    # Get DNN recommendations
    recommendations_df = get_dnn_recommendations(selected_user_id, num_recommendations=10) # Get top 10

    if recommendations_df.empty:
        return html.P(f"No recommendations found for User ID '{selected_user_id}'. This might be a new user or all movies are seen."), html.Div(seen_movies_html)
    else:
        rec_list_items = []
        for index, row in recommendations_df.iterrows():
            rec_list_items.append(
                html.Div(className="recommendation-item", children=[
                    html.Span(row['Title'], className="movie-title"),
                    html.Span(f"Score: {row['Predicted Score']:.4f}", className="movie-score")
                ])
            )
        return html.Div(rec_list_items), html.Div(seen_movies_html)


@callback(
    Output('genre-distribution-chart', 'figure'),
    Input('user-id-input', 'value') # Trigger this on user ID change or initial load
)
def update_genre_chart(user_id_for_trigger): # Value isn't used, just for triggering
    # Calculate genre counts from the global `items_dataset` (or `movie_dataset` if genre info is there)
    # To get proper genre distribution, you'd need the genre columns from items_dataset
    # Assuming `items_dataset` is loaded globally for genre info
    global items_dataset # Make sure items_dataset is accessible

    # Extract genre columns (last 19 columns in items_dataset, from 'unknown' to 'Western')
    genre_columns = column_names2[5:] # Adjust index if column_names2 is different

    # Calculate sum of each genre
    genre_counts_raw = items_dataset[genre_columns].sum()
    genre_counts = genre_counts_raw.reset_index()
    genre_counts.columns = ['Genre', 'Count']
    genre_counts = genre_counts[genre_counts['Count'] > 0] # Filter out genres with 0 movies

    fig = px.bar(
        genre_counts,
        x='Genre',
        y='Count',
        title='Distribution of Movies by Genre in Dataset',
        labels={'Genre': 'Movie Genre', 'Count': 'Number of Movies'},
        color='Genre',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        title_font_size=20,
        bargap=0.2,
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#e0e0e0')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)