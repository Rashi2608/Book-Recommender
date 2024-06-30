import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Function to load artifacts
def load_data():
    try:
        model = pickle.load(open('artifacts/model.pkl', 'rb'))
        book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
        final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
        book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
        return model, book_names, final_rating, book_pivot
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

model, book_names, final_rating, book_pivot = load_data()

def fetch_poster(suggestions, final_rating, book_pivot):
    poster_urls = []
    authors = []
    years = []
    isbns = []
    publishers = []

    for suggestion in suggestions[0]:
        book_name = book_pivot.index[suggestion]
        book_index = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[book_index]['image_url']
        author = final_rating.iloc[book_index]['author']
        year = final_rating.iloc[book_index]['year']
        isbn = final_rating.iloc[book_index]['ISBN']
        publisher = final_rating.iloc[book_index]['publisher']

        if not url.strip():
            url = "https://via.placeholder.com/150"  # Placeholder image if no URL is available
        poster_urls.append(url)
        authors.append(author)
        years.append(year)
        isbns.append(isbn)
        publishers.append(publisher)

    return poster_urls, authors, years, isbns, publishers

def recommend_book(book_name, model, book_pivot, final_rating):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
    except IndexError:
        return [], [], [], [], [], []

    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_urls, authors, years, isbns, publishers = fetch_poster(suggestions, final_rating, book_pivot)
    recommended_books = [book_pivot.index[suggestion] for suggestion in suggestions[0]]

    return recommended_books, poster_urls, authors, years, isbns, publishers

def get_top_rated_books(final_rating):
    if final_rating is None:
        print("Final rating data is not loaded.")
        return [], [], [], [], [], []
    
    top_books = final_rating.nlargest(5, 'rating')
    top_books_titles = top_books['title'].values
    top_books_posters = top_books['image_url'].values
    top_books_authors = top_books['author'].values
    top_books_years = top_books['year'].values
    top_books_isbns = top_books['ISBN'].values
    top_books_publishers = top_books['publisher'].values
    return top_books_titles, top_books_posters, top_books_authors, top_books_years, top_books_isbns, top_books_publishers

@app.route('/')
def index():
    if final_rating is None:
        return "Error loading data. Please try again later."
    
    top_books_titles, top_books_posters, top_books_authors, top_books_years, top_books_isbns, top_books_publishers = get_top_rated_books(final_rating)
    return render_template('index.html', book_names=book_names,
                           top_books_titles=top_books_titles,
                           top_books_posters=top_books_posters,
                           top_books_authors=top_books_authors,
                           top_books_years=top_books_years,
                           top_books_isbns=top_books_isbns,
                           top_books_publishers=top_books_publishers)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    selected_book = request.form['book']
    recommended_books, poster_urls, authors, years, isbns, publishers = recommend_book(selected_book, model, book_pivot, final_rating)
    return render_template('recommendation.html',
                           recommended_books=recommended_books,
                           poster_urls=poster_urls,
                           authors=authors,
                           years=years,
                           isbns=isbns,
                           publishers=publishers)


if __name__ == '__main__':
    app.run(debug=True)

