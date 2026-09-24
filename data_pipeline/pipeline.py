import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os

# 1. SCRAPING
print("Scraping books...")
base_url = "http://books.toscrape.com/catalogue/category/books_1/index.html"
books_data = [] # List of dicts: {title, price, rating, availability, category}

# NOTE: You must implement the full scraping loop to get 60+ books across 3+ categories.
# This is a simplified placeholder for structure. 
# Make sure to actually scrape 60 books before submitting!
# Parse: title, price (GBP), rating (One-Five), availability, category.

# 2. CLEANING & CONVERSION
df = pd.DataFrame(books_data)

# Clean price (e.g., "£51.77" -> 51.77)
df['price_gbp'] = df['price'].str.replace('£', '').astype(float)

# Clean rating (e.g., "Three" -> 3)
rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
df['rating'] = df['star_rating'].map(rating_map)

# Clean availability (e.g., "In stock" -> True)
df['in_stock'] = df['availability'].apply(lambda x: 'In stock' in x)

# Convert to INR (FIXED RATE)
df['price_inr'] = df['price_gbp'] * 105.50

# Handle missing values (median imputation for numeric, drop for others)
# --- YOUR MISSING VALUE LOGIC HERE ---

# 3. SQLITE DATABASE
conn = sqlite3.connect('zepto_books.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
)
''')

# Insert categories and books (Use INSERT OR IGNORE for categories)
# --- YOUR INSERT LOGIC HERE ---
conn.commit()

# 4. SQL QUERIES & PANDAS
print("\n--- Running SQL Queries ---")
queries = [
    "SELECT * FROM books LIMIT 5;",
    "SELECT title, price_gbp FROM books WHERE rating >= 4 ORDER BY price_gbp DESC LIMIT 10;",
    "SELECT DISTINCT category_id FROM books;",
    "SELECT title FROM books WHERE price_gbp BETWEEN 10 AND 20;",
    "SELECT b.title, c.category_name FROM books b JOIN categories c ON b.category_id = c.category_id LIMIT 5;"
]

for q in queries:
    print(f"\nQuery: {q}")
    res = pd.read_sql(q, conn)
    print(res)

# Pandas Merge demonstration (Task 6)
print("\n--- Pandas Merge ---")
books_df = pd.read_sql("SELECT * FROM books", conn)
cats_df = pd.read_sql("SELECT * FROM categories", conn)
merged_df = pd.merge(books_df, cats_df, on='category_id')
print(merged_df[['title', 'category_name']].head())

conn.close()
print("\nPipeline complete. Database saved as zepto_books.db")
