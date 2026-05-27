import gzip
import json
import random
import pickle
import requests
from transformers import DistilBertTokenizerFast
from utils import create_label_maps, MyDataset

GENRE_URL_DICT = {
    'poetry': 'https://ucsd.edu',
    'children': 'https://ucsd.edu',
    'comics_graphic': 'https://ucsd.edu',
    'fantasy_paranormal': 'https://ucsd.edu',
    'history_biography': 'https://ucsd.edu',
    'mystery_thriller_crime': 'https://ucsd.edu',
    'romance': 'https://ucsd.edu',
    'young_adult': 'https://ucsd.edu'
}

def load_reviews(url, head=10000, sample_size=2000):
    """Streams zipped json data from web sources and samples records."""
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_data(model_name='distilbert-base-cased', max_length=512, sample_cache_path='genre_reviews_dict.pickle'):
    """Main execution block for pulling, splitting, and encoding textual features."""
    # Fetch data or load cached copy
    try:
        with open(sample_cache_path, 'rb') as f:
            genre_reviews_dict = pickle.load(f)
    except FileNotFoundError:
        genre_reviews_dict = {}
        for genre, url in GENRE_URL_DICT.items():
            genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=2000)
        with open(sample_cache_path, 'wb') as f:
            pickle.dump(genre_reviews_dict, f)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    # Train / Test splitting per genre
    for genre, reviews in genre_reviews_dict.items():
        sampled_subset = random.sample(reviews, min(1000, len(reviews)))
        
        train_subset = sampled_subset[:800]
        test_subset = sampled_subset[800:]
        
        for review in train_subset:
            train_texts.append(review)
            train_labels.append(genre)
        for review in test_subset:
            test_texts.append(review)
            test_labels.append(genre)

    # Label Encodings
    label2id, id2label = create_label_maps(train_labels)
    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    # Tokenizer execution
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Pack objects into structural PyTorch Datasets
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset, id2label, test_labels

if __name__ == '__main__':
    prepare_data()
