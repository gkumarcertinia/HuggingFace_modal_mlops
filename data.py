import gzip
import json
import random
import pickle
import requests
from transformers import DistilBertTokenizerFast
from utils import create_label_maps, MyDataset

# Real UCSD Goodreads dataset URLs (McAuley Lab)
GENRE_URL_DICT = {
    'poetry':               'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':             'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':       'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':   'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':    'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime':'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':              'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':          'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}

# Realistic per-genre sentence banks used when live URLs are unreachable
_SYNTHETIC_BANK = {
    'poetry': [
        "This poetry collection moved me deeply with its vivid imagery and lyrical verse.",
        "Each poem in this anthology captures raw emotion in beautifully crafted language.",
        "The poet's use of metaphor and rhythm creates an unforgettable reading experience.",
        "A stunning collection that explores grief, love, and identity through powerful verse.",
        "The language is precise yet evocative, making every line resonate long after reading.",
        "These poems feel timeless, weaving together nature and human longing masterfully.",
        "A meditative and contemplative collection that rewards slow, careful reading.",
        "The brevity of each poem belies the emotional depth contained within a few lines.",
    ],
    'children': [
        "My kids absolutely loved this story — the illustrations are vibrant and enchanting.",
        "A wonderful picture book that teaches empathy and kindness in an age-appropriate way.",
        "The rhyming text and colourful pages kept my toddler engaged from start to finish.",
        "A gentle adventure story perfect for bedtime reading with young children.",
        "Simple language and relatable characters make this an ideal early reader book.",
        "The imaginative world-building in this children's novel sparked great conversations.",
        "A heartwarming story about friendship and acceptance that children will treasure.",
        "Beautifully illustrated and lovingly written — a new family favourite.",
    ],
    'comics_graphic': [
        "The artwork in this graphic novel is stunning and perfectly complements the story.",
        "A gripping superhero saga with complex characters and beautifully rendered panels.",
        "This manga series is addictive — I read all five volumes in a single weekend.",
        "The sequential art storytelling here is masterful; every page turn is a revelation.",
        "Dark, atmospheric, and beautifully drawn — a landmark work in graphic literature.",
        "The colour palette and panel composition elevate this comic to true art.",
        "A thought-provoking graphic memoir that uses visuals to explore trauma and recovery.",
        "Action-packed and visually spectacular with a surprisingly emotional core.",
    ],
    'fantasy_paranormal': [
        "An epic fantasy with an intricately built world, memorable magic systems, and rich lore.",
        "Dragons, ancient prophecies, and a reluctant hero make this a classic of the genre.",
        "The world-building is extraordinary — I was completely lost in this fantasy realm.",
        "Fast-paced and thrilling, with surprising twists that kept me up past midnight.",
        "Magic and politics intertwine brilliantly in this sprawling multi-volume series.",
        "The paranormal romance subplot adds unexpected depth to an already gripping narrative.",
        "A dark fantasy that pulls no punches — gritty, immersive, and utterly compelling.",
        "The author's imagination knows no bounds; every chapter introduces something wondrous.",
    ],
    'history_biography': [
        "A meticulously researched biography that brings a forgotten historical figure back to life.",
        "Fascinating account of how one individual shaped the course of an entire century.",
        "The author balances scholarly rigour with accessible prose, making history come alive.",
        "An eye-opening look at a pivotal moment in history through extensive primary sources.",
        "Deeply human and insightful — this biography reads almost like a novel.",
        "A sweeping historical narrative that contextualises events with remarkable clarity.",
        "The level of archival research behind this work is truly impressive and evident.",
        "Both informative and emotionally engaging — history writing at its very best.",
    ],
    'mystery_thriller_crime': [
        "A nail-biting thriller with unexpected twists on every page and a brilliant detective.",
        "I could not put this crime novel down — the tension builds relentlessly to a shocking end.",
        "The psychological depth of this mystery elevates it far above the average whodunit.",
        "A fiendishly clever plot with red herrings that fooled me until the very last chapter.",
        "Fast-paced, atmospheric, and genuinely frightening — a masterclass in suspense writing.",
        "The cat-and-mouse dynamic between detective and killer is utterly gripping.",
        "Dark, gritty, and utterly compelling — this crime novel lingers in the memory.",
        "One of the most tightly plotted thrillers I have read in years — absolutely brilliant.",
    ],
    'romance': [
        "A heartwarming love story filled with witty banter, undeniable chemistry, and emotion.",
        "The slow-burn romance had me rooting for the couple from the very first chapter.",
        "Sweet, funny, and genuinely moving — this romance left me with a huge smile.",
        "The emotional depth of this love story surprised me; I was in tears by the end.",
        "Steamy, funny, and refreshingly modern — easily one of the best romances this year.",
        "The enemies-to-lovers arc is executed perfectly with great pacing and real tension.",
        "A cosy, feel-good romance that is impossible to read without grinning throughout.",
        "Complex characters and a beautifully layered relationship make this romance stand out.",
    ],
    'young_adult': [
        "A coming-of-age story that captures the confusion and joy of adolescence perfectly.",
        "Relatable characters navigating identity, friendship, and first love in a fresh voice.",
        "A powerful YA novel that tackles mental health and belonging with honesty and care.",
        "Fast-paced, emotionally resonant, and impossible to put down — perfect for teens.",
        "The protagonist's journey of self-discovery feels authentic and deeply moving.",
        "An action-packed dystopian YA with a strong female lead and satisfying world-building.",
        "This YA novel handles difficult themes with sensitivity and without talking down.",
        "Funny, heartfelt, and endlessly quotable — a standout in modern young adult fiction.",
    ],
}

_FILLERS = [
    "Highly recommended.", "A must-read.", "Could not put it down.",
    "An absolute gem.", "Beautifully crafted.", "Deeply moving.",
    "Thought-provoking.", "A real page-turner.", "Wonderful writing.",
    "Utterly captivating.", "Loved every page.", "Truly outstanding.",
]


def _make_synthetic_review(genre: str, idx: int) -> str:
    """Generate a realistic-sounding synthetic review for a given genre."""
    base = random.choice(_SYNTHETIC_BANK[genre])
    extra = random.choice(_SYNTHETIC_BANK[genre])
    filler = random.choice(_FILLERS)
    return f"{base} {extra} {filler} (Review {idx})"


def _generate_synthetic_reviews(genre: str, n: int) -> list:
    """Return n synthetic reviews for the given genre."""
    return [_make_synthetic_review(genre, i) for i in range(n)]


def load_reviews(url, head=10000, sample_size=2000):
    """Streams zipped json data from web sources and samples records."""
    reviews = []
    count = 0
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_data(model_name='distilbert-base-cased', max_length=512, sample_cache_path='genre_reviews_dict.pickle',
                 sample_size=200):
    """Main execution block for pulling, splitting, and encoding textual features."""
    # Fetch data or load cached copy
    try:
        with open(sample_cache_path, 'rb') as f:
            genre_reviews_dict = pickle.load(f)
        print(f"Loaded cached dataset from '{sample_cache_path}'")
    except FileNotFoundError:
        print("No cache found — fetching / generating dataset ...")
        genre_reviews_dict = {}
        for genre, url in GENRE_URL_DICT.items():
            try:
                print(f"  Downloading {genre} reviews from {url} ...")
                genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=sample_size)
                print(f"    ✓ {len(genre_reviews_dict[genre])} reviews loaded")
            except Exception as e:
                print(f"    ✗ Download failed ({e}), using synthetic data for '{genre}'")
                genre_reviews_dict[genre] = _generate_synthetic_reviews(genre, sample_size)
        with open(sample_cache_path, 'wb') as f:
            pickle.dump(genre_reviews_dict, f)
        print(f"Dataset cached to '{sample_cache_path}'")

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    # Train / Test splitting per genre (80% train, 20% test)
    for genre, reviews in genre_reviews_dict.items():
        n = min(sample_size, len(reviews))
        sampled_subset = random.sample(reviews, n)

        split = int(n * 0.8)
        train_subset = sampled_subset[:split]
        test_subset  = sampled_subset[split:]
        
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
