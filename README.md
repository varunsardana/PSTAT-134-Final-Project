# PSTAT 134 Final Project
## Linguistic Variation in Subreddit Communities

### Authors
Mounami Kayitha, Varun Sardana, Prithvi Rao, Gabor Szita, Sristhi Thapar

### Project Overview
This project investigates whether different Reddit communities use measurably
different language when discussing AI topics. We collect posts from four
subreddits using the Reddit API (PRAW), then apply NLP techniques including
TF-IDF and classification models to predict which subreddit a post came from.

### Subreddits
- r/technology
- r/artificial
- r/singularity
- r/MachineLearning

### Repo Structure
```
project/
├── data/
│   ├── raw/            # raw data straight from PRAW
│   └── cleaned/        # cleaned and preprocessed data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_tfidf_topic_modeling.ipynb
│   └── 04_classification.ipynb
├── .env.example        # template for API credentials
├── requirements.txt    # python dependencies
└── README.md

```

### Setup
1. Clone the repo
2. Run: pip install -r requirements.txt
3. Copy .env.example to .env and fill in your Reddit API credentials
4. Run notebooks in order 01 to 04

### Important
Never commit your .env file — it contains your API credentials.


