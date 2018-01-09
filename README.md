# EmoryChat

Emotional neural conversational model with memory

## Getting Started

### Spacy's large English model
```
python -m spacy download en_core_web_lg
```

### Fasttext
```
wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
rm v0.1.0.zip
cd fastText-0.1.0
make
```

### Data

#### Emotion

After you download and extract all of the following corpora run src/data/emotion_data_parsers.py 

##### Hashtag Emotion Corpus
Download [Hashtag Emotion Corpus](http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip)
and extract to data/raw/emotion/Jan9-2012-tweets-clean.txt.

##### Crowdflower's The Emotion in Text
Download [The Emotion in Text](https://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv)
and extract to data/raw/emotion/text_emotion.csv.

##### Affective Text SemEval 2007
Download [Affective Text](http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz)
and extract to data/raw/emotion/AffectiveText.Semeval.2007.

##### Electoral/Political tweets annotated for sentiment, emotion, purpose and style
Download [Electoral/Political tweets annotated for sentiment, emotion, purpose and style](http://saifmohammad.com/WebDocs/ElectoralTweetsData.zip)
and extract to data/raw/emotion/ElectoralTweetsData.

##### WASSA-2017 Shared Task on Emotion Intensity (EmoInt)
Download [WASSA-2017 Shared Task on Emotion Intensity (EmoInt)](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)
and extract to data/raw/emotion/Wassa-2017.

##### Collections of love letters, hate mail, and suicide notes
Download [Collections of love letters, hate mail, and suicide notes](http://saifmohammad.com/WebDocs/LoveHateSuicide.tar.gz)
and extract to data/raw/emotion/LoveHateSuicide/love-letters.txt.

##### Movie reviews, annotated for emotion classification
Clone [Movie reviews, annotated for emotion classification](https://github.com/NLeSC/spudisc-emotion-classification)
to data/raw/emotion/spudisc-emotion-classification-master

##### NRC Emotion Lexicon
Get [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
and extract to data/raw/emotion/NRC-Sentiment-Emotion-Lexicons

#### Dialogue

##### Cornell Movie-Dialogs Corpus
Download [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
and extract to data/raw/dialogue/cornell movie-dialogs corpus.

Reformat to csv via src/data/movie_corpus_extraction.py

##### Ubuntu Dialogue Corpus v2.0
Clone [Ubuntu Dialogue Corpus v2.0](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

Translate create_ubuntu_dataset.py to python 3. Set positive example probability to 1.
Generate corpus via generate.sh. Reformat to csv via src/data/ubuntu_corpus_extraction.py

##### Microsoft Research Social Media Conversation Corpus 
Download [Microsoft Research Social Media Conversation Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52375&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F6096d3da-0c3b-42fa-a480-646929aa06f1%2F)
and extract to data/raw/dialogue/MSRSocialMediaConversationCorpus.

This dataset only has tweet IDs, so create a Twitter application to access its API.
Put your ConsumerToken, ConsumerSecret, AccessToken, AccessSecret into config.ini in the following format
```
[twitter]
ConsumerToken = abc
ConsumerSecret = abc
AccessToken = abc
AccessSecret = abc
```
Run src/data/microsoft_corpus_tweets_extraction.py to extract tweet texts.

##### Reddit comments
Download a [month of Reddit comments](https://www.kaggle.com/data/31657)
and extract to data/raw/dialogue/reddit_comments_month. 

Create utterances via src/data/reddit_comments_extraction.py.

### Installing

Run all cells in notebooks/exploration/1.0-rsh-emotion-data.ipynb to generate a combined dataset with 
a reduced number of classes.
Find the best hyperparameters for fasttext via src/models/fasttext_hypertuning.py. 
Run emotion classification training on the whole corpus with src/models/fasttext_training.py
Prepare dialogue data for fasttext through src/data/prepare_for_fasttex.py.
Run utterances emotion classification via src/models/fasttext_inference.py and create the final 
emotion dialogue dataset by running src/data/merge_with_labels.py

## Running the tests

Explain how to run the automated tests for this system

## Authors

* **Roman Shaptala** - *Everything* - [LinkedIn](https://www.linkedin.com/in/romanshaptala/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Zhou, Hao, et al. "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory." arXiv preprint arXiv:1704.01074 (2017). [[PDF]](https://arxiv.org/pdf/1704.01074)
* Ghazvininejad, Marjan, et al. "A Knowledge-Grounded Neural Conversation Model." arXiv preprint arXiv:1702.01932 (2017). [[PDF]](https://arxiv.org/pdf/1702.01932)