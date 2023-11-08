from google.colab import files
uploaded = files.upload()

pip install bertopic
pip install bertopic[visualization]
pip install bertopic[spacy]


import pandas as pd 
import numpy as np
from bertopic import BERTopic
import pandas as pd 
import io
df = pd.read_csv(io.BytesIO(uploaded['tweets_tokyo.csv']))

# select only 6000 tweets
df = df[0:6000]

model = BERTopic(verbose=True)

#convert to list
docs = df.text.to_list()

topics, probabilities = model.fit_transform(docs)
model.get_topic_freq().head(11)


model.visualize_barchart()

# can also choose the number of topics automatically
# model = BERTopic(nr_topics="auto")


