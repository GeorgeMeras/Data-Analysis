import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re
from transformers import pipeline,AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#Roberta pretrain model for sentiment analysis {fear,joy,anger..}
emotion_model = pipeline('text-classification',model='j-hartmann/emotion-english-distilroberta-base',top_k=None)

#Roberta pretrain model for sentiment analysis pos,neu,neg
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
sent_pipeline = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)

def pos(score):
    if score >= 0.05:
        return 'Positive'
    elif score < 0.05 and score >= -0.05:
        return 'Neutral'
    else:
        return 'Negative'

def cleanText(text):
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'www\.\S+',' ',text)
    text = re.sub(r'#\w+',' ',text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

def Roberta_polarity_score(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    res_pretrain = {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}
    return res_pretrain


df = pd.read_csv('./vaccination_tweets.csv')

sia = SentimentIntensityAnalyzer()
res = {}
res_rob = {}
roberta_res = {}
for i,row in tqdm(df.iterrows(), total=len(df)):
    text = cleanText(row['text'])
    myid = row['id']
    res[myid] = sia.polarity_scores(text)
    res_rob[myid] = emotion_model(text)
    roberta_res[myid] = Roberta_polarity_score(text)

robert_df = pd.DataFrame(res_rob).T
robert_df = robert_df.reset_index().rename(columns={'index':'id',0:'emotion'})
robert_df['main_emotion'] = robert_df['emotion'].apply(lambda x: max(x,key=lambda item: item['score'])['label'])

pf = pd.DataFrame(res).T
pf = pf.reset_index().rename(columns={'index':'id'})
pf['sentiment'] = pf['compound'].apply(pos)
pf = pf.merge(df, how='left', on='id')

label_map = {
    'roberta_neg':'NEGATIVE',
    'roberta_neu':'NEUTRAL',
    'roberta_pos':'POSITIVE',
}
pretrain_df = pd.DataFrame(roberta_res).T
pretrain_df = pretrain_df.reset_index().rename(columns={'index':'id'})
pretrain_df['predicted_sentiment'] = pretrain_df[['roberta_neg','roberta_neu','roberta_pos']].idxmax(axis=1)
pretrain_df['predicted_sentiment'] = pretrain_df['predicted_sentiment'].map(label_map)
pretrain_df = pretrain_df.merge(pf, how='left', on='id')
robert_df = robert_df.merge(pretrain_df, how='left', on='id')

robert_df.to_excel('Roberta-Vader_emotion_model.xlsx',index=False)

roberta_melt = robert_df.melt(id_vars=['main_emotion'],value_vars=['roberta_neg','roberta_neu','roberta_pos'],var_name='Sentiment Type',value_name='Score')
sns.boxplot(data=roberta_melt, x='Sentiment Type', y='Score',hue='main_emotion')
plt.title("Main emotion per Roberta's-Category Sentiment")
plt.show()

pf['sentiment'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%',title='Entire Text-Vader Sentiment Distribution')
plt.show()

vader_val = pd.crosstab(robert_df['main_emotion'],robert_df['sentiment'],normalize='index')*100
sns.heatmap(data=vader_val, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title("Correlation Between Vader-Sentiment and Roberta's-Emotion")
plt.show()

pretrain_df['predicted_sentiment'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%',title='Predicted Sentiment Distribution')
plt.show()

ax = sns.countplot(x='main_emotion', data=robert_df , hue='main_emotion')
for i in ax.patches:
    count = i.get_height()
    percent = count * 100 / len(robert_df)
    x = i.get_x()+i.get_width()/2
    y = i.get_height()
    ax.text(x, y,f'{percent:.1f}%', horizontalalignment='center', verticalalignment='bottom')
plt.title('Type of Sentiment')
plt.show()
