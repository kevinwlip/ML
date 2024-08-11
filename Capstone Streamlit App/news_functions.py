import pandas as pd
import datetime
import urllib.request
from bs4 import BeautifulSoup
from random import sample
from transformers import pipeline

# Print Timestamp At time of crawl
def news_date():
   news_date = str(datetime.date.today().strftime('%A, %B %d, %Y'))
   return news_date

# Get page and parse its contents
def news_scraper():
   url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
   page = urllib.request.urlopen(url)
   soup = BeautifulSoup(page, "html.parser")

   try:
      business_heading = soup.find('title')
      if 'Business' in business_heading.get_text():
         print("Google News - Business Section Found.")
   except:
      raise Exception("Was not able to navigate to the Google News - Business section.")

   a_tags = soup.find_all('a')
   business_titles = [tag.get_text() for tag in a_tags if len(tag.get_text().split()) >= 5]
   ten_titles = sample(business_titles,10)
   print(ten_titles)
   return ten_titles

# Format the output
def output_format(headlines = news_scraper()):

   current_headlines = headlines

   model1_results = []
   model2_results = []
   model3_results = []
   sentiment_models = [pipeline(model="kevinwlip/financial-sentiment-model-5000-samples"), pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"), pipeline("sentiment-analysis", model="ProsusAI/finbert")]
   for i, model in enumerate(sentiment_models):
      for j, headline in enumerate(current_headlines):
         if i == 0:
               model1_results.append(sentiment_models[i]([headline]))
         elif i == 1:
               model2_results.append(sentiment_models[i]([headline]))
         elif i == 2:
               model3_results.append(sentiment_models[i]([headline]))

   model1_vals = [item.values() for result in model1_results for item in result]
   model1_output = [f"{x.capitalize()}, Probability: {y}" for x,y in model1_vals if x ]
   model1_output = [s.replace("Label_0", "Negative").replace("Label_1", "Positive").replace("Label_2", "Neutral") for s in model1_output]

   model2_vals = [item.values() for result in model2_results for item in result]
   model2_output = [f"{x.capitalize()}, Probability: {y}" for x,y in model2_vals]

   model3_vals = [item.values() for result in model3_results for item in result]
   model3_output = [f"{x.capitalize()}, Probability: {y}" for x,y in model3_vals]

   # Dataframes for DistilRoberta Financial News Sentiment Model
   model1_df1 = pd.DataFrame([model1_output[:5]], columns = current_headlines[:5])
   model1_df2 = pd.DataFrame([model1_output[5:]], columns = current_headlines[5:])

   # Dataframes for Prosus AI Finbert Financial Sentiment Model
   model2_df1 = pd.DataFrame([model2_output[:5]], columns = current_headlines[:5])
   model2_df2 = pd.DataFrame([model2_output[5:]], columns = current_headlines[5:])

   # Dataframes for Kip's Self-Trained Model Financial Sentiment Model
   model3_df1 = pd.DataFrame([model3_output[:5]], columns = current_headlines[:5])
   model3_df2 = pd.DataFrame([model3_output[5:]], columns = current_headlines[5:])

   return model1_df1, model1_df2, model2_df1, model2_df2, model3_df1, model3_df2