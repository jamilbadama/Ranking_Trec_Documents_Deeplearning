# Ranking of 2018 Precision Medicine Track using attention-based deep learning 

Install the required packages

$ pip install -r requirements.txt
 

 
 # Ranking of 2018 Precision Medicine Track using attention-based deep learning 

This code used the contextualized word embedding of the documents and user queries combined with genetic information to find contextual similarity using BM25 based search engine for determining the relevancy score to rank the articles
 

# Instructions
This code used Python 3.7 version.

**Step 1**: Install the required Python packages: 

```
pip3 install -r requirements.txt
```

**Step 2**: Download the dataset(s) you intend to use and re-create embedding from Precision Medicine Tracks

To Classify the dataset based on targert diseases using different deep learning models given in the code by simply train the model and use it for classification.

For example to train CNN model just run the  
```
python train_cnn_model.py

```
Then classify the new data using trained model

**Step 3**: Run the Query on pretrained embedding using 2018 Precision Medicine Tracks dataset

To get the documents ranking based proposed method, run the function with parameters 

'''
query_ranking("melanoma", 1, "64-year-old male", "BRAF (V600E)", 10)
'''
topic: Target topic such as melanoma
index: Query Number such as 1
query: Query such as 64-year-old male
gene: Gene data such as BRAF (V600E)
topk: Number of top document you want to return



