import pandas as pd
from io import StringIO

from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import re
import string

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB


def categoryChecker(data, df):

    # print("here")
    # categories = [
    #     'alt.atheism',
    #     'comp.graphics',
    #     'comp.os.ms-windows.misc',
    #     'comp.sys.ibm.pc.hardware',
    #     'comp.sys.mac.hardware',
    #     'comp.windows.x',
    #     'misc.forsale',
    #     'rec.autos',
    #     'rec.motorcycles',
    #     'rec.sport.baseball',
    #     'rec.sport.hockey',
    #     'sci.crypt',
    #     'sci.electronics',
    #     'sci.med',
    #     'sci.space',
    #     'soc.religion.christian',
    #     'talk.politics.guns',
    #     'talk.politics.mideast',
    #     'talk.politics.misc',
    #     'talk.religion.misc'
    # ]

    # news_group_data = fetch_20newsgroups(
    #     subset="all", remove=("headers", "footers", "quotes"), categories=categories
    # )

    # newspaperDataDF = pd.DataFrame(
    #     dict(
    #         text=news_group_data["data"],
    #         target=news_group_data["target"]
    #     )
    # )
    # newspaperDataDF["target"] = newspaperDataDF.target.map(lambda x: categories[x])

    # newspaperDataDF["clean_text"] = newspaperDataDF.text.map(process_text)

    # df_train, df_test = train_test_split(newspaperDataDF, test_size=0.20, stratify=newspaperDataDF.target)

    # vec = CountVectorizer(
    # ngram_range=(1, 3), 
    # stop_words="english",
    # )

    # X_train = vec.fit_transform(df_train.clean_text)
    # X_test = vec.transform(df_test.clean_text)

    # y_train = df_train.target
    # y_test = df_test.target

    # nb = MultinomialNB()
    # nb.fit(X_train, y_train)

    # preds = nb.predict(X_test)

    # joblib.dump(nb, "nb.joblib")
    # joblib.dump(vec, "vec.joblib")

    nb_saved = joblib.load("nb.joblib")
    vec_saved = joblib.load("vec.joblib")

    columns = customJSONArrayParser(data['columns'])

    for column in columns:
        nameOfNewColumn = 'Prediction of Column " ' + str(column) + '"'
        for index, row in df.iterrows():
            sample_text = [str(row[column])]
            # Process the text in the same way you did when you trained it!
            clean_sample_text = process_text(sample_text)
            sample_vec = vec_saved.transform(sample_text)
            print(nb_saved.predict(sample_vec))
            df.loc[df.index[index], nameOfNewColumn] = nb_saved.predict(sample_vec)



def process_text(text):
    text = str(text).lower()
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", " ", text
    )
    text = " ".join(text.split())
    return text

def customJSONArrayParser(arrayString):
    finalArray = [] 
    while('{"value":"' in arrayString):
        start = arrayString.find('{"value":')
        end = arrayString.find('"}')
        finalArray.append(arrayString[start+10:end])

        arrayString = arrayString[end+3:: ]
    return(finalArray)


def filter(data,df):

    columns = customJSONArrayParser(data['columns'])
    keywords = customJSONArrayParser(data['keywords'])

    for index, row in df.iterrows():
            if(data['boolFilterOut'] == False):
                removeRow = False
            else:
                removeRow = True
            for column in columns:
                for keyword in keywords:
                    if(data['boolFilterOut'] == True and data['boolMatchExact'] == True):
                        if(str(row[column]) == keyword):
                            removeRow = False
                    elif(data['boolFilterOut'] == True and data['boolMatchExact'] == False):
                        if(keyword in str(row[column])):
                            print("here")
                            removeRow = False
                    elif(data['boolFilterOut'] == False and data['boolMatchExact'] == True):
                        if(str(row[column]) == keyword):
                            removeRow = True
                    elif(data['boolFilterOut'] == False and data['boolMatchExact'] == False):
                        if(keyword in str(row[column])):
                            removeRow = True
            if(removeRow == True):
                df.drop([index], inplace=True)


def split(data,df):

    columns = customJSONArrayParser(data['columns'])
    splitter = data['splitters']
    splitcheck1 = splitter.find('[/column]')
    splitcheck2 = splitter.find('[/cell-value]')

    if(splitcheck1 < splitcheck2):
        first = splitter[0:splitter.find('[/column]')]
        second = splitter[splitter.find('[/column]')+9:splitter.find('[/cell-value]')]
        third = splitter[splitter.find('[/cell-value]')+13:len(splitter)]
    elif(splitcheck1 > splitcheck2):
        first = splitter[0:splitter.find('[/cell-value]')]
        second = splitter[splitter.find('[/cell-value]')+13:splitter.find('[/column]')]
        third = splitter[splitter.find('[/column]')+9:len(splitter)]

    if(splitcheck1 != -1 and splitcheck2 != -1):
        for index, row in df.iterrows():
            for column in columns:
                string = str(row[column]).replace('\n', " ")

                while(string != ""):
                    if(first != ""):
                        if(string.find(first) == -1 and (string.find(second) == -1 or second == "")):
                            break
                        start = string.find(first)
                        string = string[start+len(first) :: ]
                    if(second != ""):
                        if(string.find(second) == -1 and (string.find(third) == -1 or third == "")):
                            break
                        start = string.find(second)
                        columnmake = string[0:start]
                        string = string[start+len(second) :: ]
                    if(third != "" and string.find(third) != -1):
                        start = string.find(second)
                        tempstring = string[0:start]
                        end = tempstring.rfind(third)
                        cell = string[0:end]
                        if(splitcheck1 < splitcheck2):
                            df.loc[index, columnmake] = str(cell)
                        else:
                            df.loc[index, str(cell)] = columnmake
                        string = string[0+len(cell)+1 :: ]
                    else:
                        if(string.find(first) == -1 or string.find(second) == -1):
                            cell = string[0 :: ]
                            if(splitcheck1 < splitcheck2):
                                df.loc[index, columnmake] = str(cell)
                            elif(splitcheck1 > splitcheck2):
                                df.loc[index, str(cell)] = columnmake
                            break
                        start = string.find(first)
                        cell = string[0:start]
                        if(splitcheck1 < splitcheck2):
                            df.loc[index, columnmake] = str(cell)
                        elif(splitcheck1 > splitcheck2):
                            df.loc[index, str(cell)] = columnmake
                        string = string[start+len(first) :: ]
    else:
        for index, row in df.iterrows():
            for column in columns:
                columnname = 0
                string = str(row[column]).replace('\n', " ")
                while(string != ""):
                    start = string.find(splitter)
                    if(start == -1):
                        break
                    value = string[0 :start ]
                    string = string[start+len(splitter) :: ]
                    df.loc[index, "default" + str(columnname)] = value
                    columnname += 1


def merge(data,df):
    columns = customJSONArrayParser(data['columns'])
    splitter = data['splitters']
    main_column = columns[0]
    for index, row in df.iterrows():
            for column in columns:
                if(column != main_column):
                    string = str(row[column]).replace('\n', " ")
                    df.loc[index, main_column] = df.at[index,main_column] + splitter + string

    for column in columns:
        if(column != main_column):
            df.drop([column], axis=1, inplace=True)

                    
#Set up Flask
app = Flask(__name__)
cors = CORS(app)

#Run the app
@app.route("/receiver", methods=["POST"])
def postME():
    data = request.get_json()
    converteddata = StringIO(data['theTable'])
    df = pd.read_csv(converteddata, sep=",")
    if(data['function'] == "filterKeywords"):
        filter(data, df)
    if(data['function'] == "Split Columns by Characters"):
        split(data,df)
    if(data['function'] == "Merge Columns by Characters"):
        merge(data,df)
    if(data['function'] == "Predict Column Category"):
        categoryChecker(data,df)

        
    # end
    arrayofarrays = []
    columnarray = []
    for col in df.columns:
        columnarray.append(col)
    arrayofarrays.append(columnarray)

    for index, row in df.iterrows():
        rowarray = []
        for column in columnarray:
            appendthis = str(row[column])
            if(str(row[column]) == "nan"):
                appendthis = ""
            rowarray.append(appendthis)
        arrayofarrays.append(rowarray)


    return jsonify(arrayofarrays)
if __name__ == "__main__": 
    app.run(debug=True)
