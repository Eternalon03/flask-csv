import pandas as pd
from io import StringIO

from flask import Flask, request, jsonify
from flask_cors import CORS

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

    print(splitcheck1)
    print(splitcheck2)

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
                        print(columnmake)
                        string = string[start+len(second) :: ]
                    if(third != "" and string.find(third) != -1):
                        start = string.find(second)
                        tempstring = string[0:start]
                        end = tempstring.rfind(third)
                        cell = string[0:end]
                        print(cell)
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
                        print(cell)
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

    print(df)
        
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
