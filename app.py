import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn. compose import ColumnTransformer
from sklearn. preprocessing import LabelEncoder, OrdinalEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import openai
openai.organization = "org-JXv1tLwAM68xdQrrpST7tbeS"
openai.api_key = "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"



app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ip = [x for x in request.form.values()]
    message=ip.pop(0)
    analyzer = SentimentIntensityAnalyzer()
    sentence = message
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence,str(vs)))
    neg = vs['neg']
    neu = vs['neu']
    pos = vs['pos']
    ip[0]=int(ip[0])
    data = pd.read_csv("survey.csv")
    data.drop(['Country', 'state', 'Timestamp','treatment', 'comments'], axis = 1, inplace = True)
    data.loc[len(data.index)] = ip
    data['self_employed'].fillna('No', inplace = True)
    data['work_interfere'].fillna('N/A', inplace = True)
    data.drop(data[(data['Age'] > 80) | (data['Age'] < 18)].index, inplace = True)
    data['Gender'].replace(['Male', 'male','M','m','Make','Cis Male','Man','Malr','Cis Man','msle','cis male'
                        'Mail','Guy (-ish) ^_^','Male (CIS)','Male-ish','maile','Mal', 'Mail', 'Male ', 'cis male'], 'Male', inplace = True)
    data['Gender'].replace(['Female', 'female','F','f','Woman','Female (cis)','cis-female/femme','femail','Cis Female','Femake',
                            'woman','Female '], 'Female', inplace = True)
    data['Gender'].replace(['Female (trans)','Trans woman','male leaning androgynous','Neuter','queer','Guy (-ish) ^_^',
                            'Enby','Agender','Trans-female','something kinda male?','queer/she/they', 'Androgyne', 'non-binary',
                            'Nah','fluid','Genderqueer','ostensibly male, unsure what that really means'], 'Non-binary', inplace = True)
    ct = ColumnTransformer([('oe', OrdinalEncoder(),['Gender','self_employed','family_history','work_interfere','no_employees',
                        'remote_work','tech_company','benefits','care_options','wellness_program', 'seek_help', 'anonymity',
                        'leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview',
                        'phys_health_interview','mental_vs_physical', 'obs_consequence'])], remainder = 'passthrough')
    x = ct.fit_transform(data)
    fnl = x[-1]
    print(x)
    print("final value of x: \n",fnl)
    final = [np.array(fnl)]
    prediction = model.predict(final)

    output = prediction[0]
    print(output)
   
    model_id = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
    model=model_id,
    messages=[
    {"role": "user", "content": "mental health quote"}
    ]
    )
    yeah=completion.choices[0].message.content
    if neg>pos and output==1:
        completion = openai.ChatCompletion.create(
        model=model_id,
        messages=[
        {"role": "user", "content": "mental health quote"}
        ]
        )
        yeah=completion.choices[0].message.content
        return render_template('resultyes.html',message=yeah)
    else:
        completion = openai.ChatCompletion.create(
        model=model_id,
        messages=[
        {"role": "user", "content": "success quote"}
        ]
        )
        yeah=completion.choices[0].message.content
        return render_template('resultno.html',message=yeah)

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/community')
def community():
    return render_template('community.html')

if __name__=='__main__':
    app.run(debug=True)
