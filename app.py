from flask import Flask
from flask import Flask, request, jsonify, render_template
from Polarity_and_Frequency import *
from simillarity import *
from subjectivity import *
import pandas as pd
from gensim.summarization import summarize
from pos_tagging import *
from word_cloud import *

app = Flask(__name__)
def say_hello(username = "World"):
    return '<h1>FE 595 Mid Term Project</h1>'

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>FE 595 Mid Term Project</title> </head>\n<body>'''
instructions = '''
    <p><a href="/services">Check out the NLP services!</a></p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '<p><b>Made by Akshat Goel, Giovanni Scalzotto, Francesco Forner, Kevin Shah</b></p> </body>\n</html>'

@app.route('/services', methods=['POST', 'GET'])
def services():
    if request.method == 'POST':
        form_data = request.form
        if form_data['service'] == 'polarity_and_frequency':
            pol_score = polarity(form_data['content'])
            freq_results = frequency(form_data['content'])
            return render_template('pol_and_freq.html', pol_score = pol_score, freq_results = freq_results)
        elif form_data['service'] == 'similarity':
            word1sim, word2sim = similarity(form_data['content'],form_data['word1'],form_data['word2'])
            word1sim = pd.DataFrame(word1sim, columns = ['Word','Similarity score with word1'])
            word1sim = word1sim.sort_values(by=['Similarity score with word1'],ascending=False)
            word2sim = pd.DataFrame(word2sim, columns = ['Word','Similarity score with word2'])
            word2sim = word2sim.sort_values(by=['Similarity score with word2'], ascending=False)
            return render_template('similarity.html',table1=[word1sim.to_html(classes='data')], title1=word1sim.columns.values,table2=[word2sim.to_html(classes='data')], title2=word2sim.columns.values )
        elif form_data['service'] == 'word_cloud': #Akshat
            word_freqs, max_freq = word_cloud_generator(form_data['content'])
            return render_template('word_cloud.html',word_freqs = word_freqs, max_freq = max_freq )
        elif form_data['service'] == 'pos_tagging':#Akshat
            result = pos_tagging(form_data['content'])
            return render_template('pos_tagging.html', data = result)
        elif form_data['service'] == 'subjectivity': #Giovanni
            sub = subjectivity(form_data['content'])
            return render_template('subjectivity.html', data1 = sub, content = form_data['content'])
        elif form_data['service'] == 'summarization':#Kevin
            valerr = None
            try:
                summary = summarize(form_data['content'], ratio=0.2)
            except ValueError as e:
                valerr = str(e)
            if valerr != None:
                return render_template('summarizationerr.html', err = valerr)
            else:
                return render_template('summarization.html',data1 = summary, content = form_data['content'])
    if request.method == 'GET':
        return render_template("imput.html")

# add a rule for the index page.
app.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug = True)