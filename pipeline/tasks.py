import luigi
import os
import nltk

class GetKaggleData(luigi.Task):
    def run(self):
        username=input("What is your kaggle username: ")
        password=input("What is your kaggle password: ")
        os.system('kg download -u {username} -p {password} -c cs5785-fall-2017-final'.format(username=username, password=password))
        os.system('mkdir data && mv data.zip data/ && cd data && unzip data.zip')
        
    def output(self):
        return luigi.LocalTarget('data/descriptions_train/0.txt')

class NLTKtarget:
    def exists(self):
        files = os.listdir(nltk.data.find('corpora'))
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        return 'words' in files and 'stopwords' in files

class NLTKrequirements(luigi.Task):
    def run(self):
        nltk.download('words')
        nltk.download('stopwords')
        nltk.download('punkt')

    def output(self):
        return NLTKtarget()

class TokenCounter(luigi.Task):
    def requires(self):
        return [GetKaggleData(),NLTKrequirements()]

    def run(self):
        os.system('python pipeline/sentence_to_tokens.py')

    def output(self):
        pipeline = os.environ['PIPELINE']
        return luigi.LocalTarget('pipeline/data/tokens_{}.json'.format(pipeline))

class TokenVector(luigi.Task):
    def requires(self):
        return TokenCounter()

    def run(self):
        os.system('python pipeline/tokens_to_vec.py')

    def output(self):
        pipeline = os.environ['PIPELINE']
        return luigi.LocalTarget('pipeline/data/tokens_{}.csv'.format(pipeline))

if __name__ == '__main__':
    luigi.run(["--local-scheduler"], main_task_cls=TokenVector)
