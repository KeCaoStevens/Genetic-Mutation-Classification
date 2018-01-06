import xml.etree.ElementTree as ET
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
logging.info('xml loading')
tree = ET.parse('C:/Users/DC20693/Downloads/pubmed_result.xml')
root = tree.getroot()
textForm = ''
#notGood = []
count = 0
logging.info('xml loaded')

for i in range(0,len(root)):
    try:
        tempRoot = root[i].find('MedlineCitation').find('Article').find('Abstract')
        count +=1
        try:
            for newRoot in tempRoot:
                textForm +=newRoot.text + '\n'
        except TypeError:
            textForm = textForm
    except AttributeError:
        tempRoot = root[i].find('BookDocument').find('Abstract')
        for newRoot in tempRoot:
            textForm +=newRoot.text + '\n'
    if (i in [0,4,9,99,499,999,1999,4999,9999,14999]):
        logging.info('file loading.' )
        
logging.info('done')

f = open('C:/Users/DC20693/Desktop/kaggle/abstractOut.txt','w',encoding='utf-8')
f.write(textForm)
f.close()
