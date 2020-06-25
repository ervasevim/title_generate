# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from database.database import database

# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
db = database()
db.select(["pre_title", "pre_content", "is_test"])
db.where("is_test = true")
db.table("eng_texts_tab")
test_datas = db.get_result()
test_titles = []
test_contents = []

for item in test_datas:
    str1 = " "
    str2 = " "
    test_titles.append( str1.join(item[0]) )
    test_contents.append( str2.join(item[1]) )

print(test_contents[1])
# predicted = model.predict( test_contents[1] )