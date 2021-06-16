import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from attacut import tokenize
import pandas as pd

regexp = r"[ก-๙a-zA-Z']+"
img = np.zeros((512, 512, 3), np.uint8)
img[::] = [255, 255, 255]
mask = np.array(Image.open("white.png"))
text = ["สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ "
        "น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ "
        "ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ "
        "หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ "
        "ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย "
        "สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ "
        "น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ "
        "ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ "
        "หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ "
        "ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย "
        "สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ "
        "น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ "
        "ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ "
        "หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ "
        "ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย "
        "สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ "
        "น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ "
        "ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย สักวา หวาน อื่น มี หมื่น แสน ไม่ เหมือน แม้น พจมาน ที่ "
        "หวาน หอม กลิ่น ประเทียบ เปรียบ ดวง พวงพะยอม อาจ จะ น้อม จิต โน้ม ด้วย โลม ลม แม้น ล้อ ลาม หยาม หยาบ ไม่ "
        "ปลาบปลื้ม ดัง ดูด ดื่ม บอระเพ็ด ต้อง เข็ด ขม ผู้ดี ไพร่ ไม่ ประกอบ ชอบ อารมณ์ ใคร ฟัง ลม เมิน หน้า ระอา เอย "]

vectorized = CountVectorizer(tokenizer=tokenize)
tfid = TfidfVectorizer(use_idf=False)
df = pd.DataFrame(columns=['name', 'count'])
transform_data = vectorized.fit_transform(text)
count = np.ravel(transform_data.sum(axis=0))
vector_name = vectorized.get_feature_names()

df['count'] = count
df['name'] = vector_name

data = df.sort_values(by=['count'], ascending=False)
data = data.drop(0)
data.plot(kind='bar', figsize=(12, 6))
word_dict = {}
for i in range(1, len(data)):
    word_dict[data.name[i]] = data['count'][i]

font_path = 'THSarabunNew.ttf'
wordcloud = WordCloud(
    font_path=font_path,
    relative_scaling=0.3,
    min_font_size=1,
    background_color="white",
    width=1024,
    height=768,
    max_words=2000,
    colormap='plasma',
    scale=3,
    font_step=4,
    #   contour_width=3,
    #   contour_color='steelblue',
    collocations=False,
    regexp=regexp,
    margin=2
).fit_words(word_dict)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
fig.show()
