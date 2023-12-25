# Tensorflow 練習
參考課程連結  
1. [(强推)TensorFlow官方入门实操课程](https://www.bilibili.com/video/BV1rz4y117p1?p=1&vd_source=32bf45c0ce711c6b1692fb70dc8e313f)  
2. [从入门到精通！不愧是公认的讲的最好的【Tensorflow全套教程】同济大佬12小时带你从入门到进阶！Tensorflow2.0/计算机视觉/神经网络与深度学习](https://www.bilibili.com/video/BV1FW4y1b7WM/?share_source=copy_web&vd_source=0d8a138a6f09448eb4a244c7f88e76ba)
## Lesson 1 beginning
+ fashion
    * begin.py         一個神經元網路
    * clothes.py       FashionMNIST
    * load_model.py    使用訓練好的模型
    * stop_train.py    自動終止訓練
    * cnn.py           卷積神經網路
## Lesson 2 Human and Hourse
+ human_horse
    * human_horse.py (save模型停不下來 還不知道是什麼問題)
    * new_human_horse.py 
## Lesson 3 cats and dogs
+ cats_dogs
    * classification.py 分類圖片(tmp/PetImages 已分類好)
    * cats_dogs.py
## Lesson 4 手寫字體辨識案例
+  write
    * write.py
## Lesson 5 剪刀石頭布案例
+ rps
    * rps.py
## Lesson 6 自然語言文本處理
+ IMDB
    * words.py    詞條化&序列化
    * sarcasm.py  諷刺數據集的詞條化&序列化
    * movie.py    可視化IMDB分類
    * sarcasm2.py 子詞對分類器的影響(採用不完整子詞訓練->很難學習正確的語意和情感)
    [imdb_reviews/subwords8k版本](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md#imdb_reviewssubwords8k)
    (下載後模型在 C:\Users\christy.hsieh\tensorflow_datasets)  
可視化網站  
https://projector.tensorflow.org/

## Lesson 7 文本生成
+ lyrics
    * lyrics_prac.py
    * lyrics.py  生成詩歌
## Lesson 8 序列、時間序列和預測
ex 可用來預測股票、天氣等等  
+ decompose
    + season.py       時間序列
    + time_series.py  時間序列的預測方法
    + rnn_prac.py     RNN樣本的生成
    + rnn.py          RNN時間序列預測  
    時間序列生成->切分數據集->生成RNN輸入樣本->定義SimpleRNN神經網路->定義參數->訓練->查看誤差曲線->定義改進後的RNN網路->使用RNN網路模型預測時間序列->評估訓練結果
    + lstm.py         雙向LSTM時間序列預測
## Lesson 9 Course2