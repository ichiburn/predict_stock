import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import yfinance as yf

st.title("AIで株価予測アプリ")
st.write('AIを使って、株価を予測してみましょう')

# トップ画像の表示
try:
    image = Image.open('stock_predict.png')
    st.image(image)
except FileNotFoundError:
    st.warning('stock_predict.png が見つかりません。画像の表示をスキップします。')

st.write('※あくまでAIによる予測です（参考値）。こちらのアプリによる損害や損失は一切補償しかねます。')

st.header("株価銘柄のティッカーシンボルを入力してください。")
stock_name = st.text_input("例：AAPL, FB, SFTBY（大文字・小文字どちらでも可）", "AAPL")

stock_name = stock_name.upper()

link = 'https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('ティッカーシンボルについては上のリンク（SBI証券）をご参照ください。')

try:
    # データの取得
    df_stock = yf.download(stock_name, start='2022-01-05')
    
    if df_stock.empty:
        st.error(f"データを取得できませんでした。ティッカーシンボル {stock_name} が正しいか確認してください。")
    else:
        st.header(f"{stock_name} 2022年1月5日から現在までの価格(USD)")
        st.write(df_stock)

        # 移動平均の計算と表示
        st.header(f"{stock_name} 終値と14日間平均(USD)")
        df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
        df_stock2 = pd.DataFrame({
        'Close': df_stock['Close'].values.flatten(),  # .flatten()を追加
        'SMA': df_stock['SMA'].values.flatten()       # .flatten()を追加
        }, index=df_stock.index)
        st.line_chart(df_stock2)

        # 値動きグラフ
        st.header(f"{stock_name} 値動き(USD)")
        df_stock['change'] = (((df_stock['Close'] - df_stock['Open'])) / df_stock['Open'] * 100)
        st.line_chart(df_stock['change'].tail(100))

# ローソク足
        fig = go.Figure(
            data=[go.Candlestick(
                x=df_stock.index,
                open=df_stock['Open'].values.flatten(),   # .values.flatten() を追加
                high=df_stock['High'].values.flatten(),   # .values.flatten() を追加
                low=df_stock['Low'].values.flatten(),     # .values.flatten() を追加
                close=df_stock['Close'].values.flatten(), # .values.flatten() を追加
                increasing_line_color='green',
                decreasing_line_color='red'
            )]
        )
        fig.update_layout(                               # レイアウトの設定を追加
            title=f'{stock_name}のローソク足チャート',
            yaxis_title='株価 (USD)',
            xaxis_title='日付'
        )
        st.header(f"{stock_name} キャンドルスティック")
        st.plotly_chart(fig)

        # 株価予測のための準備
        df_stock['label'] = df_stock['Close'].shift(-30)

        st.header(f'{stock_name} 1か月後を予測しよう（USD）')

def stock_predict():
            # 予測のための特徴量を準備
            # 欠損値を削除してから特徴量を準備
            df_pred = df_stock.dropna()  # 追加: 欠損値を持つ行を削除
            
            X = np.array(df_pred.drop(['label', 'SMA', 'change'], axis=1))
            X = StandardScaler().fit_transform(X)
            predict_data = X[-30:]
            X = X[:-30]
            y = np.array(df_pred['label'].dropna()).flatten()
            y = y[:-30]

            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # モデルの訓練
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 精度の評価
            accuracy = model.score(X_test, y_test)
            # 少数第一位で四捨五入
            st.write(f'正答率は{round(accuracy * 100, 1)}%です。')

            # 信頼度の表示
            if accuracy > 0.75:
                st.write('信頼度：高')
            elif accuracy > 0.5:
                st.write('信頼度：中')
            else:
                st.write('信頼度：低')

            st.write('オレンジの線(Predict)が予測値です。')

            # 検証データを用いて検証してみる
            predicted_data = model.predict(predict_data)
            df_stock['Predict'] = np.nan
            last_date = df_stock.iloc[-1].name
            one_day = 86400
            next_unix = last_date.timestamp() + one_day

            # 予測のグラフ化
            for data in predicted_data:
                next_date = datetime.datetime.fromtimestamp(next_unix)
                next_unix += one_day
                df_stock.loc[next_date] = np.append([np.nan] * (len(df_stock.columns)-1), data)

            df_stock['Close'].plot(figsize=(15, 6), color="green")
            df_stock['Predict'].plot(figsize=(15, 6), color="orange")
            plt.legend(['実際の価格', '予測価格'])
            st.pyplot(plt)
            plt.close()
            
            # Streamlit用のグラフ
            df_stock3 = df_stock[['Close', 'Predict']]
            st.line_chart(df_stock3)
            
        # 予測ボタン
        if st.button('予測する'):
            stock_predict()

except Exception as e:
    st.error(
        f"エラーが発生しました：{e}"
    )

st.write('Copyright © 2021 Tomoyuki Yoshikawa. All Rights Reserved.')
