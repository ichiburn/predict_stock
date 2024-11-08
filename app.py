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
            'Close': df_stock['Close'],
            'SMA': df_stock['SMA']
        })
        st.line_chart(df_stock2)

        # 値動きグラフ
        st.header(f"{stock_name} 値動き(USD)")
        df_stock['change'] = ((df_stock['Close'] - df_stock['Open']) / df_stock['Open'] * 100)
        st.line_chart(df_stock['change'].tail(100))

        # ローソク足
        fig = go.Figure(
            data=[go.Candlestick(
                x=df_stock.index,
                open=df_stock['Open'],
                high=df_stock['High'],
                low=df_stock['Low'],
                close=df_stock['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )]
        )
        st.header(f"{stock_name} キャンドルスティック")
        st.plotly_chart(fig)

        # 株価予測のための準備
        df_pred = df_stock.copy()
        df_pred['label'] = df_pred['Close'].shift(-30)

        st.header(f'{stock_name} 1か月後を予測しよう（USD）')

        def stock_predict():
            # 予測のための特徴量を準備
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            X = df_pred[feature_columns][:-30]
            y = df_pred['label'][:-30].dropna()

            # データの標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 予測用データの準備
            predict_data = scaler.transform(df_pred[feature_columns][-30:])

            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-30], y, test_size=0.2, random_state=42)

            # モデルの訓練
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 精度の評価
            accuracy = model.score(X_test, y_test)
            st.write(f'正答率は{round(accuracy * 100, 1)}%です。')

            # 信頼度の表示
            if accuracy > 0.75:
                st.write('信頼度：高')
            elif accuracy > 0.5:
                st.write('信頼度：中')
            else:
                st.write('信頼度：低')

            st.write('オレンジの線(Predict)が予測値です。')

            # 予測の実行
            predictions = model.predict(predict_data)

            # 予測値をデータフレームに追加
            future_dates = pd.date_range(
                start=df_pred.index[-1] + pd.Timedelta(days=1),
                periods=30,
                freq='B'  # 営業日のみを使用
            )

            # 予測結果を可視化用に準備
            plot_df = pd.DataFrame({
                'Date': pd.concat([df_pred.index, pd.Series(future_dates)]),
                'Actual': pd.concat([pd.Series(df_pred['Close']), pd.Series([np.nan] * 30)]),
                'Predicted': pd.concat([pd.Series([np.nan] * len(df_pred)), pd.Series(predictions)])
            })

            # Matplotlib でのプロット
            plt.figure(figsize=(15, 6))
            plt.plot(plot_df['Date'], plot_df['Actual'], color='green', label='実際の価格')
            plt.plot(plot_df['Date'], plot_df['Predicted'], color='orange', label='予測価格')
            plt.legend()
            plt.xticks(rotation=45)
            plt.title(f"{stock_name}の株価予測")
            plt.tight_layout()
            
            st.pyplot(plt)
            plt.close()

            # Streamlit用のグラフ
            plot_df.set_index('Date', inplace=True)
            st.line_chart(plot_df[['Actual', 'Predicted']])

        # 予測ボタン
        if st.button('予測する'):
            stock_predict()

except Exception as e:
    st.error(
        f"エラーが発生しました：{e}"
    )

st.write('Copyright © 2021 Tomoyuki Yoshikawa. All Rights Reserved.')
