import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    data = bytes_data.decode("utf-8")

    print(data)

    df = preprocessor.preprocess(data)

    print(df)
    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        blob =TextBlob(data)
        sentiment = blob.sentiment.polarity # -1 to 1
        print(sentiment)

        st.header("Sentiment Percentage")
        st.title(sentiment)

        st.header("Fake News Analysis")
        news_data = pd.read_csv("fake_news.csv")

        news_data['fake'] = news_data ['label'].apply(lambda x: 0 if x == "REAL" else 1)

        news_data = news_data.drop ("label",axis=1)

        x,y = news_data ['text'],news_data['fake']

        x_train, x_test, y_train,y_test =train_test_split(x,y,test_size=0.2)
        vectorizer = TfidfVectorizer(stop_words="english",max_df = 0.7)
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        # print(y_train)
        clf = LinearSVC()

        clf.fit(x_train_vectorized, y_train)

        clf.score(x_test_vectorized, y_test)

        vectorized_text = vectorizer.transform([data])

        if (clf.predict(vectorized_text)[0]==1):
            st.title("Chat Contains Fake News")
        else:
            st.title("Chat Does not Contains Fake News")
        # # emoji analysis
        # emoji_df = helper.emoji_helper(selected_user,df)
        # st.title("Emoji Analysis")

        # col1,col2 = st.columns(2)

        # with col1:
        #     st.dataframe(emoji_df)
        # with col2:
        #     fig,ax = plt.subplots()
        #     ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        #     st.pyplot(fig)











