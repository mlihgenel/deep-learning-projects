import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

def get_graph(df):
    label_en = LabelEncoder()
    df['emotion_encoded'] = label_en.fit_transform(df['emotion'])

    plt.figure(figsize=(8, 6))
    
    sns.lineplot(x='time', y='emotion_encoded', data=df, marker='o')
    plt.yticks(ticks=range(len(label_en.classes_)), labels=label_en.classes_)
    plt.title("Zamana Bağlı Duygu Değişimi")
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Duygu")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assests/per_second.png')
    plt.show()



def get_daily_report(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='emotion', hue='emotion')
    plt.xlabel('Duygu Tipleri')
    plt.ylabel('Duygu Sayıları')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('assests/daily_report.png')
    plt.show()
    