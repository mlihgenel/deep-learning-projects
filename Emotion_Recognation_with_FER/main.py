from get_emotions import get_emotion
import pandas as pd
from graph import get_graph, get_daily_report

print("======= EMOTION RECOGNATION ========")

print("How do you wanne recognation emotions...\n")
print("1 - Webcam ")
print("2 - Video File")
print("====================================")
choice = input("Please make choice...")

if choice == "1":
    choice = 0
elif choice == "2":
    print("Dosya yolunu giriniz...")
    choice = input()


if __name__ == '__main__':
    result = get_emotion(choice)
    df = pd.DataFrame(result)
    df.to_csv("analysis.csv", index=False)
    get_graph(df)
    get_daily_report(df)
    