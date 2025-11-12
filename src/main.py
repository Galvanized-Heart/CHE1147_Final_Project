import config
import dataset

def main() -> None:
    dataset.download_and_clean_data()

if __name__ == "__main__":
    cleaned_df = main()