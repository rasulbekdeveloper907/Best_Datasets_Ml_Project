import pandas as pd
import os
import logging

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        if os.path.exists(self.file_path):
            try:
                df = pd.read_csv(self.file_path)
                logging.info(f"Data loaded successfully. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Error loading the file: {e}")
                return pd.DataFrame()
        else:
            logging.error(f"File not found: {self.file_path}")
            return pd.DataFrame()

# Bu yerda to'g'ridan-to'g'ri ishga tushirilganda bajariladigan qism
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    file_path = r"C:\Users\Rasulbek907\Desktop\Project_MP\Data\Feature_Selection\Filtered_Features.csv"
    loader = DataLoader(file_path)
    data = loader.load_dataset()

    print(data.head())  # Ma'lumotlarning birinchi 5 qatorini ko'rish

