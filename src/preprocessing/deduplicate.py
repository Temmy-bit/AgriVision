import hashlib
import os
import pandas as pd

def deduplicate(df):
    # Create a new column 'image_hash' to store the hash of each image
    df['image_hash'] = df['ImagePath'].apply(lambda x: hashlib.md5(open(x, 'rb').read()).hexdigest())
    
    # Drop duplicates based on the 'image_hash' column
    deduplicated_df = df.drop_duplicates(subset='image_hash')
    
    # Drop the 'image_hash' column as it's no longer needed
    deduplicated_df = deduplicated_df.drop(columns=['image_hash'])
    print(f"Original dataset size: {len(df)}")
    print(f"Deduplicated dataset size: {len(deduplicated_df)}")
    
    return deduplicated_df


if __name__ == '__main__':
    dir = 'C:\\Users\\HP\\Desktop\\AGRIVISION\\Train.csv'
    train_df = pd.read_csv(dir)
    print(train_df.head())
    
    deduplicated_train_df = deduplicate(train_df)
    
    # Save the deduplicated DataFrame to a new CSV file
    deduplicated_train_df.to_csv(os.path.join(dir, 'Deduplicated_Train.csv'), index=False)